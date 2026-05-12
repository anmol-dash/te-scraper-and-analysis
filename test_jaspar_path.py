#!/usr/bin/env python3
"""
Test whether a JASPAR BED/BED.GZ path is valid on the remote HPC cluster.

Usage:
    python test_jaspar_path.py /remote/path/to/JASPAR.bed.gz
    python test_jaspar_path.py --host cluster.edu --user myuser /remote/path/to/JASPAR.bed.gz
"""

import argparse
import getpass
import sys

try:
    import paramiko
except ImportError:
    print("Error: paramiko is required.  pip install paramiko")
    sys.exit(1)


def _run(transport, cmd):
    ch = transport.open_session()
    ch.exec_command(cmd)
    out = ch.makefile("rb").read().decode(errors="replace").strip()
    err = ch.makefile_stderr("rb").read().decode(errors="replace").strip()
    ch.close()
    return out, err


def connect(hostname, username, password, port=22):
    password_store = [password]

    def kb_handler(title, instructions, prompt_list):
        responses = []
        for i, (prompt, _) in enumerate(prompt_list):
            p = prompt.lower()
            if i == 0 or "password" in p:
                responses.append(password_store[0])
            elif "passcode" in p or "option" in p or "duo" in p or "factor" in p:
                print("  [Duo] Sending push (approve on your phone)…", flush=True)
                responses.append("1")
            else:
                responses.append(password_store[0])
        return responses

    print(f"Connecting to {hostname}:{port} as {username}…")
    transport = paramiko.Transport((hostname, port))
    transport.banner_timeout = 60
    transport.connect()
    try:
        transport.auth_interactive(username, kb_handler)
    except paramiko.ssh_exception.AuthenticationException:
        transport.auth_password(username, password)
    if not transport.is_authenticated():
        print("Authentication failed.")
        sys.exit(1)
    print("Connected.")
    return transport


def check_remote(transport, remote_path):
    print(f"\nChecking remote path: {remote_path}\n")
    ok = True

    # existence + size
    out, _ = _run(transport, f"stat -c '%s' {remote_path!r} 2>&1 || echo MISSING")
    if "MISSING" in out or "No such file" in out:
        print("  FAIL  file does not exist on the cluster")
        return False
    try:
        size = int(out)
    except ValueError:
        print(f"  FAIL  could not read file size (got: {out!r})")
        return False
    print(f"  size  {size:,} bytes")
    if size == 0:
        print("  FAIL  file is 0 bytes — download likely incomplete or wrong path")
        return False
    if size < 64:
        print(f"  FAIL  file too small ({size} bytes)")
        return False
    print("  ok    file exists and is non-empty")

    # gzip magic check for .gz files
    if remote_path.endswith(".gz"):
        out, _ = _run(transport, f"xxd -l 2 {remote_path!r} 2>/dev/null || od -A n -N 2 -t x1 {remote_path!r}")
        if "1f 8b" in out or "1f8b" in out.replace(" ", ""):
            print("  ok    gzip magic bytes present")
        else:
            # fallback: check with file command
            out2, _ = _run(transport, f"file {remote_path!r}")
            if "gzip" in out2.lower():
                print("  ok    gzip confirmed via 'file' command")
            else:
                print(f"  FAIL  .gz extension but not a gzip file (file says: {out2})")
                ok = False

        # HTML check (bad download page)
        out, _ = _run(transport, f"zcat {remote_path!r} 2>/dev/null | head -c 16 | cat -v")
        if "<html" in out.lower() or "<!" in out:
            print("  FAIL  decompressed content is HTML — URL returned an error page")
            ok = False
    else:
        out, _ = _run(transport, f"head -c 16 {remote_path!r} | cat -v")
        if "<html" in out.lower() or "<!" in out:
            print("  FAIL  file content is HTML — not a BED file")
            ok = False

    if not ok:
        return False

    # BED format check: sample first 500 data lines, count columns
    if remote_path.endswith(".gz"):
        sample_cmd = f"zcat {remote_path!r} 2>/dev/null | grep -v '^#\\|^track' | head -500"
    else:
        sample_cmd = f"grep -v '^#\\|^track' {remote_path!r} | head -500"

    out, _ = _run(transport, sample_cmd + " | awk '{print NF}' | sort | uniq -c | sort -rn | head -5")
    if not out:
        print("  FAIL  no data lines found (empty after stripping headers)")
        return False

    print(f"  column count distribution (count freq): {out.replace(chr(10), '  ')}")
    # parse dominant column count
    first_line = out.strip().splitlines()[0].split()
    try:
        modal_cols = int(first_line[1])
    except (IndexError, ValueError):
        modal_cols = 0
    if modal_cols < 3:
        print(f"  FAIL  modal column count {modal_cols} < 3 — not a valid BED file")
        return False
    print(f"  ok    modal column count = {modal_cols} (valid BED)")

    # Quick chromosome sanity check
    if remote_path.endswith(".gz"):
        first_cmd = f"zcat {remote_path!r} 2>/dev/null | grep -v '^#\\|^track' | head -3"
    else:
        first_cmd = f"grep -v '^#\\|^track' {remote_path!r} | head -3"
    out, _ = _run(transport, first_cmd)
    print(f"\n  First data lines:\n    " + out.replace("\n", "\n    "))

    print("\n  PASS  file looks like a valid JASPAR BED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test a JASPAR BED path on the HPC cluster")
    parser.add_argument("jaspar_path", help="Remote path to the JASPAR BED/BED.GZ file")
    parser.add_argument("-H", "--host", default="", help="HPC hostname")
    parser.add_argument("-u", "--user", default="", help="HPC username")
    parser.add_argument("-p", "--port", type=int, default=22)
    args = parser.parse_args()

    hostname = args.host or input("  Hostname: ").strip()
    username = args.user or input("  Username: ").strip()
    password = getpass.getpass("  Password: ")

    transport = connect(hostname, username, password, args.port)
    try:
        ok = check_remote(transport, args.jaspar_path)
    finally:
        transport.close()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
