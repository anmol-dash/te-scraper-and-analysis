#!/usr/bin/env python3
"""
check_server.py — preflight check: can this machine run ui.py?

Stdlib-only (runs *before* requirements.txt is installed) so you can vet a
candidate server first. Checks Python version, system binaries the pipeline
shells out to, RAM/disk/CPU, outbound network access (UCSC/Dfam/GitHub), and
whether the dashboard port is free.

Usage:
    python3 scripts/check_server.py
    python3 scripts/check_server.py --port 9000
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from urllib.request import urlopen

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_SYMBOL = {OK: "\033[32m✓\033[0m", WARN: "\033[33m!\033[0m", FAIL: "\033[31m✗\033[0m"}

_results: list[tuple[str, str, str]] = []  # (status, label, detail)


def _record(status: str, label: str, detail: str = ""):
    _results.append((status, label, detail))
    line = f"  {_SYMBOL[status]} {label}"
    if detail:
        line += f"  — {detail}"
    print(line)


# ── checks ────────────────────────────────────────────────────────────────────

def check_python():
    v = sys.version_info
    if (v.major, v.minor) >= (3, 11):
        _record(OK, "Python version", f"{v.major}.{v.minor}.{v.micro}")
    elif (v.major, v.minor) >= (3, 10):
        _record(WARN, "Python version", f"{v.major}.{v.minor}.{v.micro} (3.11+ recommended)")
    else:
        _record(FAIL, "Python version", f"{v.major}.{v.minor}.{v.micro} — need 3.11+")


def check_binaries():
    # Required by query.py / te_prep.py pipeline stages
    required = ["bedtools", "mafft", "git", "gcc"]
    optional = ["liftOver", "tabix", "bgzip", "wget", "curl"]
    for name in required:
        path = shutil.which(name)
        _record(OK if path else FAIL, f"binary: {name}", path or "not on PATH")
    for name in optional:
        path = shutil.which(name)
        _record(OK if path else WARN, f"binary: {name} (optional)",
                path or "not on PATH — degrades gracefully")


def check_memory():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    gb = kb / (1024 * 1024)
                    if gb >= 12:
                        _record(OK, "RAM", f"{gb:.1f} GB total")
                    elif gb >= 8:
                        _record(WARN, "RAM", f"{gb:.1f} GB total (12 GB+ recommended for genome cache)")
                    else:
                        _record(FAIL, "RAM", f"{gb:.1f} GB total — GenomeCache alone needs ~4 GB")
                    return
    except FileNotFoundError:
        pass
    # macOS fallback
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
        gb = int(out.stdout.strip()) / (1024 ** 3)
        if gb >= 12:
            _record(OK, "RAM", f"{gb:.1f} GB total")
        elif gb >= 8:
            _record(WARN, "RAM", f"{gb:.1f} GB total (12 GB+ recommended for genome cache)")
        else:
            _record(FAIL, "RAM", f"{gb:.1f} GB total — GenomeCache alone needs ~4 GB")
    except Exception:
        _record(WARN, "RAM", "could not determine — check manually (need 12 GB+)")


def check_cpu():
    n = os.cpu_count() or 0
    if n >= 4:
        _record(OK, "CPU cores", str(n))
    elif n > 0:
        _record(WARN, "CPU cores", f"{n} (4+ recommended — clustering/alignment stages parallelize)")
    else:
        _record(WARN, "CPU cores", "could not determine")


def check_disk():
    try:
        usage = shutil.disk_usage(".")
        free_gb = usage.free / (1024 ** 3)
        if free_gb >= 20:
            _record(OK, "Free disk space", f"{free_gb:.1f} GB")
        elif free_gb >= 8:
            _record(WARN, "Free disk space", f"{free_gb:.1f} GB (genome FASTA ~3.1 GB + rmsk ~150 MB + results)")
        else:
            _record(FAIL, "Free disk space", f"{free_gb:.1f} GB — too little for genome + results")
    except OSError as exc:
        _record(WARN, "Free disk space", f"could not determine ({exc})")


def check_network():
    targets = [
        ("UCSC (hgdownload.soe.ucsc.edu)", "https://hgdownload.soe.ucsc.edu/"),
        ("Dfam (dfam.org)", "https://dfam.org/"),
        ("GitHub (github.com)", "https://github.com/"),
        ("PyPI (pypi.org)", "https://pypi.org/"),
    ]
    for label, url in targets:
        try:
            with urlopen(url, timeout=8) as resp:
                _record(OK, f"network: {label}", f"HTTP {resp.status}")
        except Exception as exc:
            _record(FAIL, f"network: {label}", f"unreachable ({exc.__class__.__name__})")


def check_port(port: int):
    host = "0.0.0.0"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            _record(OK, f"dashboard port {port}", "free to bind")
        except OSError as exc:
            _record(WARN, f"dashboard port {port}", f"in use or blocked ({exc}) — set GAMECA_PORT to override")


def check_packages():
    # Mirrors requirements.txt; only reports presence (pip install handles the rest)
    pkgs = [
        "numpy", "pandas", "scipy", "statsmodels", "requests", "bs4",
        "dash", "plotly", "matplotlib", "sklearn", "umap", "hdbscan",
        "numba", "Bio", "pysam", "paramiko", "yaml", "Cython",
    ]
    missing = []
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if not missing:
        _record(OK, "Python packages", "all importable (requirements.txt already installed)")
    else:
        _record(WARN, "Python packages", f"not yet installed: {', '.join(missing)} "
                                          f"— run `pip install -r requirements.txt`")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Check whether this server can run ui.py")
    parser.add_argument("--port", type=int, default=8765,
                        help="dashboard port to test (default: GAMECA_PORT/8765)")
    parser.add_argument("--skip-network", action="store_true",
                        help="skip outbound connectivity checks")
    args = parser.parse_args()

    print("\nGAMECA server preflight check")
    print("=" * 40)

    print("\n[Python & packages]")
    check_python()
    check_packages()

    print("\n[System binaries]")
    check_binaries()

    print("\n[Hardware]")
    check_cpu()
    check_memory()
    check_disk()

    print("\n[Network]")
    if args.skip_network:
        print("  (skipped --skip-network)")
    else:
        check_network()

    print("\n[Dashboard]")
    check_port(args.port)

    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]

    print("\n" + "=" * 40)
    if fails:
        print(f"  {_SYMBOL[FAIL]}  Cannot run ui.py as-is — {len(fails)} blocking issue(s):")
        for _, label, detail in fails:
            print(f"      - {label}: {detail}")
        sys.exit(1)
    elif warns:
        print(f"  {_SYMBOL[WARN]}  Can probably run ui.py, with {len(warns)} caveat(s) — see above.")
        sys.exit(0)
    else:
        print(f"  {_SYMBOL[OK]}  This server looks ready to run ui.py.")
        sys.exit(0)


if __name__ == "__main__":
    main()
