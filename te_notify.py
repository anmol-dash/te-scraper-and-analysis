"""Shared email notification utility for GAMECA pipeline subscripts.

Sends via the Resend HTTPS API (routed through the cluster's HTTPS proxy if
one is configured), which works on nodes that block raw SMTP sockets.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

_API_KEY = "re_VNAgkap7_KFTasPNHnQeMu3QuED4iDDtW"
_FROM    = "GAMECA <no-reply@anmol-dash.com>"


def _detect_proxy() -> str | None:
    for v in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY",
              "all_proxy", "ALL_PROXY"):
        x = os.environ.get(v)
        if x:
            return x.strip()
    home = Path.home()
    for rc in (home / ".curlrc",):
        if rc.exists():
            for raw in rc.read_text(errors="ignore").splitlines():
                ln = raw.strip()
                if not ln or ln.startswith("#"):
                    continue
                if ln.lower().startswith(("proxy", "-x", "--proxy")):
                    for sep in ("=", " "):
                        if sep in ln:
                            c = ln.split(sep, 1)[1].strip().strip('"').strip("'")
                            if c:
                                return c if "://" in c else "http://" + c
    cc = home / ".condarc"
    if cc.exists():
        txt = cc.read_text(errors="ignore")
        for key in ("https:", "http:"):
            if key in txt:
                seg = txt.split(key, 1)[1].splitlines()[0].strip().strip('"').strip("'")
                if seg.startswith(("http://", "https://")):
                    return seg
    for wg in (home / ".wgetrc", Path("/etc/wgetrc")):
        if wg.exists():
            for raw in wg.read_text(errors="ignore").splitlines():
                ln = raw.strip()
                if ln.lower().startswith(("https_proxy", "http_proxy")) and "=" in ln:
                    c = ln.split("=", 1)[1].strip().strip('"').strip("'")
                    if c:
                        return c if "://" in c else "http://" + c
    return None


def send_completion_email(
    to: str,
    script_name: str,
    out_dir: str,
    summary: str = "",
) -> None:
    """Send a job-done notification via the Resend HTTPS API.

    No-op (with a printed warning) if `to` is empty or the API key is missing.
    """
    if not to or not _API_KEY:
        return

    subject = f"GAMECA complete: {script_name}"
    lines = [
        f"Pipeline step '{script_name}' finished successfully.",
        "",
        f"Output directory: {out_dir}",
    ]
    if summary:
        lines += ["", summary]
    body = "\n".join(lines)

    proxy = _detect_proxy()
    handler = (
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        if proxy
        else urllib.request.ProxyHandler({})
    )
    opener = urllib.request.build_opener(handler)

    payload = json.dumps({
        "from": _FROM,
        "to": [to],
        "subject": subject,
        "text": body,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        method="POST",
        headers={
            "Authorization": "Bearer " + _API_KEY,
            "Content-Type": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        },
    )

    try:
        with opener.open(req, timeout=45) as r:
            resp = json.loads(r.read().decode())
        if resp.get("id"):
            print(f"  [notify] Completion email sent to {to} (id={resp['id']})")
        else:
            print(f"  [notify] Resend returned no id: {resp}")
    except urllib.error.HTTPError as e:
        print(f"  [notify] Email send failed (HTTP {e.code}): {e.read().decode(errors='ignore')[:200]}")
    except Exception as exc:
        print(f"  [notify] Email send failed: {exc}")
