"""Shared email notification utility for GAMECA pipeline subscripts.

Reads Gmail credentials from ~/.hpc_te_state.json (same file used by hpc_client.py).
Set up credentials once via option [g] in hpc_client.py / ui.py.
"""

from __future__ import annotations

import json
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

_STATE_FILE = Path.home() / ".hpc_te_state.json"
_APP_PASSWORD_KEY = "gmail_app_password"
_SENDER_EMAIL_KEY = "gmail_sender_email"


def send_completion_email(to: str, script_name: str, out_dir: str, summary: str = "") -> None:
    """Send a job-done notification via Gmail SMTP.

    No-op (with a printed warning) if credentials are not stored or `to` is empty.
    """
    if not to:
        return

    try:
        state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}

    sender = state.get(_SENDER_EMAIL_KEY, "").strip()
    pw = state.get(_APP_PASSWORD_KEY, "").strip()

    if not sender or not pw:
        print(
            f"  [notify] Email credentials not found — run option [g] in ui.py to set up Gmail.\n"
            f"           Skipping notification to {to}."
        )
        return

    subject = f"GAMECA complete: {script_name}"
    lines = [
        f"Pipeline step '{script_name}' finished successfully.",
        f"",
        f"Output directory: {out_dir}",
    ]
    if summary:
        lines += ["", summary]

    msg = MIMEText("\n".join(lines))
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, pw)
            smtp.send_message(msg)
        print(f"  [notify] Completion email sent to {to}")
    except smtplib.SMTPAuthenticationError:
        print(
            "  [notify] Gmail auth failed — run option [g] in ui.py to update your credentials."
        )
    except Exception as exc:
        print(f"  [notify] Email send failed: {exc}")
