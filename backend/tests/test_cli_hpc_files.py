from __future__ import annotations

import base64
from types import SimpleNamespace

from yourtool import cli


class FallbackOnlyClient:
    connected = True
    use_sftp = False
    sftp = None

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.commands: list[str] = []

    def run_command(self, cmd: str, timeout: int = 60, **_kwargs):
        self.commands.append(cmd)
        return base64.b64encode(self.payload).decode("ascii"), "", 0


def test_hpc_read_file_uses_base64_fallback_without_scp_get(monkeypatch) -> None:
    client = FallbackOnlyClient(b"alpha\nbeta\n")
    monkeypatch.setattr(cli, "_HPC_CLIENT", client)

    result = cli._cmd_hpc_read_file(
        SimpleNamespace(path="/remote/results/summary.txt", max_mb=1),
        None,
    )

    assert result == {
        "ok": True,
        "type": "text",
        "mime": "text/plain",
        "text": "alpha\nbeta\n",
        "size": 11,
    }
    assert client.commands == ['head -c 1048577 "/remote/results/summary.txt" | base64']


def test_hpc_read_html_returns_local_html_from_fallback(monkeypatch) -> None:
    html = b"<html><body><div>plotly</div></body></html>"
    client = FallbackOnlyClient(html)
    monkeypatch.setattr(cli, "_HPC_CLIENT", client)

    result = cli._cmd_hpc_read_file(
        SimpleNamespace(path="/remote/a/report.html", max_mb=1),
        None,
    )

    assert result["ok"] is True
    assert result["type"] == "local_html"
    assert result["mime"] == "text/html"
    assert result["local_path"].endswith("_report.html")
    with open(result["local_path"], "rb") as fh:
        assert fh.read() == html
    assert client.commands == ['cat "/remote/a/report.html" | base64']


def test_hpc_download_file_uses_base64_fallback_without_scp_get(monkeypatch, tmp_path) -> None:
    client = FallbackOnlyClient(b"downloaded bytes")
    monkeypatch.setattr(cli, "_HPC_CLIENT", client)
    local = tmp_path / "nested" / "out.bin"

    result = cli._cmd_hpc_download_file(
        SimpleNamespace(remote_path="/remote/results/out.bin", local_path=str(local)),
        None,
    )

    assert result == {"ok": True, "local_path": str(local)}
    assert local.read_bytes() == b"downloaded bytes"
    assert client.commands == ['cat "/remote/results/out.bin" | base64']
