#!/usr/bin/env bash
# Rebuild the Python sidecar binary and copy it into src-tauri/binaries/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[sidecar] Clearing stale .pyc cache…"
find "$ROOT" -name "*.pyc" -path "*/__pycache__/*" -delete 2>/dev/null || true

echo "[sidecar] Building pytool with PyInstaller…"
cd "$ROOT/backend"
python -m PyInstaller pyinstaller.spec --noconfirm

echo "[sidecar] Copying binary to src-tauri/binaries/"
cp "$ROOT/backend/dist/pytool" "$ROOT/src-tauri/binaries/pytool-aarch64-apple-darwin"
echo "[sidecar] Done — $(du -sh "$ROOT/src-tauri/binaries/pytool-aarch64-apple-darwin" | cut -f1)"
