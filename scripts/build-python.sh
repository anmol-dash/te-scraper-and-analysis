#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${ROOT}/backend"
DEST="${ROOT}/src-tauri/binaries"

TRIPLE="$(rustc -vV 2>/dev/null | grep '^host:' | awk '{print $2}')"
if [[ -z "${TRIPLE}" ]]; then
  echo "error: could not read host triple from \`rustc -vV\` (need '^host:' line)" >&2
  exit 1
fi

mkdir -p "${DEST}"

cd "${BACKEND}"
PY="${PYTHON:-python3}"
"${PY}" -m PyInstaller --noconfirm pyinstaller.spec

if compgen -G "dist/pytool.exe" >/dev/null 2>&1; then
  cp "dist/pytool.exe" "${DEST}/pytool-${TRIPLE}.exe"
  chmod 0755 "${DEST}/pytool-${TRIPLE}.exe"
  echo "Installed ${DEST}/pytool-${TRIPLE}.exe"
else
  cp "dist/pytool" "${DEST}/pytool-${TRIPLE}"
  chmod 0755 "${DEST}/pytool-${TRIPLE}"
  echo "Installed ${DEST}/pytool-${TRIPLE}"
fi
