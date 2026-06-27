#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${ROOT}/backend"
DEST="${ROOT}/src-tauri/binaries"

# SIDECAR_TRIPLE overrides the output suffix (e.g. universal-apple-darwin for a
# macOS universal build); defaults to the rustc host triple for native builds.
TRIPLE="${SIDECAR_TRIPLE:-$(rustc -vV 2>/dev/null | grep '^host:' | awk '{print $2}')}"
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

# For universal builds, fail fast unless the binary really carries both slices.
if [[ "${PYINSTALLER_TARGET_ARCH:-}" == "universal2" ]]; then
  archs="$(lipo -archs "${DEST}/pytool-${TRIPLE}" 2>/dev/null || true)"
  if [[ "${archs}" != *x86_64* || "${archs}" != *arm64* ]]; then
    echo "error: expected a universal2 (x86_64 + arm64) sidecar, got: '${archs}'." >&2
    echo "       The build Python must be a universal2 build for target_arch=universal2." >&2
    exit 1
  fi
  echo "Verified universal2 sidecar (${archs})"
fi
