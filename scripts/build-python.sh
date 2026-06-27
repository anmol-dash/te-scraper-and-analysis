#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${ROOT}/backend"
DEST="${ROOT}/src-tauri/binaries"

mkdir -p "${DEST}"

cd "${BACKEND}"
PY="${PYTHON:-python3}"
"${PY}" -m PyInstaller --noconfirm pyinstaller.spec

# macOS universal build: PyInstaller (with target_arch=universal2) emits one fat
# binary. `tauri build --target universal-apple-darwin` needs both the per-arch
# sidecars (its build.rs checks each triple while compiling that arch) AND the
# fat -universal-apple-darwin one (copied into the bundle). Provide all three.
if [[ "${PYINSTALLER_TARGET_ARCH:-}" == "universal2" ]]; then
  archs="$(lipo -archs "dist/pytool" 2>/dev/null || true)"
  if [[ "${archs}" != *x86_64* || "${archs}" != *arm64* ]]; then
    echo "error: expected a universal2 (x86_64 + arm64) sidecar, got: '${archs}'." >&2
    echo "       The build Python must be a universal2 build for target_arch=universal2." >&2
    exit 1
  fi
  cp "dist/pytool" "${DEST}/pytool-universal-apple-darwin"
  lipo "dist/pytool" -thin x86_64 -output "${DEST}/pytool-x86_64-apple-darwin"
  lipo "dist/pytool" -thin arm64 -output "${DEST}/pytool-aarch64-apple-darwin"
  chmod 0755 "${DEST}/pytool-universal-apple-darwin" \
    "${DEST}/pytool-x86_64-apple-darwin" "${DEST}/pytool-aarch64-apple-darwin"
  echo "Installed universal + per-arch sidecars from universal2 binary (${archs}):"
  echo "  ${DEST}/pytool-universal-apple-darwin"
  echo "  ${DEST}/pytool-x86_64-apple-darwin"
  echo "  ${DEST}/pytool-aarch64-apple-darwin"
  exit 0
fi

# Native build: name the sidecar after the rustc host triple.
TRIPLE="$(rustc -vV 2>/dev/null | grep '^host:' | awk '{print $2}')"
if [[ -z "${TRIPLE}" ]]; then
  echo "error: could not read host triple from \`rustc -vV\` (need '^host:' line)" >&2
  exit 1
fi

if compgen -G "dist/pytool.exe" >/dev/null 2>&1; then
  cp "dist/pytool.exe" "${DEST}/pytool-${TRIPLE}.exe"
  chmod 0755 "${DEST}/pytool-${TRIPLE}.exe"
  echo "Installed ${DEST}/pytool-${TRIPLE}.exe"
else
  cp "dist/pytool" "${DEST}/pytool-${TRIPLE}"
  chmod 0755 "${DEST}/pytool-${TRIPLE}"
  echo "Installed ${DEST}/pytool-${TRIPLE}"
fi
