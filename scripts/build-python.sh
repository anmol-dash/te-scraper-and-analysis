#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${ROOT}/backend"
DEST="${ROOT}/src-tauri/binaries"

mkdir -p "${DEST}"

cd "${BACKEND}"
PY="${PYTHON:-python3}"

# Install the app's bundle deps (certifi, paramiko + crypto stack). These must be
# importable at build time for PyInstaller to collect them; certifi previously
# rode on the runner's ambient install, but paramiko is not ambient.
"${PY}" -m pip install -r requirements.txt

# universal2: paramiko's compiled deps install arch-specific on Apple Silicon
# runners, which would leave the x86_64 slice of the fat sidecar unable to
# import them. Make every crypto extension fat before PyInstaller embeds it.
#   - cryptography still ships a universal2 wheel  -> force it.
#   - cffi 2.x dropped universal2 wheels           -> lipo a fat _cffi_backend
#                                                     from the two per-arch wheels.
#   - bcrypt / pynacl already ship universal2       -> nothing to do.
# (pynacl imports cffi's _cffi_backend at runtime, so cffi genuinely must be fat.)
if [[ "${PYINSTALLER_TARGET_ARCH:-}" == "universal2" ]]; then
  SITE="$("${PY}" -c 'import site; print(site.getsitepackages()[0])')"

  "${PY}" -m pip install --force-reinstall --no-deps --only-binary=:all: --upgrade \
    --platform macosx_11_0_universal2 --target "${SITE}" cryptography

  # Fat cffi: download both per-arch wheels, lipo their _cffi_backend together,
  # and overwrite the arch-specific copy pip installed into site-packages.
  CFFI_TMP="$(mktemp -d)"
  "${PY}" -m pip download --no-deps --only-binary=:all: --python-version 311 --implementation cp \
    --platform macosx_11_0_arm64  -d "${CFFI_TMP}/arm64" cffi
  "${PY}" -m pip download --no-deps --only-binary=:all: --python-version 311 --implementation cp \
    --platform macosx_10_15_x86_64 -d "${CFFI_TMP}/x86_64" cffi
  ( cd "${CFFI_TMP}/arm64"  && unzip -oq cffi-*.whl -d ../a )
  ( cd "${CFFI_TMP}/x86_64" && unzip -oq cffi-*.whl -d ../x )
  arm_so="$(find "${CFFI_TMP}/a" -name '_cffi_backend*.so' | head -1)"
  x86_so="$(find "${CFFI_TMP}/x" -name '_cffi_backend*.so' | head -1)"
  dest_so="${SITE}/$(basename "${arm_so}")"
  lipo -create "${arm_so}" "${x86_so}" -output "${dest_so}"
  echo "Built fat cffi: $(lipo -archs "${dest_so}") -> ${dest_so}"

  # Gate: the sidecar is onefile, so its embedded .so's can't be lipo-checked
  # after the build. Assert every crypto extension is fat *before* PyInstaller
  # embeds it — an arm64-only lib crashes the x86_64 slice at `import paramiko`,
  # exactly the failure this release fixes.
  bad=0
  while IFS= read -r so; do
    a="$(lipo -archs "${so}" 2>/dev/null || echo '?')"
    echo "  bundled: ${a}  $(basename "${so}")"
    [[ "${a}" == *x86_64* && "${a}" == *arm64* ]] || { echo "error: ${so} is not universal2 (${a})" >&2; bad=1; }
  done < <(find "${SITE}" -path '*cryptography*_rust*.so' -o -path '*_cffi_backend*.so' \
                          -o -path '*bcrypt*_bcrypt*.so' -o -path '*nacl*_sodium*.so')
  [[ "${bad}" -eq 0 ]] || { echo "error: crypto stack is not fully universal2; Intel slice would crash on import." >&2; exit 1; }
fi

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
