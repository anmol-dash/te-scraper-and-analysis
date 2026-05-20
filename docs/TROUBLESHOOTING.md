# Troubleshooting

## “Sidecar not found” / worker exits immediately

**Likely causes**

- **Target triple mismatch:** Built PyInstaller/Linux artifact on x86_64 but running on aarch64 (or mixed musl/glibc). Rust `#[cfg(target_os)]` sidecar paths must match **exactly** the triple you ship (`x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`, etc.).
- **Missing bundled binary:** Resource paths wrong — `resource_dir()` vs dev `PATH`. Confirm `tauri.conf.json > bundle > externalBin` (or equivalent) lists the executable name **without** extension where required.
- **Permissions:** Unix `chmod +x` missing on extracted helper.

**Fix:** Run `pnpm tauri build` on the target OS or cross-compile consistently; print resolved sidecar path at debug log level; verify first-message `ping` succeeds.

---

## Python crashes on launch (frozen EXE / PyInstaller)

**Symptoms:** Blank window, immediate exit code non-zero, `ImportError` in stderr capture.

**Likely cause:** Dynamic imports hidden from PyInstaller analysis (`__import__`, lazy imports inside optional deps).

**Fix:**

- Add **`hiddenimports`** in the `.spec` file for every dynamically imported module (and vendor `.pyz` collections if used).
- Use `--collect-all` for problematic packages or runtime hooks (`hook-yourtool.py`).
- Re-run **onedir** builds with console enabled once to capture tracebacks (`console=True`), then revert for release.

---

## Updater reports “No update available” right after a release

Checklist:

1. **CDN / GitHub caching:** Release assets and `latest.json` (or update endpoint JSON) may be cached. Bump cache headers or wait; fetch JSON with `curl -sSL` from a neutral network.
2. **Wrong `pubkey`:** The updater `pubkey` in app config must match the **private key** pair used to sign the update artifact. Rotate keys deliberately — mismatched pubkey ⇒ verification fails silently or rejects manifest.
3. **Signature mismatch:** Signed payload vs uploaded asset differs (re-uploaded binary without re-signing). Re-run signer on the exact bytes users download.
4. **Version comparison:** Installer version (`Cargo.toml` / tauri.conf) must be **semver lower** than published JSON or endpoint returns no-op.

---

## macOS says the app “is damaged” and won’t open

**Cause:** Gatekeeper expects Apple-notarized Developer ID-signed bundles for distribution outside the Mac App Store. Unsigned or **ad-hoc** signed builds trigger quarantine refusal (“damaged”).

**Mitigations:**

- **Developer ID Application** cert + **`codesign`** all nested binaries (including Python sidecar) + **`notarytool submit`** stapling.
- Document **`xattr -cr`** only for trusted local dev builds — do **not** ship that as end-user guidance.

---

## Windows SmartScreen “Unknown publisher” / blocked install

**Cause:** Binaries lack an Authenticode certificate customers trust, or cert has **no reputation** yet.

**Mitigations:**

- Purchase an **EV code-signing** cert when possible (instant SmartScreen reputation vs OV lag).
- Sign **both** installer and bundled helpers (`signtool sign`).
- Expect warnings during early releases; publish checksums on GitHub Releases and instruct enterprises only if needed.

---

## IPC “invalid_json” / silence after command

- Ensure **UTF-8** NDJSON lines without embedded raw newlines inside JSON strings from UI.
- Respect **`MAX_LINE_BYTES`** on thin bridge (`256 KiB`).
- Worker protocol (`main.py`) requires **`command`** key — sending `method` by mistake yields `UNKNOWN_COMMAND`.
