# Tauri updater feed (`latest.json`)

## Update signing keys (separate from code signing)

Tauri’s updater verifies artifacts with a **dedicated Ed25519 key pair**. Code-signing certificates (Apple, Windows) do not replace this.

### Generate keys

From the project root (with the Tauri CLI available):

```bash
pnpm tauri signer generate -w ~/.tauri/myapp.key
```

The CLI prints the **public** key and writes the **private** key to the path you pass to `-w`. Keep the private key offline except in CI.

### Private key in GitHub Actions

Store the private key material as a repository secret:

| Secret | Purpose |
|--------|---------|
| `TAURI_SIGNING_PRIVATE_KEY` | Contents of the `.key` file **or** its path-dependent raw key string (see [upstream docs](https://v2.tauri.app/plugin/updater/)) |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | Passphrase for the key, if you set one when generating (use empty string if none) |

Reference both in `.github/workflows/release.yml` when running `pnpm tauri build` so bundles and `.sig` files are produced.

### Public key in the app

Put the **public** key in `src-tauri/tauri.conf.json` under the updater plugin (not a path—the PEM/content string itself):

```json
{
  "plugins": {
    "updater": {
      "pubkey": "<paste public key content>",
      "endpoints": [
        "https://github.com/OWNER/REPO/releases/latest/download/latest.json"
      ]
    }
  }
}
```

If you rotate keys, ship a new app build with the new `pubkey` before publishing updates signed with the new private key.

## Static JSON schema (Tauri 2)

The updater expects a JSON document when using a static URL (e.g. GitHub Releases). Required shape is compatible with:

```json
{
  "version": "1.2.3",
  "notes": "Markdown release notes",
  "pub_date": "2026-05-16T12:00:00Z",
  "platforms": {
    "darwin-x86_64": {
      "signature": "<contents of the .sig file>",
      "url": "https://github.com/OWNER/REPO/releases/download/v1.2.3/YourApp_1.2.3_x64.app.tar.gz"
    },
    "darwin-aarch64": {
      "signature": "...",
      "url": "..."
    },
    "windows-x86_64": {
      "signature": "...",
      "url": "..."
    },
    "linux-x86_64": {
      "signature": "...",
      "url": "..."
    }
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `version` | Yes | SemVer (with or without leading `v` in the string). |
| `platforms.<target>` | Yes (per shipped platform) | Keys are `OS-ARCH`, e.g. `linux-x86_64`, `windows-x86_64`, `darwin-x86_64`, `darwin-aarch64`. |
| `platforms.<target>.url` | Yes | HTTPS URL to the update bundle (NSIS `.exe`, `.app.tar.gz`, or `.AppImage` depending on OS). |
| `platforms.<target>.signature` | Yes | **Literal** signature text from the `.sig` artifact—not a path or URL. |
| `notes` | No | Shown to users (often Markdown). |
| `pub_date` | No | RFC 3339 timestamp string. |

Tauri validates signatures and the payload before applying semver logic; every listed platform entry must be complete and valid.

## Feed URL pattern on GitHub

Stable channel, always pointing at the newest release asset:

`https://github.com/OWNER/REPO/releases/latest/download/latest.json`

Clients always resolve `latest` to the newest **non-prerelease** GitHub release. Prereleases are skipped, so betas do not replace this file unless you publish them as latest or use a different URL.

Upload **`latest.json` at the root of each release’s assets** (same relative path for every tag) so the `latest/download/latest.json` URL stays correct.

### Per-channel feeds (stable vs beta)

If you need separate trains later (not implemented in this repo’s workflow), typical patterns are:

- **Dedicated tags / branches**: e.g. `https://github.com/OWNER/REPO/releases/download/beta-feed/latest-beta.json` updated only from a beta pipeline.
- **Different JSON file names** per channel, each wired in a different `endpoints` entry or runtime `updater_builder().endpoints(...)` configuration.

Document the channel policy where you expose download links; only automate extra channels when you have a clear signing and release policy.
