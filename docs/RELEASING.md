# Releasing

Human checklist for shipping a version with GitHub Releases and the Tauri updater.

## Prerequisites

- [Tauri updater](https://v2.tauri.app/plugin/updater/) configured in `src-tauri/tauri.conf.json` (`bundle.createUpdaterArtifacts`, `plugins.updater.pubkey`, `plugins.updater.endpoints` pointing at `https://github.com/<OWNER>/<REPO>/releases/latest/download/latest.json`).
- Repository secrets for updater signing and (if used) code signing—see [UPDATER_FEED.md](./UPDATER_FEED.md).
- Frontend capabilities include `updater:default` (and process relaunch permissions for `@tauri-apps/plugin-process`).

## Steps

1. **Bump versions** (root `package.json` + `src-tauri/tauri.conf.json` stay in sync):

   ```bash
   ./scripts/sync-version.sh 1.2.3
   ```

2. **Update the changelog**  
   Edit `CHANGELOG.md` (or your release notes source) for `1.2.3`.

3. **Commit the version bump + changelog**

   ```bash
   git add -A
   git commit -m "Release v1.2.3"
   ```

4. **Tag and push** (annotated tag recommended so the release workflow can use the tag body as notes):

   ```bash
   git tag -a v1.2.3 -m "v1.2.3"
   git push origin main   # or your release branch
   git push origin v1.2.3
   ```

5. **Watch CI**  
   Open **Actions → Release** for the `v1.2.3` run. Confirm all matrix jobs (Linux, macOS x86_64, macOS aarch64, Windows) succeed.

6. **Verify the GitHub Release**  
   - **Stable** tags (`v1.2.3` with no prerelease segment): the workflow publishes a **non-draft** release. Check **Releases** for assets: per-platform installers/archives, each `.sig` sibling if you browse the same tag’s artifacts, and `latest.json`.  
   - **Prerelease** tags (e.g. `v1.2.3-beta.1`): the workflow creates a **draft** (and marks prerelease). Open the draft, confirm assets and `latest.json`, then **Publish release** when ready.

7. **Smoke-test the updater**  
   Install the previous public build, launch with network available, confirm the prompt appears and that “Install now” or “Install on next launch” works against the new `latest.json`.

## Version / tag rules

- Tags must match `v*.*.*` to trigger `.github/workflows/release.yml`.
- `./scripts/sync-version.sh --check` in CI requires the tag (without `v`) to match both `package.json` and `src-tauri/tauri.conf.json` `version` fields.
