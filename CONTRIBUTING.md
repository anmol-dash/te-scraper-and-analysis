# Contributing

## Prerequisites

Install the following before building or testing locally:

| Tool | Version / notes |
|------|------------------|
| **Node.js** | 20+ |
| **pnpm** | Latest stable (`corepack enable` or npm global install) |
| **Rust** | Stable toolchain (`rustup default stable`) |
| **Python** | 3.11+ |

### Tauri platform dependencies

- **Linux:** WebKitGTK **4.1** (`webkit2gtk-4.1`, plus GTK/Cairo deps per [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/)).
- **macOS:** Xcode Command Line Tools (`xcode-select --install`).
- **Windows:** Visual Studio Build Tools with **Desktop development with C++** workload (MSVC, Windows SDK).

## Clone and install (frontend)

```bash
pnpm install
```

Python backend deps (from repo root or `backend/` as applicable):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Optional: pip install pytest for backend IPC tests
```

## Run development UI only

From the app package root (once `src-tauri/` and frontend exist):

```bash
pnpm tauri dev
```

This starts the Vite/dev server and launches the desktop shell with hot reload on the webview assets.

To exercise Python IPC without the full UI:

```bash
cd backend
PYTHONPATH=. python -m python_bridge.ipc_sidecar   # thin NDJSON bridge (tests)
# or full worker loop:
PYTHONPATH=. python main.py   # when wired as __main__ / packaged entry
```

## Tests

Run each layer where it applies:

```bash
# Frontend / TS unit tests (if configured)
pnpm test

# Rust (from src-tauri/)
cd src-tauri && cargo test

# Python IPC bridge (backend/)
cd backend && pytest tests/
```

Fix failures before opening a PR; add tests when changing IPC boundaries or cancellation behaviour.

## Branch naming

Use short, scoped prefixes:

- `feat/` — new user-visible capability  
- `fix/` — bug fixes  
- `docs/` — documentation only  
- `chore/` — tooling, deps, CI  
- `refactor/` — behaviour-preserving code moves  

Example: `feat/ipc-progress-events`, `fix/sidecar-shutdown-race`.

## Commit messages

Follow **[Conventional Commits](https://www.conventionalcommits.org/)**:

```
<type>(<optional scope>): <imperative description>

[optional body]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.

Examples:

- `feat(ipc): stream tqdm progress as NDJSON events`
- `fix(tauri): resolve sidecar path on aarch64 macOS`

## Pull request checklist

- [ ] **Purpose** — Linked issue or short rationale in the PR description  
- [ ] **Scope** — Diff stays focused; no unrelated formatting churn  
- [ ] **Tests** — Added/updated tests for IPC, Rust commands, or UI logic as appropriate  
- [ ] **Docs** — Updated `docs/` or README if behaviour or setup changed  
- [ ] **Manual smoke** — `pnpm tauri dev` or documented equivalent still starts  
- [ ] **Breaking changes** — Called out explicitly with migration notes  

Maintainers may request signed commits or CI green checks before merge.
