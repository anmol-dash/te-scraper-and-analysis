# Adding a command (Python → Tauri → UI)

End-to-end recipe to expose a new **`yourtool`** subcommand through the bundled IPC worker and desktop shell.

Assume paths:

- `backend/yourtool/cli.py` — argparse handlers invoked by IPC `run`.
- `backend/main.py` — NDJSON worker loop (**usually unchanged** if behaviour fits CLI argv).
- `src-tauri/src/commands.rs` — Tauri command wrappers.
- `src/lib/ipc.ts` — frontend bridge + types.
- UI components — buttons/forms wired to ipc helpers.

---

## 1. CLI handler (`backend/yourtool/cli.py`)

```diff
 def main(argv: Sequence[str] | None = None, cancel_event: Event | None = None) -> int:
@@
     p_sleep.add_argument("--seconds", type=float, default=1.0, help="Total time to sleep (default: 1).")
     p_sleep.set_defaults(handler=_cmd_sleep)
+
+    p_echo = sub.add_parser("echo", help="Print a message (demo IPC plumbing).")
+    p_echo.add_argument("text", help="Message to print.")
+    p_echo.set_defaults(handler=_cmd_echo)
@@
 def _cmd_sleep(args: argparse.Namespace, cancel_event: Event | None) -> int:
@@
     return 0
+
+
+def _cmd_echo(args: argparse.Namespace, cancel_event: Event | None) -> int:
+    print(args.text, flush=True)
+    return 0
```

Extend **`_cmd_*`** to poll `cancel_event` on long loops so **Cancel** from the UI maps to exit code **130** (pattern already used in `_cmd_sleep`).

---

## 2. `backend/main.py`

No edit required when the command is reachable via **`{'command':'run','args':{'argv':['echo','hello']}}`**. Only touch **`ipc_main`** if introducing a **new top-level IPC verb** (rare).

---

## 3. Rust invoke (`src-tauri/src/commands.rs`)

Sketch:

```diff
+#[tauri::command]
+pub fn run_echo(app: tauri::AppHandle, text: String) -> Result<String, String> {
+    let argv = vec!["echo".into(), text];
+    ipc::spawn_run(&app, argv).map_err(|e| e.to_string())
+}
```

Register in **`invoke_handler`**:

```diff
      .invoke_handler(tauri::generate_handler![
          ping_sidecar,
+         run_echo,
      ])
```

Reuse shared **`ipc`** helpers that:

1. Allocate request **`id`** (UUID).
2. Write NDJSON line to worker stdin containing **`run`** + **`args.argv`** (prepend binary name if argv convention requires `yourtool`).
3. Attach listeners streaming **`log`** / **`progress`** events to the webview.

---

## 4. TypeScript IPC (`src/lib/ipc.ts`)

```diff
+export async function echoViaSidecar(text: string): Promise<void> {
+  const id = crypto.randomUUID();
+  await invoke('run_echo', { text });
+  // Alternatively call a generic invoke('ipc_run', { argv: ['echo', text], id })
+}
+
+listen('ipc:log', (event) => {
+  // append event.payload.line to terminal component
+});
```

Prefer **one generic `ipc_run`** wrapper plus typed helpers so argv stays canonical between Rust and Python.

---

## 5. UI wiring

```diff
 function EchoPanel() {
+  const [out, setOut] = useState('');
   async function onSubmit(msg: string) {
-    // placeholder
+    setOut('');
+    await echoViaSidecar(msg);
   }
   return (
     <>
       <input />
       <button onClick={() => onSubmit(/* … */)}>Echo</button>
+      <pre>{out}</pre>
     </>
   );
 }
```

Subscribe to **`ipc:log`** (or equivalent channel your Rust emits) to populate **`out`** incrementally.

---

## Verification

```bash
cd backend && PYTHONPATH=. python -c "from yourtool.cli import main; raise SystemExit(main(['echo','hi']))"
pytest tests/test_ipc.py          # thin bridge regressions
pnpm tauri dev                    # manual UI click-test
```

If logs never arrive, confirm Rust parses **`type":"log"`** lines from **`main.py`** (worker protocol), not **`method`** responses from **`ipc_sidecar`** — mismatch here is the most common wiring bug after argv drift.
