use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dashmap::DashMap;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};
use tauri_plugin_shell::process::{Command, CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;
use tokio::sync::{mpsc, oneshot, Mutex as TokioMutex};
use uuid::Uuid;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(3600 * 8);
const PING_TIMEOUT: Duration = Duration::from_secs(5);
const SHUTDOWN_WAIT: Duration = Duration::from_secs(3);
const MAX_RESTARTS: u32 = 3;
const INITIAL_BACKOFF_MS: u64 = 250;

/// Inbound messages sent to the bridge writer task (sidecar stdin).
#[derive(Debug)]
pub enum Request {
    WriteStdin(Vec<u8>),
}

/// Response delivered to a waiting `request()` caller once a matching JSON line arrives.
#[derive(Debug)]
pub enum Response {
    Ok(Value),
    Err(String),
}

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("bridge channel closed")]
    ChannelClosed,
    #[error("request timed out")]
    Timeout,
    #[error("sidecar is unavailable after repeated crashes: {0}")]
    Fatal(String),
    #[error("sidecar reported error: {0}")]
    PythonError(String),
    #[error("serialization error: {0}")]
    Serde(String),
}

/// Drives the bundled Python sidecar (`pytool`) over line-delimited JSON on stdin/stdout.
///
/// The live process handle is [`CommandChild`] from [`tauri_plugin_shell`] (`shell().sidecar("pytool")`).
/// Tauri resolves `bundle.externalBin` paths for dev and production; this is the supported API
/// instead of constructing a detached [`tokio::process::Child`] yourself.
pub struct PythonBridge {
    child: Arc<TokioMutex<Option<CommandChild>>>,
    request_tx: mpsc::Sender<Request>,
    pending: Arc<DashMap<Uuid, oneshot::Sender<Response>>>,
    app: AppHandle,
    shutdown_intent: Arc<AtomicBool>,
    fatal: Arc<AtomicBool>,
    fatal_reason: Arc<Mutex<Option<String>>>,
    generation: Arc<AtomicU32>,
}

impl PythonBridge {
    pub fn spawn(app: AppHandle) -> Result<Self, BridgeError> {
        let child = Arc::new(TokioMutex::new(None));
        let pending: Arc<DashMap<Uuid, oneshot::Sender<Response>>> = Arc::new(DashMap::new());
        let (request_tx, request_rx) = mpsc::channel::<Request>(256);
        let shutdown_intent = Arc::new(AtomicBool::new(false));
        let fatal = Arc::new(AtomicBool::new(false));
        let fatal_reason: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let generation = Arc::new(AtomicU32::new(0));

        let bridge = Self {
            child: child.clone(),
            request_tx: request_tx.clone(),
            pending: pending.clone(),
            app: app.clone(),
            shutdown_intent: shutdown_intent.clone(),
            fatal: fatal.clone(),
            fatal_reason: fatal_reason.clone(),
            generation: generation.clone(),
        };

        let child_w = child.clone();
        tauri::async_runtime::spawn(stdin_writer_task(request_rx, child_w));

        let app_sup = app.clone();
        tauri::async_runtime::spawn(supervisor_task(
            app_sup,
            child,
            pending,
            shutdown_intent,
            fatal,
            fatal_reason,
            generation,
        ));

        Ok(bridge)
    }

    pub async fn request(&self, command: &str, args: Value) -> Result<Value, BridgeError> {
        self.request_with_timeout(command, args, None).await
    }

    pub async fn request_with_timeout(
        &self,
        command: &str,
        args: Value,
        timeout: Option<Duration>,
    ) -> Result<Value, BridgeError> {
        self.request_raw(command, args, timeout)
            .await
            .map(|(_, v)| v)
    }

    /// Same as [`Self::request_with_timeout`], but returns the JSON-RPC request id for correlating `cancel` / UI.
    pub async fn request_raw(
        &self,
        command: &str,
        args: Value,
        timeout: Option<Duration>,
    ) -> Result<(Uuid, Value), BridgeError> {
        if self.fatal.load(Ordering::SeqCst) {
            let msg = self
                .fatal_reason
                .lock()
                .ok()
                .and_then(|g| g.clone())
                .unwrap_or_else(|| "sidecar permanently failed".into());
            return Err(BridgeError::Fatal(msg));
        }

        let timeout = timeout.unwrap_or_else(|| {
            if command == "ping" {
                PING_TIMEOUT
            } else {
                DEFAULT_TIMEOUT
            }
        });

        let id = Uuid::new_v4();
        let (tx, rx) = oneshot::channel::<Response>();
        self.pending.insert(id, tx);

        let line = serde_json::to_vec(&json!({
            "id": id.to_string(),
            "command": command,
            "args": args,
        }))
        .map_err(|e| BridgeError::Serde(e.to_string()))?;
        let mut frame = line;
        frame.push(b'\n');

        self.request_tx
            .send(Request::WriteStdin(frame))
            .await
            .map_err(|_| BridgeError::ChannelClosed)?;

        let gen_start = self.generation.load(Ordering::SeqCst);
        let outcome = tokio::time::timeout(timeout, rx).await;

        match outcome {
            Err(_) => {
                self.pending.remove(&id);
                Err(BridgeError::Timeout)
            }
            Ok(Err(_)) => {
                self.pending.remove(&id);
                if self.generation.load(Ordering::SeqCst) != gen_start {
                    Err(BridgeError::Fatal(
                        self.fatal_reason
                            .lock()
                            .ok()
                            .and_then(|g| g.clone())
                            .unwrap_or_else(|| "sidecar restarted".into()),
                    ))
                } else {
                    Err(BridgeError::ChannelClosed)
                }
            }
            Ok(Ok(Response::Ok(v))) => Ok((id, v)),
            Ok(Ok(Response::Err(s))) => Err(BridgeError::PythonError(s)),
        }
    }

    pub async fn shutdown(&self) {
        self.shutdown_intent.store(true, Ordering::SeqCst);
        fail_all_pending(&self.pending, "shutting down");
        let payload = serde_json::to_vec(&json!({
            "command": "shutdown",
            "args": serde_json::Map::new(),
        }))
        .unwrap_or_else(|_| br#"{"command":"shutdown","args":{}}\n"#.to_vec());
        let mut frame = payload;
        if !frame.ends_with(b"\n") {
            frame.push(b'\n');
        }
        let _ = self.request_tx.send(Request::WriteStdin(frame)).await;

        tokio::time::sleep(SHUTDOWN_WAIT).await;

        if let Some(ch) = self.child.lock().await.take() {
            let _ = ch.kill();
        }
    }
}

async fn stdin_writer_task(mut rx: mpsc::Receiver<Request>, child: Arc<TokioMutex<Option<CommandChild>>>) {
    while let Some(req) = rx.recv().await {
        match req {
            Request::WriteStdin(bytes) => {
                let mut guard = child.lock().await;
                if let Some(ch) = guard.as_mut() {
                    if let Err(e) = ch.write(&bytes) {
                        eprintln!("[gameca] stdin write error: {e}");
                    }
                }
            }
        }
    }
}

async fn supervisor_task(
    app: AppHandle,
    child: Arc<TokioMutex<Option<CommandChild>>>,
    pending: Arc<DashMap<Uuid, oneshot::Sender<Response>>>,
    shutdown_intent: Arc<AtomicBool>,
    fatal: Arc<AtomicBool>,
    fatal_reason: Arc<Mutex<Option<String>>>,
    generation: Arc<AtomicU32>,
) {
    let mut restarts = 0u32;
    let mut backoff_ms = INITIAL_BACKOFF_MS;
    let mut stdout_buf = String::new();

    loop {
        if fatal.load(Ordering::SeqCst) {
            return;
        }
        if shutdown_intent.load(Ordering::SeqCst) {
            return;
        }

        let sidecar_cmd: Command = match app.shell().sidecar("pytool") {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[gameca] sidecar command build failed: {e}");
                record_fatal(&fatal, &fatal_reason, format!("sidecar build: {e}"));
                let _ = app.emit(
                    "python://crashed",
                    json!({ "fatal": true, "reason": "sidecar build failed", "detail": e.to_string() }),
                );
                return;
            }
        };

        let (mut event_rx, spawned) = match sidecar_cmd.spawn() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[gameca] spawn failed: {e}");
                if restarts >= MAX_RESTARTS {
                    record_fatal(&fatal, &fatal_reason, format!("spawn failed: {e}"));
                    let _ = app.emit(
                        "python://crashed",
                        json!({ "fatal": true, "reason": "spawn failed after retries", "detail": e.to_string() }),
                    );
                    return;
                }
                let _ = app.emit(
                    "python://crashed",
                    json!({
                        "fatal": false,
                        "phase": "spawn",
                        "retry": restarts + 1,
                        "detail": e.to_string(),
                    }),
                );
                restarts += 1;
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(5_000);
                continue;
            }
        };

        restarts = 0;
        backoff_ms = INITIAL_BACKOFF_MS;
        *child.lock().await = Some(spawned);
        generation.fetch_add(1, Ordering::SeqCst);
        stdout_buf.clear();
        let mut saw_valid_line = false;
        let mut exit_detail = String::new();

        loop {
            let event = match event_rx.recv().await {
                Some(e) => e,
                None => {
                    exit_detail = "event channel closed".into();
                    break;
                }
            };

            match event {
                CommandEvent::Stdout(bytes) => {
                    let health = append_stdout_and_dispatch(&mut stdout_buf, &bytes, &pending, &app);
                    if health {
                        saw_valid_line = true;
                        restarts = 0;
                        backoff_ms = INITIAL_BACKOFF_MS;
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    let _ = app.emit("python://stderr", json!({ "line": text }));
                }
                CommandEvent::Error(err) => {
                    exit_detail = format!("command error: {err}");
                    eprintln!("[gameca] {exit_detail}");
                }
                CommandEvent::Terminated(payload) => {
                    exit_detail = format!("terminated: {payload:?}");
                    break;
                }
                _ => {}
            }
        }

        *child.lock().await = None;

        if shutdown_intent.load(Ordering::SeqCst) {
            return;
        }

        eprintln!("[gameca] sidecar exited unexpectedly: {exit_detail}");
        fail_all_pending(&pending, "sidecar process exited");

        restarts += 1;
        if restarts > MAX_RESTARTS {
            let msg = format!("sidecar exited: {exit_detail}");
            record_fatal(&fatal, &fatal_reason, msg);
            let _ = app.emit(
                "python://crashed",
                json!({ "fatal": true, "reason": "max restarts exceeded", "detail": exit_detail }),
            );
            return;
        }

        let _ = app.emit(
            "python://crashed",
            json!({
                "fatal": false,
                "retry": restarts,
                "detail": exit_detail,
                "had_valid_stdout": saw_valid_line,
            }),
        );
        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
        backoff_ms = (backoff_ms * 2).min(5_000);
    }
}

fn record_fatal(fatal: &AtomicBool, reason_store: &Mutex<Option<String>>, msg: String) {
    if let Ok(mut g) = reason_store.lock() {
        *g = Some(msg);
    }
    fatal.store(true, Ordering::SeqCst);
}

fn fail_all_pending(pending: &DashMap<Uuid, oneshot::Sender<Response>>, message: &str) {
    let ids: Vec<Uuid> = pending.iter().map(|e| *e.key()).collect();
    for id in ids {
        if let Some(tx) = pending.remove(&id) {
            let _ = tx.1.send(Response::Err(message.to_string()));
        }
    }
}

fn append_stdout_and_dispatch(
    buf: &mut String,
    bytes: &[u8],
    pending: &DashMap<Uuid, oneshot::Sender<Response>>,
    app: &AppHandle,
) -> bool {
    let chunk = String::from_utf8_lossy(bytes);
    buf.push_str(&chunk);
    let mut parsed_any = false;

    while let Some(pos) = buf.find('\n') {
        let line = buf[..pos].trim().to_string();
        buf.drain(..=pos);

        if line.is_empty() {
            continue;
        }

        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let _ = app.emit(
                    "python://log",
                    json!({ "type": "log", "message": "stdout json parse error", "detail": e.to_string(), "line": line }),
                );
                continue;
            }
        };

        parsed_any = true;
        dispatch_json_line(v, pending, app);
    }

    parsed_any
}

fn resolve_pending_response(
    v: &Value,
    pending: &DashMap<Uuid, oneshot::Sender<Response>>,
) -> bool {
    let Some(id_val) = v.get("id") else {
        return false;
    };
    let Some(id_str) = id_val.as_str() else {
        return false;
    };
    let Ok(id) = Uuid::parse_str(id_str) else {
        return false;
    };
    let Some((_, tx)) = pending.remove(&id) else {
        return false;
    };

    if let Some(err) = v.get("error") {
        let msg = match err {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let _ = tx.send(Response::Err(msg));
        return true;
    }

    if v.get("type").and_then(|x| x.as_str()) == Some("error") {
        let payload = v.get("payload").cloned().unwrap_or(Value::Null);
        let msg = payload
            .get("msg")
            .and_then(|x| x.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| payload.to_string());
        let _ = tx.send(Response::Err(msg));
        return true;
    }

    if let Some(result) = v.get("result") {
        let _ = tx.send(Response::Ok(result.clone()));
        return true;
    }

    if v.get("type").and_then(|x| x.as_str()) == Some("result") {
        let _ = tx.send(Response::Ok(v.get("payload").cloned().unwrap_or(Value::Null)));
        return true;
    }

    let _ = tx.send(Response::Err("response missing result/error".into()));
    true
}

fn dispatch_json_line(v: Value, pending: &DashMap<Uuid, oneshot::Sender<Response>>, app: &AppHandle) {
    if let Some(t) = v.get("type").and_then(|x| x.as_str()) {
        match t {
            "progress" => {
                let _ = app.emit("python://progress", &v);
                return;
            }
            "log" => {
                let _ = app.emit("python://log", &v);
                return;
            }
            "setup" => {
                let _ = app.emit("python://setup", &v);
                return;
            }
            "update" => {
                let _ = app.emit("python://update", &v);
                return;
            }
            _ => {}
        }
    }

    let _ = resolve_pending_response(&v, pending);
}

#[cfg(test)]
mod framing_tests {
    use super::{resolve_pending_response, Response};
    use dashmap::DashMap;
    use serde_json::json;
    use tokio::io::duplex;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::sync::oneshot;
    use uuid::Uuid;

    async fn read_line_utf8<R>(reader: &mut BufReader<R>) -> std::io::Result<Option<String>>
    where
        R: tokio::io::AsyncRead + Unpin,
    {
        let mut buf = String::new();
        let n = reader.read_line(&mut buf).await?;
        if n == 0 {
            return Ok(None);
        }
        Ok(Some(buf.trim_end_matches(['\r', '\n']).to_string()))
    }

    #[tokio::test]
    async fn line_splitter_duplex_frames_ndjson() {
        let (mut w, r) = duplex(4096);
        w.write_all(b"{\"x\":1}\n{\"y\":2}\n").await.unwrap();
        drop(w);

        let mut reader = BufReader::new(r);
        assert_eq!(
            read_line_utf8(&mut reader).await.unwrap().unwrap(),
            "{\"x\":1}"
        );
        assert_eq!(
            read_line_utf8(&mut reader).await.unwrap().unwrap(),
            "{\"y\":2}"
        );
        assert!(read_line_utf8(&mut reader).await.unwrap().is_none());
    }

    #[test]
    fn request_multiplexer_resolves_result_by_uuid() {
        let pending = DashMap::new();
        let id = Uuid::new_v4();
        let (tx, mut rx) = oneshot::channel();
        pending.insert(id, tx);

        assert!(resolve_pending_response(
            &json!({"id": id.to_string(), "result": {"phase": "ok"}}),
            &pending,
        ));
        assert!(pending.is_empty());

        match rx.blocking_recv().expect("reply") {
            Response::Ok(v) => assert_eq!(v["phase"], "ok"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn request_multiplexer_unknown_id_is_no_op() {
        let pending = DashMap::new();
        assert!(!resolve_pending_response(
            &json!({"id": Uuid::new_v4().to_string(), "result": true}),
            &pending,
        ));
    }
}
