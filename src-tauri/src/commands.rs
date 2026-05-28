use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value};
use tauri::{AppHandle, State};

use crate::github_report::{submit_fatal_report, FatalReport};
use crate::python_bridge::{BridgeError, PythonBridge};

fn map_err(e: BridgeError) -> String {
    e.to_string()
}

const MAX_PREVIEW_HTML_BYTES: u64 = 25 * 1024 * 1024;

#[derive(Serialize)]
pub struct PreviewHtml {
    ok: bool,
    html: String,
    size: u64,
}

fn preview_root() -> PathBuf {
    std::env::temp_dir().join("gameca_preview")
}

fn validate_preview_html_path(local_path: &Path) -> Result<PathBuf, String> {
    let ext = local_path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase());
    if !matches!(ext.as_deref(), Some("html" | "htm")) {
        return Err("preview file must be .html or .htm".to_string());
    }

    let root = preview_root()
        .canonicalize()
        .map_err(|_| "preview directory does not exist".to_string())?;
    let file = local_path
        .canonicalize()
        .map_err(|_| "preview file does not exist".to_string())?;

    if !file.starts_with(&root) {
        return Err("preview file is outside the GAMECA preview directory".to_string());
    }

    Ok(file)
}

#[tauri::command]
pub async fn py_run(state: State<'_, PythonBridge>, argv: Vec<String>) -> Result<Value, String> {
    let (rid, payload) = state
        .request_raw("run", json!({ "argv": argv }), None)
        .await
        .map_err(map_err)?;
    Ok(json!({ "id": rid.to_string(), "result": payload }))
}

#[tauri::command]
pub async fn py_cancel(state: State<'_, PythonBridge>, id: String) -> Result<(), String> {
    state
        .request_with_timeout(
            "cancel",
            json!({ "target_id": id }),
            Some(Duration::from_secs(30)),
        )
        .await
        .map(|_| ())
        .map_err(map_err)
}

#[tauri::command]
pub async fn py_ping(state: State<'_, PythonBridge>) -> Result<Value, String> {
    state.request("ping", json!({})).await.map_err(map_err)
}

#[tauri::command]
pub async fn py_version(state: State<'_, PythonBridge>) -> Result<String, String> {
    let v = state.request("version", json!({})).await.map_err(map_err)?;
    match v {
        Value::String(s) => Ok(s),
        other => serde_json::to_string(&other).map_err(|e| e.to_string()),
    }
}

#[tauri::command]
pub async fn py_setup(state: State<'_, PythonBridge>) -> Result<Value, String> {
    state
        .request_with_timeout("setup", json!({}), Some(Duration::from_secs(10)))
        .await
        .map_err(map_err)
}

#[tauri::command]
pub fn read_preview_html(local_path: String) -> Result<PreviewHtml, String> {
    let path = validate_preview_html_path(Path::new(&local_path))?;
    let metadata = fs::metadata(&path).map_err(|e| format!("failed to stat preview file: {e}"))?;
    let size = metadata.len();

    if size > MAX_PREVIEW_HTML_BYTES {
        return Err(format!(
            "preview HTML is too large ({:.1} MB, limit is {:.1} MB)",
            size as f64 / (1024.0 * 1024.0),
            MAX_PREVIEW_HTML_BYTES as f64 / (1024.0 * 1024.0)
        ));
    }

    let html = fs::read_to_string(&path)
        .map_err(|e| format!("failed to read preview HTML as UTF-8: {e}"))?;
    Ok(PreviewHtml {
        ok: true,
        html,
        size,
    })
}

#[tauri::command]
pub fn app_exit(app: AppHandle) {
    app.exit(0);
}

#[tauri::command]
pub fn app_relaunch(app: AppHandle) {
    app.restart();
}

#[tauri::command]
pub async fn app_report_fatal(
    source: String,
    reason: String,
    detail: String,
) -> Result<Value, String> {
    let url = submit_fatal_report(FatalReport {
        source,
        reason,
        detail,
    })
    .await?;

    Ok(json!({ "ok": true, "url": url }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_preview_file(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before epoch")
            .as_nanos();
        let dir = preview_root().join(format!("test_{nonce}"));
        fs::create_dir_all(&dir).expect("create preview test dir");
        dir.join(name)
    }

    #[test]
    fn validate_preview_html_path_allows_html_under_preview_root() {
        let file = unique_preview_file("report.html");
        fs::write(&file, "<html></html>").expect("write preview html");

        let validated = validate_preview_html_path(&file).expect("valid preview path");

        assert_eq!(
            validated,
            file.canonicalize().expect("canonical preview file")
        );
    }

    #[test]
    fn validate_preview_html_path_rejects_non_html_extension() {
        let file = unique_preview_file("report.txt");
        fs::write(&file, "plain text").expect("write preview text");

        let err = validate_preview_html_path(&file).expect_err("non-html should fail");

        assert!(err.contains(".html"));
    }

    #[test]
    fn validate_preview_html_path_rejects_file_outside_preview_root() {
        let file = std::env::temp_dir().join("gameca_preview_outside_report.html");
        fs::write(&file, "<html></html>").expect("write outside html");

        let err = validate_preview_html_path(&file).expect_err("outside path should fail");

        assert!(err.contains("outside"));
        let _ = fs::remove_file(file);
    }
}
