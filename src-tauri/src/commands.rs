use std::time::Duration;

use serde_json::{json, Value};
use tauri::{AppHandle, State};

use crate::github_report::{submit_fatal_report, FatalReport};
use crate::python_bridge::{BridgeError, PythonBridge};

fn map_err(e: BridgeError) -> String {
    e.to_string()
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
        .request_with_timeout("cancel", json!({ "target_id": id }), Some(Duration::from_secs(30)))
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
