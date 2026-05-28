mod commands;
mod github_report;
mod python_bridge;

use tauri::{Manager, RunEvent};

use crate::python_bridge::PythonBridge;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("[gameca PANIC] {info}");
    }));

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .register_uri_scheme_protocol("gamecapreview", |_app, request| {
            // Serve HTML preview files from $TMPDIR/gameca_preview/ via a custom scheme
            // so WKWebView executes the inline JavaScript (blob: / srcdoc iframes don't).
            fn url_decode(s: &str) -> String {
                let mut out = String::with_capacity(s.len());
                let bytes = s.as_bytes();
                let mut i = 0;
                while i < bytes.len() {
                    if bytes[i] == b'%' && i + 2 < bytes.len() {
                        if let (Some(h), Some(l)) = (
                            (bytes[i + 1] as char).to_digit(16),
                            (bytes[i + 2] as char).to_digit(16),
                        ) {
                            out.push(char::from((h * 16 + l) as u8));
                            i += 3;
                            continue;
                        }
                    }
                    out.push(char::from(bytes[i]));
                    i += 1;
                }
                out
            }

            let raw = request.uri().path().trim_start_matches('/');
            let filename = url_decode(raw);

            // Reject any path traversal attempts.
            if filename.is_empty()
                || filename.contains('/')
                || filename.contains('\\')
                || filename.starts_with('.')
            {
                return tauri::http::Response::builder()
                    .status(403)
                    .header("Content-Type", "text/plain")
                    .body(b"Forbidden".to_vec())
                    .unwrap();
            }

            let file_path = std::env::temp_dir().join("gameca_preview").join(&filename);
            match std::fs::read(&file_path) {
                Ok(data) => tauri::http::Response::builder()
                    .status(200)
                    .header("Content-Type", "text/html; charset=utf-8")
                    .body(data)
                    .unwrap(),
                Err(e) => tauri::http::Response::builder()
                    .status(404)
                    .header("Content-Type", "text/plain")
                    .body(format!("Not found: {e}").into_bytes())
                    .unwrap(),
            }
        })
        .setup(|app| {
            let bridge = PythonBridge::spawn(app.handle().clone())?;
            app.manage(bridge);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::py_run,
            commands::py_cancel,
            commands::py_ping,
            commands::py_version,
            commands::py_setup,
            commands::read_preview_html,
            commands::app_exit,
            commands::app_relaunch,
            commands::app_report_fatal,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let RunEvent::ExitRequested { .. } = event {
                let bridge = app_handle.state::<PythonBridge>();
                tauri::async_runtime::block_on(bridge.shutdown());
            }
        });
}
