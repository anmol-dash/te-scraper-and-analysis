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
