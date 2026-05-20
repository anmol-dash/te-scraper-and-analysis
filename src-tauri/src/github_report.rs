use std::{env, fs, path::PathBuf};

use serde::Deserialize;
use serde_json::{json, Value};

const DEFAULT_REPO: &str = "anmol-dash/te-scraper-and-analysis";

#[derive(Debug, Deserialize)]
struct ReportConfig {
    github_report_repo: Option<String>,
    github_report_token: Option<String>,
    update_repo: Option<String>,
}

#[derive(Clone, Debug)]
pub struct FatalReport {
    pub source: String,
    pub reason: String,
    pub detail: String,
}

fn gameca_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        env::var_os("APPDATA").map(|p| PathBuf::from(p).join("GAMECA"))
    } else {
        env::var_os("HOME").map(|p| PathBuf::from(p).join(".gameca"))
    }
}

fn read_config() -> Option<ReportConfig> {
    let path = gameca_dir()?.join("config.json");
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

fn report_target() -> Option<(String, String)> {
    let cfg = read_config();
    let repo = env::var("GAMECA_GITHUB_REPORT_REPO")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .or_else(|| {
            cfg.as_ref()
                .and_then(|c| c.github_report_repo.clone())
                .filter(|v| !v.trim().is_empty())
        })
        .or_else(|| {
            cfg.as_ref()
                .and_then(|c| c.update_repo.clone())
                .filter(|v| !v.trim().is_empty())
        })
        .unwrap_or_else(|| DEFAULT_REPO.to_string());

    let token = env::var("GAMECA_GITHUB_REPORT_TOKEN")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .or_else(|| env::var("GITHUB_TOKEN").ok().filter(|v| !v.trim().is_empty()))
        .or_else(|| {
            cfg.and_then(|c| c.github_report_token)
                .filter(|v| !v.trim().is_empty())
        })?;

    Some((repo, token))
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut out = s.chars().take(max.saturating_sub(3)).collect::<String>();
    out.push_str("...");
    out
}

fn issue_body(report: &FatalReport) -> String {
    format!(
        "A GAMECA fatal error was reported automatically before shutdown.\n\n\
         ## Summary\n\
         - Source: `{}`\n\
         - Reason: `{}`\n\
         - App version: `{}`\n\
         - OS: `{}`\n\
         - Arch: `{}`\n\n\
         ## Detail\n\
         ```text\n{}\n```\n",
        truncate(&report.source, 120),
        truncate(&report.reason, 240),
        env!("CARGO_PKG_VERSION"),
        env::consts::OS,
        env::consts::ARCH,
        truncate(&report.detail, 8_000),
    )
}

pub async fn submit_fatal_report(report: FatalReport) -> Result<Option<String>, String> {
    let Some((repo, token)) = report_target() else {
        return Ok(None);
    };

    let title = format!("GAMECA fatal error: {}", truncate(&report.reason, 120));
    let body = issue_body(&report);
    let url = format!("https://api.github.com/repos/{repo}/issues");
    let client = reqwest::Client::new();

    let res = client
        .post(url)
        .bearer_auth(token)
        .header("User-Agent", "GAMECA-crash-reporter/1.0")
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .json(&json!({ "title": title, "body": body }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let status = res.status();
    let payload: Value = res.json().await.unwrap_or_else(|_| json!({}));
    if !status.is_success() {
        return Err(format!(
            "GitHub issue creation failed with HTTP {status}: {}",
            payload
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
        ));
    }

    Ok(payload
        .get("html_url")
        .and_then(Value::as_str)
        .map(ToString::to_string))
}

pub fn spawn_fatal_report(report: FatalReport) {
    tauri::async_runtime::spawn(async move {
        match submit_fatal_report(report).await {
            Ok(Some(url)) => eprintln!("[gameca] fatal report created: {url}"),
            Ok(None) => eprintln!(
                "[gameca] fatal report skipped: GAMECA_GITHUB_REPORT_TOKEN is not configured"
            ),
            Err(e) => eprintln!("[gameca] fatal report failed: {e}"),
        }
    });
}
