/**
 * Opt-in desktop smoke test hook.
 *
 * Official Tauri 2 CLI does not ship `tauri test`; desktop WebDriver goes through
 * `tauri-driver` (Linux WebKitWebDriver / Windows EdgeDriver). macOS desktop
 * WebDriver is not supported in upstream docs.
 *
 * Usage:
 *   RUN_E2E=1 node smoke.mjs
 *
 * To run a real check, set WEBDRIVER_URL (e.g. http://127.0.0.1:4444) and
 * TAURI_WEBVIEW_URI (printed by tauri-driver) and extend this script with WDIO.
 */

const run = () => {
  if (process.env.RUN_E2E !== "1") {
    console.log("E2E skipped (set RUN_E2E=1 to opt in).");
    return;
  }

  if (!process.env.WEBDRIVER_URL) {
    console.warn(
      "[e2e] RUN_E2E=1 but WEBDRIVER_URL is unset — no WebDriver session started.",
    );
    console.warn(
      "[e2e] Follow https://v2.tauri.app/develop/tests/webdriver/ — install tauri-driver and a native driver.",
    );
    return;
  }

  console.warn(
    "[e2e] WEBDRIVER_URL is set; extend smoke.mjs with your WebDriver client to:",
  );
  console.warn("      1) wait for [data-testid=ping-status] to contain ping:ok");
  console.warn("      2) click [data-testid=run-example]");
  console.warn(
    "      3) assert [data-testid=log-region] eventually shows run argv=",
  );
};

run();
