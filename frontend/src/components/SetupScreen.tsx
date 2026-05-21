import { useEffect, useRef, useState } from "react";
import { getVersion } from "@tauri-apps/api/app";
import { appExit, pySetup } from "@/lib/ipc";
import { useAppStore } from "@/store/appStore";

const ART = `  ██████╗  █████╗ ███╗   ███╗███████╗ ██████╗  █████╗
 ██╔════╝ ██╔══██╗████╗ ████║██╔════╝██╔════╝ ██╔══██╗
 ██║  ███╗███████║██╔████╔██║█████╗  ██║      ███████║
 ██║   ██║██╔══██║██║╚██╔╝██║██╔══╝  ██║      ██╔══██║
 ╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗╚██████╗ ██║  ██║
  ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝`;

const PHASE_LABEL: Record<string, string> = {
  checking:   "Checking environment…",
  creating:   "Creating virtual environment…",
  installing: "Installing packages…",
  done:       "Ready",
  error:      "Setup failed",
};

export default function SetupScreen() {
  const phase    = useAppStore((s) => s.setupPhase);
  const fraction = useAppStore((s) => s.setupFraction);
  const message  = useAppStore((s) => s.setupMessage);
  const error    = useAppStore((s) => s.setupError);
  const logs     = useAppStore((s) => s.setupLogs);

  const [appVersion, setAppVersion] = useState("");
  const [logsOpen, setLogsOpen] = useState(true);

  useEffect(() => {
    void getVersion().then(setAppVersion);
  }, []);
  const logEndRef = useRef<HTMLDivElement>(null);

  // Instantly scroll to the latest log line whenever new lines arrive
  // or the panel is opened.
  useEffect(() => {
    if (logsOpen) {
      logEndRef.current?.scrollIntoView({ behavior: "instant" });
    }
  }, [logs, logsOpen]);

  // Open logs automatically if setup fails.
  useEffect(() => {
    if (phase === "error") setLogsOpen(true);
  }, [phase]);

  const pct          = Math.round(fraction * 100);
  const label        = phase && phase !== "log" ? (PHASE_LABEL[phase] ?? phase) : "Starting…";
  const isError      = phase === "error";
  const indeterminate = !phase || phase === "checking" || phase === "creating" || phase === "log";

  return (
    <div className="fixed inset-0 z-40 flex flex-col items-center justify-center gap-6 bg-[var(--app-bg)] px-6">
      {/* ASCII art */}
      <div className="flex flex-col items-center gap-1">
        <pre className="select-none font-mono text-[5.5px] leading-[1.2] text-[var(--app-accent)]">
          {ART}
        </pre>
        {appVersion && (
          <span className="font-mono text-[9px] tracking-widest text-[var(--app-muted)] select-none">
            v{appVersion}
          </span>
        )}
      </div>

      <div className="flex w-full max-w-lg flex-col gap-4">
        {/* Phase label + current message */}
        <div className="text-center">
          <p className="text-sm font-semibold text-[var(--app-text)]">{label}</p>
          <p className="mt-1 min-h-[1.25rem] truncate text-xs text-[var(--app-muted)]">
            {message}
          </p>
        </div>

        {/* Progress bar */}
        <div
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={indeterminate ? undefined : pct}
          className="h-2 w-full overflow-hidden rounded-full bg-[var(--app-border)]"
        >
          {indeterminate ? (
            <div className="h-full w-1/3 animate-[slide_1.4s_ease-in-out_infinite] rounded-full bg-[var(--app-accent)]" />
          ) : (
            <div
              className="h-full rounded-full bg-[var(--app-accent)] transition-all duration-300"
              style={{ width: `${pct}%` }}
            />
          )}
        </div>

        {/* Percentage */}
        {!indeterminate && !isError && (
          <p className="text-center text-xs tabular-nums text-[var(--app-muted)]">
            {pct}%
          </p>
        )}

        {/* Error box */}
        {isError && (
          <div className="rounded-md border border-red-400/40 bg-red-500/10 p-3 text-xs text-red-400">
            <p className="font-semibold">Setup failed</p>
            <p className="mt-1 whitespace-pre-wrap break-words">{error}</p>
          </div>
        )}

        {/* ── Log panel ───────────────────────────────────────────────── */}
        <div className="rounded-md border border-[var(--app-border)] bg-[var(--app-panel)]">
          {/* Toggle header */}
          <button
            type="button"
            onClick={() => setLogsOpen((o) => !o)}
            className="flex w-full items-center justify-between px-3 py-2 text-xs font-medium text-[var(--app-muted)] transition hover:text-[var(--app-text)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)] focus-visible:ring-inset"
          >
            <span>
              {logsOpen ? "▾" : "▸"}&nbsp; Installation logs
              {logs.length > 0 && (
                <span className="ml-2 tabular-nums opacity-60">
                  ({logs.length} lines)
                </span>
              )}
            </span>
            <span className="opacity-50">{logsOpen ? "hide" : "show"}</span>
          </button>

          {/* Scrollable log body */}
          {logsOpen && (
            <div className="max-h-52 overflow-y-auto border-t border-[var(--app-border)] px-3 py-2">
              {logs.length === 0 ? (
                <p className="text-xs italic text-[var(--app-muted)]">
                  No log output yet…
                </p>
              ) : (
                <pre className="whitespace-pre-wrap break-words font-mono text-[10px] leading-relaxed text-[var(--app-text)]">
                  {logs.join("\n")}
                </pre>
              )}
              <div ref={logEndRef} />
            </div>
          )}
        </div>

        {/* Action buttons */}
        <div className="flex justify-center gap-3">
          {isError && (
            <button
              type="button"
              onClick={() => void pySetup()}
              className="rounded-md bg-[var(--app-accent)] px-4 py-2 text-sm font-semibold text-white transition hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)]"
            >
              Retry
            </button>
          )}
          <button
            type="button"
            onClick={() => void appExit()}
            className="rounded-md border border-[var(--app-border)] px-4 py-2 text-sm font-medium text-[var(--app-muted)] transition hover:bg-[var(--app-border)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)]"
          >
            Quit
          </button>
        </div>
      </div>
    </div>
  );
}
