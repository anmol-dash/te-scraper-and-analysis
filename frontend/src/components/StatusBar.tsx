import { useAppStore } from "@/store/appStore";
import { pyPing } from "@/lib/ipc";
import { useEffect, useMemo, useRef, useState } from "react";

export default function StatusBar() {
  const runPhase = useAppStore((s) => s.runPhase);
  const runElapsedLabel = useAppStore((s) => s.runElapsedLabel);
  const progress = useAppStore((s) => s.progress);
  const setFatalMessage = useAppStore((s) => s.setFatalMessage);
  const hpcConnected = useAppStore((s) => s.hpcConnected);
  const hpcHasInternet = useAppStore((s) => s.hpcHasInternet);
  const fileViewerOpen = useAppStore((s) => s.fileViewerOpen);
  const toggleFileViewer = useAppStore((s) => s.toggleFileViewer);
  const toggleNotifications = useAppStore((s) => s.toggleNotifications);
  const [pingOk, setPingOk] = useState<boolean | null>(null);
  const [version, setVersion] = useState<string>("");
  const failStreakRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const res = await pyPing();
        failStreakRef.current = 0;
        if (!cancelled) {
          setPingOk(Boolean(res.ok));
          setVersion(res.version ?? "");
        }
      } catch {
        failStreakRef.current += 1;
        if (!cancelled) {
          setPingOk(false);
        }
        if (failStreakRef.current >= 5) {
          setFatalMessage(
            "The desktop shell is not responding to internal health checks. You can quit or try restarting the application.",
          );
        }
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), 2_500);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [setFatalMessage]);

  const runLabel = useMemo(() => {
    if (runPhase === "idle") {
      return "Idle";
    }
    if (runPhase === "running") {
      return `Running (${runElapsedLabel})`;
    }
    if (runPhase === "done") {
      return `Done (${runElapsedLabel})`;
    }
    return `Errored (${runElapsedLabel})`;
  }, [runPhase, runElapsedLabel]);

  const fraction = useMemo(() => {
    if (!progress?.fraction && progress?.current != null && progress.total) {
      return progress.current / progress.total;
    }
    if (typeof progress?.fraction === "number") {
      return Math.min(1, Math.max(0, progress.fraction));
    }
    return runPhase === "running" ? null : 0;
  }, [progress, runPhase]);

  return (
    <header className="flex shrink-0 items-center gap-4 border-b border-[var(--app-border)] bg-[var(--app-panel)] px-4 py-2">
      <div className="flex shrink-0 flex-col items-start">
        <pre className="font-mono text-[4.5px] leading-[1.15] text-[var(--app-accent)] select-none">{
`  ██████╗  █████╗ ███╗   ███╗███████╗ ██████╗  █████╗
 ██╔════╝ ██╔══██╗████╗ ████║██╔════╝██╔════╝ ██╔══██╗
 ██║  ███╗███████║██╔████╔██║█████╗  ██║      ███████║
 ██║   ██║██╔══██║██║╚██╔╝██║██╔══╝  ██║      ██╔══██║
 ╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗╚██████╗ ██║  ██║
  ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝`
        }</pre>
        {version ? (
          <span className="pl-px font-mono text-[8px] tracking-widest text-[var(--app-muted)] select-none">
            v{version}
          </span>
        ) : null}
      </div>
      <div className="flex items-center gap-2 text-sm">
        <span
          className="inline-flex h-2.5 w-2.5 rounded-full"
          style={{
            backgroundColor:
              pingOk === null ? "#94a3b8" : pingOk ? "#16a34a" : "#dc2626",
          }}
          aria-hidden
        />
        <span className="sr-only">
          {pingOk === null
            ? "Ping status loading"
            : pingOk
              ? "Ping connected"
              : "Ping failed"}
        </span>
        <span className="text-[var(--app-muted)]">Ping</span>
        <span
          data-testid="ping-status"
          className="text-xs font-medium tabular-nums"
        >
          {pingOk === null ? "ping:pending" : pingOk ? "ping:ok" : "ping:fail"}
        </span>
      </div>

      <div className="text-sm font-medium">{runLabel}</div>

      {hpcConnected && (
        <>
          <span
            className="inline-flex items-center gap-1 text-xs"
            title={hpcHasInternet ? "Cluster has internet access" : "Cluster node has no internet — UCSC/Dfam fetch unavailable"}
          >
            <span
              className="inline-flex h-2 w-2 rounded-full"
              style={{ backgroundColor: hpcHasInternet ? "#16a34a" : "#d97706" }}
            />
            <span className={hpcHasInternet ? "text-green-600 dark:text-green-400" : "text-amber-600 dark:text-amber-400"}>
              {hpcHasInternet ? "internet ✓" : "no internet"}
            </span>
          </span>
          <button
            type="button"
            onClick={toggleFileViewer}
            className={`shrink-0 rounded px-2.5 py-1 text-xs font-medium transition ${
              fileViewerOpen
                ? "bg-[var(--app-accent)] text-white"
                : "border border-[var(--app-border)] text-[var(--app-muted)] hover:text-[var(--app-text)]"
            }`}
          >
            {fileViewerOpen ? "✕ Files" : "📁 Files"}
          </button>
        </>
      )}

      <button
        type="button"
        onClick={toggleNotifications}
        title="Email notification settings"
        className="shrink-0 rounded px-2.5 py-1 text-xs font-medium border border-[var(--app-border)] text-[var(--app-muted)] hover:text-[var(--app-text)] transition"
      >
        Notifications
      </button>

      <div className="min-w-0 flex-1">
        {progress?.message ? (
          <p className="mb-1 truncate text-xs text-[var(--app-muted)]">
            {progress.message}
          </p>
        ) : null}
        <div
          className="h-2 w-full overflow-hidden rounded bg-[var(--app-bg)]"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={
            fraction == null ? undefined : Math.round(fraction * 100)
          }
        >
          <div
            className="h-full bg-[var(--app-accent)] transition-[width] duration-150"
            style={{
              width: `${fraction == null ? 0 : Math.min(100, Math.max(0, fraction * 100))}%`,
            }}
          />
        </div>
      </div>
    </header>
  );
}
