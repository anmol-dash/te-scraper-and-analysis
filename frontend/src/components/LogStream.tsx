import { useVirtualizer } from "@tanstack/react-virtual";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { LogLevel } from "@/lib/logLevel";
import { useAppStore } from "@/store/appStore";

function levelPillClass(level: LogLevel): string {
  switch (level) {
    case "warn":
      return "bg-amber-100 text-amber-900 dark:bg-amber-950/50 dark:text-amber-200";
    case "error":
      return "bg-red-100 text-red-900 dark:bg-red-950/50 dark:text-red-200";
    default:
      return "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-100";
  }
}

const LEVELS: Array<"all" | LogLevel> = ["all", "info", "warn", "error"];

export default function LogStream() {
  const logs = useAppStore((s) => s.logs);
  const clearLogs = useAppStore((s) => s.clearLogs);
  const logLevelFilter = useAppStore((s) => s.logLevelFilter);
  const setLogLevelFilter = useAppStore((s) => s.setLogLevelFilter);
  const logSearch = useAppStore((s) => s.logSearch);
  const setLogSearch = useAppStore((s) => s.setLogSearch);

  const [autoScroll, setAutoScroll] = useState(true);
  const parentRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    const q = logSearch.trim().toLowerCase();
    return logs.filter((line) => {
      if (logLevelFilter !== "all" && line.level !== logLevelFilter) {
        return false;
      }
      if (q && !line.message.toLowerCase().includes(q)) {
        return false;
      }
      return true;
    });
  }, [logLevelFilter, logSearch, logs]);

  const rowVirtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 26,
    overscan: 24,
  });

  useEffect(() => {
    if (!autoScroll) {
      return;
    }
    const last = filtered.length - 1;
    if (last < 0) {
      return;
    }
    rowVirtualizer.scrollToIndex(last, { align: "end" });
  }, [autoScroll, filtered.length, rowVirtualizer]);

  const onScroll = useCallback(() => {
    const el = parentRef.current;
    if (!el) {
      return;
    }
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
    setAutoScroll(nearBottom);
  }, []);

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col bg-[var(--app-bg)]">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-[var(--app-border)] bg-[var(--app-panel)] px-3 py-2">
        <span className="text-xs font-semibold text-[var(--app-muted)]">
          Level
        </span>
        <div className="flex flex-wrap gap-1">
          {LEVELS.map((lvl) => (
            <button
              key={lvl}
              type="button"
              className={
                "rounded-full px-2 py-1 text-xs font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)] " +
                (logLevelFilter === lvl
                  ? "bg-[var(--app-accent)] text-white"
                  : "border border-[var(--app-border)] bg-[var(--app-bg)] text-[var(--app-text)] hover:bg-[var(--app-panel)]")
              }
              onClick={() => setLogLevelFilter(lvl)}
            >
              {lvl === "all" ? "All" : lvl}
            </button>
          ))}
        </div>

        <label className="ml-2 flex min-w-[10rem] flex-1 items-center gap-2 text-xs text-[var(--app-muted)]">
          Search
          <input
            value={logSearch}
            onChange={(e) => setLogSearch(e.target.value)}
            placeholder="Filter messages"
            className="min-w-0 flex-1 rounded-md border border-[var(--app-border)] bg-[var(--app-bg)] px-2 py-1 text-sm text-[var(--app-text)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)]"
          />
        </label>

        <label className="flex items-center gap-2 text-xs text-[var(--app-muted)]">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
            className="h-4 w-4 rounded border-[var(--app-border)] text-[var(--app-accent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)]"
          />
          Autoscroll
        </label>

        <button
          type="button"
          aria-label="Clear log output"
          className="rounded-md border border-[var(--app-border)] px-2 py-1 text-xs font-medium transition hover:bg-[var(--app-bg)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)]"
          onClick={() => clearLogs()}
        >
          Clear
        </button>
      </div>

      <div
        ref={parentRef}
        data-testid="log-region"
        className="min-h-0 flex-1 overflow-y-auto font-mono text-xs"
        onScroll={onScroll}
      >
        {filtered.length === 0 ? (
          <p className="p-4 text-sm text-[var(--app-muted)]">
            No log lines match the current filters.
          </p>
        ) : (
          <div
            className="relative w-full"
            style={{ height: `${rowVirtualizer.getTotalSize()}px` }}
          >
            {rowVirtualizer.getVirtualItems().map((vi) => {
              const line = filtered[vi.index];
              return (
                <div
                  key={line.id}
                  className="absolute left-0 top-0 w-full border-b border-[var(--app-border)]/60 px-3 py-1"
                  style={{ transform: `translateY(${vi.start}px)` }}
                >
                  <div className="flex items-start gap-2">
                    <span
                      className={
                        "mt-0.5 inline-flex shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide " +
                        levelPillClass(line.level)
                      }
                    >
                      {line.level}
                    </span>
                    <span className="whitespace-pre-wrap break-words text-[var(--app-text)]">
                      {line.message}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
