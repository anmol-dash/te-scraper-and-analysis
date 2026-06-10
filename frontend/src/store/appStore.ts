import { create } from "zustand";
import type { ProgressEvent } from "@/lib/ipc";
import type { LogLevel } from "@/lib/logLevel";

export type RunPhase = "idle" | "running" | "done" | "errored";

export type SetupPhase = "checking" | "creating" | "installing" | "done" | "error" | "log";

export type LogLine = {
  id: string;
  level: LogLevel;
  message: string;
  ts: number;
  source?: "stdout" | "stderr";
};

const LOG_CAP = 10_000;

type Toast = { id: string; message: string };

type AppStore = {
  logs: LogLine[];
  suppressLogs: boolean;
  setSuppressLogs: (v: boolean) => void;
  pushLog: (line: Omit<LogLine, "id"> & { id?: string }) => void;
  clearLogs: () => void;

  logLevelFilter: "all" | LogLevel;
  setLogLevelFilter: (v: "all" | LogLevel) => void;
  logSearch: string;
  setLogSearch: (v: string) => void;

  progress: ProgressEvent | null;
  setProgress: (p: ProgressEvent | null) => void;

  runPhase: RunPhase;
  runId: string | null;
  runStartedAt: number | null;
  runElapsedLabel: string;
  setRunElapsedLabel: (v: string) => void;
  beginRun: (id: string) => void;
  finishRun: (phase: "done" | "errored") => void;
  resetRun: () => void;

  fatalMessage: string | null;
  setFatalMessage: (v: string | null) => void;

  toasts: Toast[];
  pushToast: (message: string) => void;
  dismissToast: (id: string) => void;

  hpcConnected: boolean;
  hpcHost: string;
  hpcScheduler: string;
  hpcHome: string;
  hpcHasInternet: boolean;
  setHpcConnection: (info: { host: string; scheduler: string; home: string; hasInternet?: boolean } | null) => void;

  hpcJobId: string | null;
  setHpcJobId: (id: string | null) => void;

  fileViewerOpen: boolean;
  toggleFileViewer: () => void;

  notificationsOpen: boolean;
  toggleNotifications: () => void;

  mode: "local" | "hpc";
  setMode: (m: "local" | "hpc") => void;

  releaseUpdate: { version: string; currentVersion: string; downloadUrl: string; assetName: string; sizeBytes: number } | null;
  setReleaseUpdate: (u: AppStore["releaseUpdate"]) => void;
  clearReleaseUpdate: () => void;

  setupPhase: SetupPhase | null;
  setupFraction: number;
  setupMessage: string;
  setupError: string;
  setupLogs: string[];
  setSetupEvent: (phase: SetupPhase, fraction: number, message: string) => void;
};

export const useAppStore = create<AppStore>((set) => ({
  logs: [],
  suppressLogs: false,
  setSuppressLogs: (v) => set({ suppressLogs: v }),
  pushLog: (line) =>
    set((s) => {
      if (s.suppressLogs) return {};
      const id = line.id ?? globalThis.crypto.randomUUID();
      const nextLine: LogLine = {
        id,
        level: line.level,
        message: line.message,
        ts: line.ts ?? Date.now(),
        source: line.source,
      };
      const logs = [...s.logs, nextLine];
      return { logs: logs.length > LOG_CAP ? logs.slice(-LOG_CAP) : logs };
    }),
  clearLogs: () => set({ logs: [] }),

  logLevelFilter: "all",
  setLogLevelFilter: (v) => set({ logLevelFilter: v }),
  logSearch: "",
  setLogSearch: (v) => set({ logSearch: v }),

  progress: null,
  setProgress: (p) => set({ progress: p }),

  runPhase: "idle",
  runId: null,
  runStartedAt: null,
  runElapsedLabel: "0.0s",
  setRunElapsedLabel: (v) => set({ runElapsedLabel: v }),
  beginRun: (id) =>
    set({
      runPhase: "running",
      runId: id,
      runStartedAt: Date.now(),
      progress: null,
      runElapsedLabel: "0.0s",
    }),
  finishRun: (phase) =>
    set({
      runPhase: phase,
      progress: null,
    }),
  resetRun: () =>
    set({
      runPhase: "idle",
      runId: null,
      runStartedAt: null,
      progress: null,
      runElapsedLabel: "0.0s",
    }),

  fatalMessage: null,
  setFatalMessage: (v) => set({ fatalMessage: v }),

  toasts: [],
  pushToast: (message) =>
    set((s) => ({
      toasts: [
        ...s.toasts,
        { id: globalThis.crypto.randomUUID(), message },
      ].slice(-4),
    })),
  dismissToast: (id) =>
    set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),

  hpcConnected: false,
  hpcHost: "",
  hpcScheduler: "",
  hpcHome: "",
  hpcHasInternet: false,
  setHpcConnection: (info) =>
    set(
      info
        ? { hpcConnected: true, hpcHost: info.host, hpcScheduler: info.scheduler, hpcHome: info.home, hpcHasInternet: info.hasInternet ?? false }
        : { hpcConnected: false, hpcHost: "", hpcScheduler: "", hpcHome: "", hpcHasInternet: false },
    ),

  hpcJobId: null,
  setHpcJobId: (id) => set({ hpcJobId: id }),

  fileViewerOpen: false,
  toggleFileViewer: () => set((s) => ({ fileViewerOpen: !s.fileViewerOpen })),

  notificationsOpen: false,
  toggleNotifications: () => set((s) => ({ notificationsOpen: !s.notificationsOpen })),

  mode: "local",
  setMode: (m) => set({ mode: m }),

  releaseUpdate: null,
  setReleaseUpdate: (u) => set({ releaseUpdate: u }),
  clearReleaseUpdate: () => set({ releaseUpdate: null }),

  setupPhase: null,
  setupFraction: 0,
  setupMessage: "",
  setupError: "",
  setupLogs: [],
  setSetupEvent: (phase, fraction, message) =>
    set((s) => {
      // "log" events only append a line — they don't touch phase/progress.
      if (phase === "log") {
        return { setupLogs: message ? [...s.setupLogs, message] : s.setupLogs };
      }
      return {
        setupPhase: phase,
        setupFraction: fraction,
        setupMessage: message,
        setupError: phase === "error" ? message : s.setupError,
        setupLogs: message ? [...s.setupLogs, message] : s.setupLogs,
      };
    }),
}));
