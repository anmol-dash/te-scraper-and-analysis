import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export type RunResult = { id: string };

export type LogEvent = {
  level: string;
  message: string;
  ts?: number;
  source?: "stdout" | "stderr";
};

export type ProgressEvent = {
  fraction?: number;
  current?: number;
  total?: number;
  message?: string;
};

export type RunFinishedEvent = { id: string; code: number };

export type CrashEvent = {
  fatal?: boolean;
  reason?: string;
  detail?: string;
  retry?: number;
  phase?: string;
  had_valid_stdout?: boolean;
};

export async function pyRun(argv: string[]): Promise<RunResult> {
  const raw = await invoke<{ id: string; result?: unknown }>("py_run", {
    argv,
  });
  return { id: raw.id };
}

export async function pyRunFull(
  argv: string[],
): Promise<{ id: string; result?: unknown }> {
  return invoke<{ id: string; result?: unknown }>("py_run", { argv });
}

export interface RemoteEntry {
  name: string;
  path: string;
  size: number;
  is_dir: boolean;
  mtime: number;
}

export interface DirListing {
  ok?: boolean;
  path?: string;
  entries?: RemoteEntry[];
}

export interface FileContent {
  ok?: boolean;
  type?: "text" | "binary";
  mime?: string;
  data?: string;   // base64 for binary
  text?: string;   // for text files
  size?: number;
  error?: string;
}

export async function hpcListDir(path: string): Promise<DirListing> {
  const raw = await pyRunFull(["hpc-list-dir", "--path", path]);
  return (raw.result ?? {}) as DirListing;
}

export async function hpcReadFile(path: string, maxMb = 10): Promise<FileContent> {
  const raw = await pyRunFull(["hpc-read-file", "--path", path, "--max-mb", String(maxMb)]);
  return (raw.result ?? {}) as FileContent;
}

export interface DownloadResult {
  ok?: boolean;
  local_path?: string;
  error?: string;
}

export async function hpcDownloadFile(remotePath: string, localPath: string): Promise<DownloadResult> {
  const raw = await pyRunFull(["hpc-download-file", "--remote-path", remotePath, "--local-path", localPath]);
  return (raw.result ?? {}) as DownloadResult;
}

export async function hpcDownloadDir(remotePath: string, localPath: string): Promise<DownloadResult> {
  const raw = await pyRunFull(["hpc-download-dir", "--remote-path", remotePath, "--local-path", localPath]);
  return (raw.result ?? {}) as DownloadResult;
}

export async function pyCancel(id: string): Promise<void> {
  return invoke("py_cancel", { id });
}

export async function pyPing(): Promise<{ ok: boolean; version?: string }> {
  const raw = await invoke<{ ok?: boolean; version?: string }>("py_ping");
  return { ok: Boolean(raw?.ok), version: raw?.version };
}

export async function appExit(): Promise<void> {
  return invoke("app_exit");
}

export async function appReportFatal(args: {
  source: string;
  reason: string;
  detail: string;
}): Promise<{ ok: boolean; url?: string | null }> {
  const raw = await invoke<{ ok?: boolean; url?: string | null }>("app_report_fatal", args);
  return { ok: Boolean(raw?.ok), url: raw?.url ?? null };
}

export async function pySetup(): Promise<{ ok: boolean; done: boolean }> {
  const raw = await invoke<{ ok?: boolean; done?: boolean }>("py_setup");
  return { ok: Boolean(raw?.ok), done: Boolean(raw?.done) };
}

export type SetupEvent = {
  phase: "checking" | "creating" | "installing" | "done" | "error";
  fraction?: number;
  message?: string;
};

export type UpdateEvent = {
  phase: "checking" | "downloading" | "done" | "up_to_date" | "skipped" | "offline" | "log" | "release_available";
  message?: string;
  sha?: string;
  updated?: string[];
  failed?: string[];
  // release_available fields
  version?: string;
  current_version?: string;
  download_url?: string;
  asset_name?: string;
  size_bytes?: number;
};

export async function appRelaunchInvoke(): Promise<void> {
  return invoke("app_relaunch");
}

function chainListen<T>(
  event: string,
  handler: (payload: T) => void,
  disposedRef: { current: boolean },
  bucket: UnlistenFn[],
): void {
  void listen<T>(event, (e) => {
    if (!disposedRef.current) {
      handler(e.payload);
    }
  }).then((un) => {
    if (disposedRef.current) {
      un();
    } else {
      bucket.push(un);
    }
  });
}

function chainListenUnknown(
  event: string,
  handler: (payload: unknown) => void,
  disposedRef: { current: boolean },
  bucket: UnlistenFn[],
): void {
  void listen<unknown>(event, (e) => {
    if (!disposedRef.current) {
      handler(e.payload);
    }
  }).then((un) => {
    if (disposedRef.current) {
      un();
    } else {
      bucket.push(un);
    }
  });
}

function normalizeLogPayload(payload: unknown): LogEvent {
  if (typeof payload === "object" && payload !== null) {
    const p = payload as Record<string, unknown>;
    if (typeof p.message === "string") {
      return {
        level: typeof p.level === "string" ? p.level : "info",
        message: p.message,
        ts: typeof p.ts === "number" ? p.ts : undefined,
        source:
          p.source === "stdout" || p.source === "stderr" ? p.source : undefined,
      };
    }
    const inner = p.payload as Record<string, unknown> | undefined;
    if (inner && typeof inner.line === "string") {
      return {
        level:
          typeof inner.level === "string" && inner.level === "logging"
            ? "info"
            : "info",
        message: inner.line,
        source: "stdout",
      };
    }
  }
  return {
    level: "info",
    message: typeof payload === "string" ? payload : JSON.stringify(payload),
  };
}

export function onLog(cb: (e: LogEvent) => void): UnlistenFn {
  const disposedRef = { current: false };
  const unsubs: UnlistenFn[] = [];

  chainListenUnknown(
    "python://log",
    (payload) => {
      cb(normalizeLogPayload(payload));
    },
    disposedRef,
    unsubs,
  );
  chainListen<{ message: string; ts?: number }>(
    "python://stderr",
    (payload) => {
      cb({
        level: "error",
        message: payload.message,
        ts: payload.ts,
        source: "stderr",
      });
    },
    disposedRef,
    unsubs,
  );

  return () => {
    disposedRef.current = true;
    unsubs.forEach((u) => u());
    unsubs.length = 0;
  };
}

export function onProgress(cb: (e: ProgressEvent) => void): UnlistenFn {
  const disposedRef = { current: false };
  const unsubs: UnlistenFn[] = [];
  chainListen<ProgressEvent>("python://progress", cb, disposedRef, unsubs);
  return () => {
    disposedRef.current = true;
    unsubs.forEach((u) => u());
    unsubs.length = 0;
  };
}

export function onUpdate(cb: (e: UpdateEvent) => void): UnlistenFn {
  const disposedRef = { current: false };
  const unsubs: UnlistenFn[] = [];
  chainListen<UpdateEvent>("python://update", cb, disposedRef, unsubs);
  return () => {
    disposedRef.current = true;
    unsubs.forEach((u) => u());
    unsubs.length = 0;
  };
}

export function onSetup(cb: (e: SetupEvent) => void): UnlistenFn {
  const disposedRef = { current: false };
  const unsubs: UnlistenFn[] = [];
  chainListen<SetupEvent>("python://setup", cb, disposedRef, unsubs);
  return () => {
    disposedRef.current = true;
    unsubs.forEach((u) => u());
    unsubs.length = 0;
  };
}

export function onCrash(cb: (e: CrashEvent) => void): UnlistenFn {
  const disposedRef = { current: false };
  const unsubs: UnlistenFn[] = [];
  chainListen<CrashEvent>("python://crashed", cb, disposedRef, unsubs);
  return () => {
    disposedRef.current = true;
    unsubs.forEach((u) => u());
    unsubs.length = 0;
  };
}

export function onRunFinished(cb: (e: RunFinishedEvent) => void): UnlistenFn {
  const disposedRef = { current: false };
  const unsubs: UnlistenFn[] = [];
  chainListen<RunFinishedEvent>(
    "python://run_finished",
    cb,
    disposedRef,
    unsubs,
  );
  return () => {
    disposedRef.current = true;
    unsubs.forEach((u) => u());
    unsubs.length = 0;
  };
}
