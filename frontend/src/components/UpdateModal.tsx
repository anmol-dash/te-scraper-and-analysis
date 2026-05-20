import { useState } from "react";
import { pyRunFull } from "@/lib/ipc";
import { useAppStore } from "@/store/appStore";

function fmtMb(bytes: number): string {
  return `${(bytes / 1_048_576).toFixed(0)} MB`;
}

export default function UpdateModal() {
  const update         = useAppStore((s) => s.releaseUpdate);
  const clearUpdate    = useAppStore((s) => s.clearReleaseUpdate);
  const pushToast      = useAppStore((s) => s.pushToast);

  const [phase, setPhase]     = useState<"idle" | "downloading" | "done" | "error">("idle");
  const [progress, setProgress] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  if (!update) return null;

  const { version, currentVersion, downloadUrl, assetName, sizeBytes } = update;

  async function handleDownload() {
    setPhase("downloading");
    setProgress("Starting download…");
    try {
      const res = await pyRunFull([
        "download-update",
        "--url", downloadUrl,
        "--filename", assetName,
      ]);
      // Collect the last stdout line as progress (it streams via log events)
      const result = res.result as { ok?: boolean; path?: string; error?: string } | undefined;
      if (result?.ok) {
        setPhase("done");
        pushToast(`GAMECA ${version} downloaded — drag it to Applications to finish.`);
      } else {
        setPhase("error");
        setErrorMsg(result?.error ?? "Download failed.");
      }
    } catch (e) {
      setPhase("error");
      setErrorMsg(String(e));
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="update-title"
        className="w-full max-w-sm rounded-lg border border-[var(--app-border)] bg-[var(--app-panel)] p-6 shadow-xl"
      >
        <h2 id="update-title" className="text-base font-semibold text-[var(--app-text)]">
          Update available
        </h2>

        <div className="mt-3 space-y-1 text-sm text-[var(--app-muted)]">
          <p>
            <span className="font-medium text-[var(--app-text)]">New:</span>{" "}
            v{version}
          </p>
          <p>
            <span className="font-medium text-[var(--app-text)]">Current:</span>{" "}
            v{currentVersion}
          </p>
          {sizeBytes > 0 && (
            <p>
              <span className="font-medium text-[var(--app-text)]">Size:</span>{" "}
              {fmtMb(sizeBytes)}
            </p>
          )}
        </div>

        {/* Status area */}
        {phase === "downloading" && (
          <div className="mt-4 rounded-md border border-[var(--app-border)] bg-[var(--app-bg)] p-3 text-xs text-[var(--app-muted)]">
            <p className="font-medium text-[var(--app-text)]">Downloading…</p>
            <p className="mt-1">{progress || "Please wait…"}</p>
            <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-[var(--app-border)]">
              <div className="h-full w-1/3 animate-[slide_1.4s_ease-in-out_infinite] rounded-full bg-[var(--app-accent)]" />
            </div>
          </div>
        )}

        {phase === "done" && (
          <div className="mt-4 rounded-md border border-green-500/30 bg-green-500/10 p-3 text-xs text-green-400">
            <p className="font-semibold">Download complete</p>
            <p className="mt-1">
              The DMG has opened. Drag <strong>GAMECA.app</strong> to your
              Applications folder, then relaunch.
            </p>
          </div>
        )}

        {phase === "error" && (
          <div className="mt-4 rounded-md border border-red-400/40 bg-red-500/10 p-3 text-xs text-red-400">
            <p className="font-semibold">Download failed</p>
            <p className="mt-1 break-words">{errorMsg}</p>
          </div>
        )}

        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={clearUpdate}
            disabled={phase === "downloading"}
            className="rounded-md border border-[var(--app-border)] px-3 py-2 text-sm font-medium text-[var(--app-muted)] transition hover:bg-[var(--app-bg)] disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)]"
          >
            {phase === "done" ? "Close" : "Later"}
          </button>

          {phase === "idle" && (
            <button
              type="button"
              onClick={() => void handleDownload()}
              className="rounded-md bg-[var(--app-accent)] px-3 py-2 text-sm font-semibold text-white transition hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)]"
            >
              Download &amp; Install
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
