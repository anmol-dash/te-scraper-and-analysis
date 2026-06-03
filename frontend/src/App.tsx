import { useEffect, useRef, useState } from "react";

import ArgumentsPanel from "@/components/ArgumentsPanel";
import FatalModal from "@/components/FatalModal";
import FileViewer from "@/components/FileViewer";
import LogStream from "@/components/LogStream";
import NotificationsModal from "@/components/NotificationsModal";
import SetupScreen from "@/components/SetupScreen";
import UpdateModal from "@/components/UpdateModal";
import StatusBar from "@/components/StatusBar";
import ToastStack from "@/components/ToastStack";
import { onCrash, onLog, onProgress, onSetup, onUpdate, pySetup } from "@/lib/ipc";
import type { LogLevel } from "@/lib/logLevel";
import { useAppStore } from "@/store/appStore";

function Splitter({ onMouseDown }: { onMouseDown: (e: React.MouseEvent) => void }) {
  return (
    <div
      onMouseDown={onMouseDown}
      className="w-1 shrink-0 cursor-col-resize bg-[var(--app-border)] hover:bg-[var(--app-accent)] transition-colors"
    />
  );
}

function coerceLevel(level: string): LogLevel {
  if (level === "warn" || level === "error") {
    return level;
  }
  return "info";
}

export default function App() {
  const fileViewerOpen = useAppStore((s) => s.fileViewerOpen);
  const setupPhase     = useAppStore((s) => s.setupPhase);

  const [leftW,  setLeftW]  = useState(320);
  const [rightW, setRightW] = useState(480);
  const drag = useRef<{ side: "left" | "right"; startX: number; startW: number } | null>(null);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!drag.current) return;
      const { side, startX, startW } = drag.current;
      const delta = e.clientX - startX;
      if (side === "left") {
        setLeftW(Math.max(180, Math.min(600, startW + delta)));
      } else {
        setRightW(Math.max(280, Math.min(800, startW - delta)));
      }
    };
    const onUp = () => { drag.current = null; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  const startDrag = (side: "left" | "right") => (e: React.MouseEvent) => {
    e.preventDefault();
    drag.current = { side, startX: e.clientX, startW: side === "left" ? leftW : rightW };
  };
  const setSetupEvent  = useAppStore((s) => s.setSetupEvent);
  const pushToast        = useAppStore((s) => s.pushToast);
  const setReleaseUpdate = useAppStore((s) => s.setReleaseUpdate);
  const setFatalMessage  = useAppStore((s) => s.setFatalMessage);

  useEffect(() => {
    const pushLog     = useAppStore.getState().pushLog;
    const setProgress = useAppStore.getState().setProgress;

    const unsubLog = onLog((e) => {
      pushLog({
        level: coerceLevel(e.level),
        message: e.message,
        ts: e.ts ?? Date.now(),
        source: e.source,
      });
      if (e.message.startsWith("[JOB DONE]")) {
        useAppStore.getState().pushToast(e.message.replace("[JOB DONE] ", ""));
      }
    });

    const unsubProgress = onProgress((p) => {
      setProgress(p);
    });

    const unsubSetup = onSetup((e) => {
      setSetupEvent(e.phase, e.fraction ?? 0, e.message ?? "");
    });

    const unsubUpdate = onUpdate((e) => {
      if (e.phase === "done" && e.updated && e.updated.length > 0) {
        pushToast(`Updated to ${e.sha ?? "latest"} (${e.updated.length} file${e.updated.length === 1 ? "" : "s"})`);
      }
      if (
        e.phase === "release_available" &&
        e.version && e.current_version && e.download_url && e.asset_name
      ) {
        setReleaseUpdate({
          version: e.version,
          currentVersion: e.current_version,
          downloadUrl: e.download_url,
          assetName: e.asset_name,
          sizeBytes: e.size_bytes ?? 0,
        });
      }
    });

    const unsubCrash = onCrash((e) => {
      if (!e.fatal) {
        pushToast(`Python sidecar restarted${e.retry ? ` (${e.retry})` : ""}`);
        return;
      }
      const reason = e.reason ?? "fatal sidecar error";
      const detail = e.detail ? `\n\n${e.detail}` : "";
      setFatalMessage(`${reason}${detail}`);
    });

    // Explicitly ask the sidecar for setup status now that we're subscribed.
    // This eliminates the race where the sidecar emitted "done" before
    // the listener was registered (e.g. on repeat launches).
    void pySetup()
      .then((res) => {
        if (res.done) {
          setSetupEvent("done", 1.0, "Environment ready.");
        }
      })
      .catch(() => {
        // Bridge not ready yet — setup events will arrive via the listener
        // once the sidecar connects.
      });

    return () => {
      unsubLog();
      unsubProgress();
      unsubSetup();
      unsubUpdate();
      unsubCrash();
    };
  }, [setSetupEvent, pushToast, setReleaseUpdate, setFatalMessage]);

  // Show the setup overlay until the venv is confirmed ready.
  if (setupPhase !== "done") {
    return (
      <>
        <SetupScreen />
        <FatalModal />
        <UpdateModal />
      </>
    );
  }

  return (
    <div className="flex h-screen flex-col bg-[var(--app-bg)] text-[var(--app-text)]">
      <StatusBar />
      <div className="flex min-h-0 flex-1 select-none">
        <ArgumentsPanel width={leftW} />
        <Splitter onMouseDown={startDrag("left")} />
        <LogStream />
        {fileViewerOpen && (
          <>
            <Splitter onMouseDown={startDrag("right")} />
            <FileViewer width={rightW} />
          </>
        )}
      </div>
      <ToastStack />
      <NotificationsModal />
      <FatalModal />
      <UpdateModal />
    </div>
  );
}
