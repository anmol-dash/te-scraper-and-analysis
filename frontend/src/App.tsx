import { useEffect } from "react";

import ArgumentsPanel from "@/components/ArgumentsPanel";
import FatalModal from "@/components/FatalModal";
import FileViewer from "@/components/FileViewer";
import LogStream from "@/components/LogStream";
import SetupScreen from "@/components/SetupScreen";
import UpdateModal from "@/components/UpdateModal";
import StatusBar from "@/components/StatusBar";
import ToastStack from "@/components/ToastStack";
import { onCrash, onLog, onProgress, onSetup, onUpdate, pySetup } from "@/lib/ipc";
import type { LogLevel } from "@/lib/logLevel";
import { useAppStore } from "@/store/appStore";

function coerceLevel(level: string): LogLevel {
  if (level === "warn" || level === "error") {
    return level;
  }
  return "info";
}

export default function App() {
  const fileViewerOpen = useAppStore((s) => s.fileViewerOpen);
  const setupPhase     = useAppStore((s) => s.setupPhase);
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
      <div className="flex min-h-0 flex-1">
        <ArgumentsPanel />
        <LogStream />
        {fileViewerOpen && <FileViewer />}
      </div>
      <ToastStack />
      <FatalModal />
      <UpdateModal />
    </div>
  );
}
