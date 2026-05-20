import { useState } from "react";

import { appExit, appRelaunchInvoke, appReportFatal } from "@/lib/ipc";
import { useAppStore } from "@/store/appStore";

export default function FatalModal() {
  const fatalMessage = useAppStore((s) => s.fatalMessage);
  const setFatalMessage = useAppStore((s) => s.setFatalMessage);
  const [reporting, setReporting] = useState(false);

  if (!fatalMessage) {
    return null;
  }

  const reportThen = async (action: "quit" | "restart") => {
    setReporting(true);
    try {
      await appReportFatal({
        source: "fatal modal shutdown",
        reason: fatalMessage.split("\n")[0] || "Application error",
        detail: fatalMessage,
      });
    } catch {
      // Shutdown should not be blocked by unavailable reporting credentials/network.
    } finally {
      if (action === "quit") {
        void appExit();
      } else {
        setFatalMessage(null);
        void appRelaunchInvoke();
      }
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="fatal-title"
        className="w-full max-w-md rounded-lg border border-[var(--app-border)] bg-[var(--app-panel)] p-6 shadow-xl"
      >
        <h2 id="fatal-title" className="text-lg font-semibold">
          Application error
        </h2>
        <p className="mt-2 text-sm text-[var(--app-muted)]">{fatalMessage}</p>
        <div className="mt-6 flex justify-end gap-2">
          <button
            type="button"
            className="rounded-md border border-[var(--app-border)] px-3 py-2 text-sm font-medium transition hover:bg-[var(--app-bg)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--app-panel)]"
            disabled={reporting}
            onClick={() => void reportThen("quit")}
          >
            Quit
          </button>
          <button
            type="button"
            className="rounded-md bg-[var(--app-accent)] px-3 py-2 text-sm font-semibold text-white transition hover:opacity-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--app-panel)]"
            disabled={reporting}
            onClick={() => void reportThen("restart")}
          >
            {reporting ? "Reporting..." : "Restart"}
          </button>
        </div>
      </div>
    </div>
  );
}
