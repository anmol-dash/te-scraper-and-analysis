import { appExit, appRelaunchInvoke } from "@/lib/ipc";
import { useAppStore } from "@/store/appStore";

export default function FatalModal() {
  const fatalMessage = useAppStore((s) => s.fatalMessage);
  const setFatalMessage = useAppStore((s) => s.setFatalMessage);

  if (!fatalMessage) {
    return null;
  }

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
            onClick={() => void appExit()}
          >
            Quit
          </button>
          <button
            type="button"
            className="rounded-md bg-[var(--app-accent)] px-3 py-2 text-sm font-semibold text-white transition hover:opacity-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--app-panel)]"
            onClick={() => {
              setFatalMessage(null);
              void appRelaunchInvoke();
            }}
          >
            Restart
          </button>
        </div>
      </div>
    </div>
  );
}
