import { useEffect } from "react";
import { useAppStore } from "@/store/appStore";

function ToastItem({
  id,
  message,
  onDismiss,
}: {
  id: string;
  message: string;
  onDismiss: (id: string) => void;
}) {
  useEffect(() => {
    const t = window.setTimeout(() => onDismiss(id), 4_500);
    return () => window.clearTimeout(t);
  }, [id, message, onDismiss]);

  return (
    <div className="pointer-events-auto rounded-md border border-[var(--app-border)] bg-[var(--app-panel)] p-3 text-sm shadow-lg">
      <div className="flex items-start gap-2">
        <p className="flex-1">{message}</p>
        <button
          type="button"
          aria-label="Dismiss notification"
          className="rounded px-2 py-1 text-xs font-medium text-[var(--app-muted)] transition hover:bg-[var(--app-bg)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)]"
          onClick={() => onDismiss(id)}
        >
          Close
        </button>
      </div>
    </div>
  );
}

export default function ToastStack() {
  const toasts = useAppStore((s) => s.toasts);
  const dismissToast = useAppStore((s) => s.dismissToast);

  if (toasts.length === 0) {
    return null;
  }

  return (
    <div
      className="pointer-events-none fixed bottom-4 right-4 z-40 flex w-[min(24rem,calc(100vw-2rem))] flex-col gap-2"
      aria-live="polite"
    >
      {toasts.map((t) => (
        <ToastItem
          key={t.id}
          id={t.id}
          message={t.message}
          onDismiss={dismissToast}
        />
      ))}
    </div>
  );
}
