import { useEffect, useState } from "react";
import { notifyGet, notifySet } from "@/lib/ipc";
import { useAppStore } from "@/store/appStore";

export default function NotificationsModal() {
  const open = useAppStore((s) => s.notificationsOpen);
  const toggle = useAppStore((s) => s.toggleNotifications);
  const pushToast = useAppStore((s) => s.pushToast);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [configured, setConfigured] = useState(false);
  const [saving, setSaving] = useState(false);
  const [statusMsg, setStatusMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);

  // Load saved email when modal opens
  useEffect(() => {
    if (!open) return;
    setStatusMsg(null);
    setPassword("");
    notifyGet()
      .then((res) => {
        setEmail(res.email);
        setConfigured(res.configured);
      })
      .catch(() => {});
  }, [open]);

  if (!open) return null;

  async function handleSave() {
    if (!email.trim()) { setStatusMsg({ kind: "err", text: "Gmail address is required." }); return; }
    if (!password.trim()) { setStatusMsg({ kind: "err", text: "App password is required." }); return; }
    setSaving(true);
    setStatusMsg(null);
    try {
      const res = await notifySet(email.trim(), password.trim());
      if (res.ok) {
        setConfigured(true);
        setPassword("");
        setStatusMsg({ kind: "ok", text: "Credentials saved. Notifications are enabled." });
        pushToast("Email notifications enabled.");
      } else {
        setStatusMsg({ kind: "err", text: res.error ?? "Save failed." });
      }
    } catch (e) {
      setStatusMsg({ kind: "err", text: String(e) });
    } finally {
      setSaving(false);
    }
  }

  const inp =
    "w-full rounded-md border border-[var(--app-border)] bg-[var(--app-bg)] px-2 py-1.5 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)] disabled:opacity-50";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="presentation"
      onClick={(e) => { if (e.target === e.currentTarget) toggle(); }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="notify-title"
        className="w-full max-w-sm rounded-lg border border-[var(--app-border)] bg-[var(--app-panel)] p-6 shadow-xl"
      >
        <div className="flex items-center justify-between">
          <h2 id="notify-title" className="text-base font-semibold text-[var(--app-text)]">
            Email notifications
          </h2>
          <button
            type="button"
            onClick={toggle}
            className="rounded p-1 text-[var(--app-muted)] hover:text-[var(--app-text)] transition"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <p className="mt-2 text-xs text-[var(--app-muted)]">
          Receive an email when any pipeline step completes. Uses Gmail with an
          App Password — your regular password is never stored.
        </p>

        {/* Setup status badge */}
        <div className={`mt-3 flex items-center gap-2 rounded-md px-3 py-2 text-xs font-medium ${
          configured
            ? "border border-green-500/30 bg-green-500/10 text-green-400"
            : "border border-[var(--app-border)] bg-[var(--app-bg)] text-[var(--app-muted)]"
        }`}>
          <span className={`h-2 w-2 shrink-0 rounded-full ${configured ? "bg-green-400" : "bg-[var(--app-muted)]"}`} />
          {configured ? `Configured — sending from ${email}` : "Not configured"}
        </div>

        <div className="mt-4 space-y-3">
          <div className="space-y-1">
            <label className="block text-xs font-medium">Gmail address</label>
            <input
              className={inp}
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@gmail.com"
              disabled={saving}
            />
          </div>

          <div className="space-y-1">
            <label className="block text-xs font-medium">
              App Password
              <span className="ml-1 font-normal text-[var(--app-muted)]">(16 chars, not your Gmail password)</span>
            </label>
            <input
              className={inp}
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={configured ? "Enter to replace saved password" : "xxxx xxxx xxxx xxxx"}
              disabled={saving}
              onKeyDown={(e) => { if (e.key === "Enter" && !saving) void handleSave(); }}
            />
          </div>

          <a
            href="https://myaccount.google.com/apppasswords"
            target="_blank"
            rel="noreferrer"
            className="block text-xs text-[var(--app-accent)] hover:underline"
          >
            Generate an App Password at myaccount.google.com/apppasswords →
          </a>
        </div>

        {/* Status message */}
        {statusMsg && (
          <div className={`mt-3 rounded-md px-3 py-2 text-xs ${
            statusMsg.kind === "ok"
              ? "border border-green-500/30 bg-green-500/10 text-green-400"
              : "border border-red-400/30 bg-red-500/10 text-red-400"
          }`}>
            {statusMsg.text}
          </div>
        )}

        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={toggle}
            className="rounded-md border border-[var(--app-border)] px-3 py-2 text-sm font-medium text-[var(--app-muted)] transition hover:bg-[var(--app-bg)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)]"
          >
            Close
          </button>
          <button
            type="button"
            onClick={() => void handleSave()}
            disabled={saving}
            className="rounded-md bg-[var(--app-accent)] px-3 py-2 text-sm font-semibold text-white transition hover:opacity-90 disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)]"
          >
            {saving ? "Saving…" : "Save credentials"}
          </button>
        </div>
      </div>
    </div>
  );
}
