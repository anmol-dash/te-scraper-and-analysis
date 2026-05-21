import { save } from "@tauri-apps/plugin-dialog";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { hpcListDir, hpcReadFile, hpcDownloadFile, hpcDownloadDir, type RemoteEntry } from "@/lib/ipc";
import { useAppStore } from "@/store/appStore";

// ── helpers ──────────────────────────────────────────────────────────────────

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fileIcon(entry: RemoteEntry): string {
  if (entry.is_dir) return "📁";
  const ext = entry.name.toLowerCase().split(".").pop() ?? "";
  if (["png","jpg","jpeg","gif","bmp","webp","svg"].includes(ext)) return "🖼";
  if (["csv","tsv"].includes(ext)) return "📊";
  if (ext === "html" || ext === "htm") return "🌐";
  if (["py","sh","r","rmd"].includes(ext)) return "📝";
  if (["log","out","err"].includes(ext)) return "📋";
  if (ext === "pdf") return "📄";
  return "📄";
}

function csvToTable(text: string): { headers: string[]; rows: string[][] } {
  const lines = text.trimEnd().split("\n").map((l) => l.split(",").map((c) => c.trim().replace(/^"|"$/g, "")));
  if (lines.length === 0) return { headers: [], rows: [] };
  return { headers: lines[0], rows: lines.slice(1, 1001) };
}

// ── context menu ──────────────────────────────────────────────────────────────

type CtxMenu = { x: number; y: number; entry: RemoteEntry };

function ContextMenu({
  menu,
  onClose,
  onDownload,
}: {
  menu: CtxMenu;
  onClose: () => void;
  onDownload: (entry: RemoteEntry) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  return (
    <div
      ref={ref}
      style={{ top: menu.y, left: menu.x }}
      className="fixed z-50 min-w-[140px] rounded-md border border-[var(--app-border)] bg-[var(--app-panel)] py-1 shadow-lg text-xs"
    >
      <button
        type="button"
        className="w-full px-3 py-1.5 text-left hover:bg-[var(--app-bg)] transition"
        onClick={() => { onDownload(menu.entry); onClose(); }}
      >
        Download{menu.entry.is_dir ? " as .tar.gz" : ""}
      </button>
    </div>
  );
}

// ── HTML frame (blob URL so WKWebView renders Plotly correctly) ───────────────

function HtmlFrame({ html, title }: { html: string; title: string }) {
  const blobUrl = useMemo(() => {
    const blob = new Blob([html], { type: "text/html" });
    return URL.createObjectURL(blob);
  }, [html]);

  useEffect(() => () => URL.revokeObjectURL(blobUrl), [blobUrl]);

  return (
    <iframe
      src={blobUrl}
      className="w-full border-0"
      style={{ height: "calc(100vh - 120px)", minHeight: "480px" }}
      title={title}
    />
  );
}

// ── file content view ─────────────────────────────────────────────────────────

type ViewFile = {
  path: string;
  name: string;
  loading: boolean;
  error: string | null;
  type: "image" | "csv" | "html" | "text" | null;
  src?: string;
  text?: string;
  headers?: string[];
  rows?: string[][];
};

function ContentView({
  file,
  onBack,
  onDownload,
}: {
  file: ViewFile;
  onBack: () => void;
  onDownload: () => void;
}) {
  // Header always visible so Download is clickable even while loading
  const header = (
    <div className="flex shrink-0 items-center gap-2 border-b border-[var(--app-border)] px-3 py-1.5">
      <button type="button" onClick={onBack} className="text-xs text-[var(--app-accent)] hover:underline">← Back</button>
      <span className="min-w-0 flex-1 truncate text-xs font-medium">{file.name}</span>
      <button
        type="button"
        onClick={onDownload}
        className="shrink-0 rounded border border-[var(--app-border)] px-2 py-0.5 text-xs hover:bg-[var(--app-bg)] transition"
        title="Download file"
      >
        ↓ Download
      </button>
    </div>
  );

  if (file.loading) {
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        {header}
        <div className="flex flex-1 items-center justify-center gap-3 text-xs text-[var(--app-muted)]">
          <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-[var(--app-accent)] border-t-transparent" />
          Loading {file.name}…
        </div>
      </div>
    );
  }

  if (file.error) {
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        {header}
        <div className="flex flex-1 flex-col items-center justify-center gap-2 p-4 text-center">
          <p className="text-xs text-red-500">{file.error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {header}

      <div className="min-h-0 flex-1 overflow-auto">
        {file.type === "image" && (
          <div className="flex items-center justify-center p-2">
            <img src={file.src} alt={file.name} className="max-w-full object-contain" />
          </div>
        )}

        {file.type === "csv" && (
          <table className="min-w-full border-collapse text-xs">
            <thead className="sticky top-0 bg-[var(--app-panel)]">
              <tr>
                {file.headers?.map((h, i) => (
                  <th key={i} className="border border-[var(--app-border)] px-2 py-1 text-left font-medium whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {file.rows?.map((row, ri) => (
                <tr key={ri} className={ri % 2 === 0 ? "bg-[var(--app-bg)]" : ""}>
                  {row.map((cell, ci) => (
                    <td key={ci} className="border border-[var(--app-border)] px-2 py-0.5 whitespace-nowrap max-w-[200px] truncate">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {file.type === "html" && <HtmlFrame html={file.text ?? ""} title={file.name} />}

        {file.type === "text" && (
          <pre className="whitespace-pre-wrap break-words p-3 font-mono text-xs leading-relaxed">
            {file.text}
            {(file.text?.length ?? 0) >= 2 * 1024 * 1024 - 1 && (
              <span className="block mt-2 text-[var(--app-muted)] italic">— truncated at 2 MB —</span>
            )}
          </pre>
        )}
      </div>
    </div>
  );
}

// ── directory listing ─────────────────────────────────────────────────────────

type SortField = "name" | "date";
type SortDir   = "asc" | "desc";

function sortEntries(entries: RemoteEntry[], field: SortField, dir: SortDir): RemoteEntry[] {
  const dirs  = entries.filter((e) => e.is_dir);
  const files = entries.filter((e) => !e.is_dir);
  const cmp = (a: RemoteEntry, b: RemoteEntry) => {
    const v = field === "name"
      ? a.name.localeCompare(b.name, undefined, { sensitivity: "base" })
      : a.mtime - b.mtime;
    return dir === "asc" ? v : -v;
  };
  return [...dirs.sort(cmp), ...files.sort(cmp)];
}

function DirList({
  entries,
  loading,
  error,
  onNavigate,
  onOpenFile,
  onContextMenu,
}: {
  entries: RemoteEntry[];
  loading: boolean;
  error: string | null;
  onNavigate: (p: string) => void;
  onOpenFile: (entry: RemoteEntry) => void;
  onContextMenu: (e: React.MouseEvent, entry: RemoteEntry) => void;
}) {
  const [sortField, setSortField] = useState<SortField>("name");
  const [sortDir,   setSortDir]   = useState<SortDir>("asc");

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir(field === "date" ? "desc" : "asc");
    }
  };

  const sorted = sortEntries(entries, sortField, sortDir);
  const arrow  = (field: SortField) =>
    sortField === field ? (sortDir === "asc" ? " ↑" : " ↓") : "";

  if (loading) {
    return <div className="flex flex-1 items-center justify-center text-xs text-[var(--app-muted)]">Loading…</div>;
  }
  if (error) {
    return <div className="p-3 text-xs text-red-500">{error}</div>;
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Sort controls */}
      <div className="flex shrink-0 gap-1 border-b border-[var(--app-border)] px-3 py-1">
        <button
          type="button"
          onClick={() => toggleSort("name")}
          className={`rounded px-2 py-0.5 text-[10px] font-medium transition ${
            sortField === "name"
              ? "bg-[var(--app-accent)] text-white"
              : "text-[var(--app-muted)] hover:text-[var(--app-text)]"
          }`}
        >
          Name{arrow("name")}
        </button>
        <button
          type="button"
          onClick={() => toggleSort("date")}
          className={`rounded px-2 py-0.5 text-[10px] font-medium transition ${
            sortField === "date"
              ? "bg-[var(--app-accent)] text-white"
              : "text-[var(--app-muted)] hover:text-[var(--app-text)]"
          }`}
        >
          Date{arrow("date")}
        </button>
        <span className="ml-auto text-[10px] text-[var(--app-muted)]">
          {entries.length} item{entries.length !== 1 ? "s" : ""}
        </span>
      </div>

      {entries.length === 0 ? (
        <div className="p-3 text-xs text-[var(--app-muted)]">Empty directory</div>
      ) : (
        <div className="min-h-0 flex-1 overflow-y-auto">
          {sorted.map((e) => (
            <button
              key={e.path}
              type="button"
              className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs hover:bg-[var(--app-bg)] transition"
              onClick={() => (e.is_dir ? onNavigate(e.path) : onOpenFile(e))}
              onContextMenu={(ev) => onContextMenu(ev, e)}
            >
              <span className="shrink-0">{fileIcon(e)}</span>
              <span className="min-w-0 flex-1 truncate">{e.name}{e.is_dir ? "/" : ""}</span>
              {!e.is_dir && (
                <span className="shrink-0 text-[var(--app-muted)]">{fmtSize(e.size)}</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ── main component ────────────────────────────────────────────────────────────

export default function FileViewer() {
  const hpcHome = useAppStore((s) => s.hpcHome);
  const hpcConnected = useAppStore((s) => s.hpcConnected);
  const pushToast = useAppStore((s) => s.pushToast);

  const [path, setPath] = useState("");
  const [pathInput, setPathInput] = useState("");
  const [editingPath, setEditingPath] = useState(false);
  const [entries, setEntries] = useState<RemoteEntry[]>([]);
  const [dirLoading, setDirLoading] = useState(false);
  const [dirError, setDirError] = useState<string | null>(null);
  const [openFile, setOpenFile] = useState<ViewFile | null>(null);
  const [ctxMenu, setCtxMenu] = useState<CtxMenu | null>(null);
  const pathInputRef = useRef<HTMLInputElement>(null);

  const navigate = useCallback(async (p: string) => {
    setOpenFile(null);
    setPath(p);
    setPathInput(p);
    setEditingPath(false);
    setDirLoading(true);
    setDirError(null);
    try {
      const result = await hpcListDir(p);
      if (result.ok) {
        setEntries(result.entries ?? []);
      } else {
        setDirError("Failed to list directory");
      }
    } catch (e) {
      setDirError(e instanceof Error ? e.message : String(e));
    } finally {
      setDirLoading(false);
    }
  }, []);

  const goUp = useCallback(() => {
    const parent = path.split("/").slice(0, -1).join("/") || hpcHome || "~";
    void navigate(parent);
  }, [path, hpcHome, navigate]);

  useEffect(() => {
    if (hpcConnected && hpcHome) {
      void navigate(hpcHome);
    }
  }, [hpcConnected, hpcHome, navigate]);

  const openFileForViewing = useCallback(async (entry: RemoteEntry) => {
    const ext = entry.name.toLowerCase().split(".").pop() ?? "";
    // HTML files (Plotly/Bokeh plots) can be 10-20 MB; text files cap at 2 MB
    const maxMb = (ext === "html" || ext === "htm") ? 20 : 2;
    setOpenFile({ path: entry.path, name: entry.name, loading: true, error: null, type: null });
    try {
      const fc = await hpcReadFile(entry.path, maxMb);
      if (!fc.ok) {
        setOpenFile((prev) => prev && { ...prev, loading: false, error: fc.error ?? "Failed to read file" });
        return;
      }
      if (fc.type === "binary") {
        const src = `data:${fc.mime};base64,${fc.data}`;
        setOpenFile((prev) => prev && { ...prev, loading: false, type: "image", src });
      } else if (fc.type === "text") {
        const text = fc.text ?? "";
        if (ext === "csv" || ext === "tsv") {
          const { headers, rows } = csvToTable(text);
          setOpenFile((prev) => prev && { ...prev, loading: false, type: "csv", headers, rows, text });
        } else if (ext === "html" || ext === "htm") {
          setOpenFile((prev) => prev && { ...prev, loading: false, type: "html", text });
        } else {
          setOpenFile((prev) => prev && { ...prev, loading: false, type: "text", text });
        }
      }
    } catch (e) {
      setOpenFile((prev) => prev && {
        ...prev,
        loading: false,
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }, []);

  const downloadEntry = useCallback(async (entry: RemoteEntry) => {
    try {
      if (entry.is_dir) {
        const dest = await save({
          defaultPath: entry.name + ".tar.gz",
          filters: [{ name: "Archive", extensions: ["tar.gz", "tgz"] }],
        });
        if (!dest) return;
        const result = await hpcDownloadDir(entry.path, dest);
        if (!result.ok) pushToast(result.error ?? "Download failed");
        else pushToast(`Saved to ${result.local_path}`);
      } else {
        const dest = await save({ defaultPath: entry.name });
        if (!dest) return;
        const result = await hpcDownloadFile(entry.path, dest);
        if (!result.ok) pushToast(result.error ?? "Download failed");
        else pushToast(`Saved to ${result.local_path}`);
      }
    } catch (e) {
      pushToast(e instanceof Error ? e.message : String(e));
    }
  }, [pushToast]);

  const downloadCurrentFile = useCallback(async () => {
    if (!openFile) return;
    const fakeEntry: RemoteEntry = { path: openFile.path, name: openFile.name, size: 0, is_dir: false, mtime: 0 };
    await downloadEntry(fakeEntry);
  }, [openFile, downloadEntry]);

  const handleContextMenu = useCallback((e: React.MouseEvent, entry: RemoteEntry) => {
    e.preventDefault();
    setCtxMenu({ x: e.clientX, y: e.clientY, entry });
  }, []);

  const commitPathInput = useCallback(() => {
    const p = pathInput.trim();
    if (p && p !== path) void navigate(p);
    else setEditingPath(false);
  }, [pathInput, path, navigate]);

  const segments = path.split("/").filter(Boolean);

  return (
    <aside className="flex w-[480px] shrink-0 flex-col border-l border-[var(--app-border)] bg-[var(--app-panel)]">
      {/* Path bar */}
      <div className="flex shrink-0 items-center gap-1 border-b border-[var(--app-border)] bg-[var(--app-bg)] px-2 py-1.5 text-xs">
        <button
          type="button"
          onClick={goUp}
          disabled={!path || path === hpcHome}
          className="shrink-0 rounded px-1 py-0.5 text-[var(--app-muted)] hover:text-[var(--app-text)] disabled:opacity-30 transition"
          title="Up one level"
        >
          ↑
        </button>

        {editingPath ? (
          <input
            ref={pathInputRef}
            className="min-w-0 flex-1 rounded border border-[var(--app-border)] bg-[var(--app-panel)] px-1.5 py-0.5 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-[var(--app-accent)]"
            value={pathInput}
            onChange={(e) => setPathInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitPathInput();
              if (e.key === "Escape") { setPathInput(path); setEditingPath(false); }
            }}
            onBlur={commitPathInput}
            autoFocus
          />
        ) : (
          <div
            className="flex min-w-0 flex-1 cursor-text flex-wrap items-center gap-0.5 overflow-hidden rounded px-1 py-0.5 hover:bg-[var(--app-panel)] transition"
            title="Click to edit path"
            onClick={() => { setPathInput(path); setEditingPath(true); }}
          >
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); void navigate(hpcHome); }}
              className="shrink-0 text-[var(--app-accent)] hover:underline"
            >
              ~
            </button>
            {segments.map((seg, i) => {
              const segPath = "/" + segments.slice(0, i + 1).join("/");
              return (
                <span key={i} className="flex items-center gap-0.5">
                  <span className="text-[var(--app-muted)]">/</span>
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); void navigate(segPath); }}
                    className="max-w-[100px] truncate text-[var(--app-accent)] hover:underline"
                  >
                    {seg}
                  </button>
                </span>
              );
            })}
          </div>
        )}

        <button
          type="button"
          onClick={() => void navigate(path)}
          className="shrink-0 text-[var(--app-muted)] hover:text-[var(--app-text)] transition"
          title="Refresh"
        >
          ↻
        </button>
      </div>

      {/* Body: either file list or content view */}
      {openFile ? (
        <ContentView file={openFile} onBack={() => setOpenFile(null)} onDownload={() => void downloadCurrentFile()} />
      ) : (
        <DirList
          entries={entries}
          loading={dirLoading}
          error={dirError}
          onNavigate={(p) => void navigate(p)}
          onOpenFile={(e) => void openFileForViewing(e)}
          onContextMenu={handleContextMenu}
        />
      )}

      {/* Statusline */}
      <div className="shrink-0 border-t border-[var(--app-border)] px-3 py-1 text-xs text-[var(--app-muted)]">
        {openFile ? openFile.name : path}
      </div>

      {/* Context menu */}
      {ctxMenu && (
        <ContextMenu
          menu={ctxMenu}
          onClose={() => setCtxMenu(null)}
          onDownload={(entry) => void downloadEntry(entry)}
        />
      )}
    </aside>
  );
}

