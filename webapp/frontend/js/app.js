/* TERRA static UI — orchestration. Talks to webapp/api. */
(function () {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const esc = (s) => String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function el(tag, props, ...kids) {
    const n = document.createElement(tag);
    if (props) for (const [k, v] of Object.entries(props)) {
      if (k === "class") n.className = v;
      else if (k === "html") n.innerHTML = v;
      else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
      else if (v != null) n.setAttribute(k, v);
    }
    for (const kid of kids.flat()) {
      if (kid == null) continue;
      n.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
    }
    return n;
  }

  const state = {
    clFile: null, guFile: null,
    clInspect: null, guInspect: null,
    clusterJobId: null, clusterFamily: null,
    assemblies: ["hg38", "hg19", "mm10", "mm39", "hs1", "chm13"],
  };

  // ── Theme ──────────────────────────────────────────────────────────────
  (function initTheme() {
    const saved = localStorage.getItem("terra_theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
    $("#btnTheme").addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme");
      const isDark = cur ? cur === "dark"
        : matchMedia("(prefers-color-scheme: dark)").matches;
      const next = isDark ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("terra_theme", next);
    });
  })();

  // ── Tabs ───────────────────────────────────────────────────────────────
  $$(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  function switchTab(name) {
    $$(".tab").forEach((t) => t.classList.toggle("is-active", t.dataset.tab === name));
    $$(".panel").forEach((p) => p.classList.toggle("is-active", p.id === "panel-" + name));
  }

  // ── Settings + health ──────────────────────────────────────────────────
  $("#btnSettings").addEventListener("click", () => $("#settings").classList.toggle("hidden"));
  $("#apiUrl").value = window.TERRA.getApiBase();
  $("#btnSaveApi").addEventListener("click", async () => {
    window.TERRA.setApiBase($("#apiUrl").value);
    $("#settings").classList.add("hidden");
    await checkHealth();
  });

  async function checkHealth() {
    const status = $("#apiStatus");
    const text = $(".api-status-text", status);
    status.className = "api-status";
    text.textContent = "checking…";
    if (!window.TERRA.hasApiBase()) {
      status.classList.add("down"); text.textContent = "no API set";
      $("#settings").classList.remove("hidden");
      return;
    }
    try {
      const h = await window.API.health();
      status.classList.add("ok"); text.textContent = "online";
      if (Array.isArray(h.assemblies) && h.assemblies.length) {
        state.assemblies = h.assemblies; populateAssemblies();
      }
    } catch (e) {
      status.classList.add("down"); text.textContent = "offline";
    }
  }

  function populateAssemblies() {
    const sel = $("#clAssembly");
    sel.innerHTML = "";
    state.assemblies.forEach((a) => sel.appendChild(el("option", { value: a }, a)));
  }

  // ── Segmented controls ──────────────────────────────────────────────────
  function wireSegmented(rootId) {
    const root = $("#" + rootId);
    const panelHost = root.closest(".card");
    root.addEventListener("click", (e) => {
      const seg = e.target.closest(".seg"); if (!seg) return;
      $$(".seg", root).forEach((s) => s.classList.toggle("is-active", s === seg));
      $$(".mode-panel", panelHost).forEach((p) =>
        p.classList.toggle("is-active", p.dataset.mode === seg.dataset.mode));
    });
  }
  wireSegmented("clInputMode");
  wireSegmented("guInputMode");
  function activeMode(rootId) { return $("#" + rootId + " .seg.is-active").dataset.mode; }

  // ── File inspect + column mapping ───────────────────────────────────────
  const CL_ROLES = [
    { key: "sequence", label: "Sequence *", required: true },
    { key: "name", label: "Name" },
    { key: "start", label: "Start" },
    { key: "stop", label: "Stop" },
    { key: "strand", label: "Strand" },
    { key: "chr", label: "Chromosome" },
  ];
  const GU_ROLES = [
    { key: "sequence", label: "Sequence *", required: true },
    { key: "cluster", label: "Cluster" },
  ];

  function roleSelect(prefix, role, cols, guess) {
    const sel = el("select", { id: `${prefix}_${role.key}` });
    if (!role.required) sel.appendChild(el("option", { value: "" }, "— none —"));
    cols.forEach((c) => sel.appendChild(el("option", { value: c }, c)));
    if (guess && cols.includes(guess)) sel.value = guess;
    else if (role.required && cols.length) sel.value = cols[0];
    return el("label", {}, role.label, sel);
  }

  function renderMapping(container, inspect, roles, prefix) {
    container.innerHTML = "";
    const cols = inspect.columns;
    const g = inspect.guesses || {};
    roles.forEach((r) => container.appendChild(roleSelect(prefix, r, cols, g[r.key])));

    // Expression multi-select (numeric columns).
    const exprCands = (inspect.numeric_columns || []).filter(
      (c) => !roles.some((r) => g[r.key] === c && r.key !== "cluster"));
    const chips = el("div", { class: "expr-list", id: `${prefix}_expr` });
    const preset = new Set(g.expression || []);
    exprCands.forEach((c) => {
      const cb = el("input", { type: "checkbox", value: c });
      if (preset.has(c)) cb.checked = true;
      const chip = el("label", { class: "chip" + (cb.checked ? " on" : "") }, cb, c);
      cb.addEventListener("change", () => chip.classList.toggle("on", cb.checked));
      chips.appendChild(chip);
    });
    const exprBlock = el("div", { class: "expr-cols" },
      el("label", {}, exprCands.length ? "Expression columns" : "Expression columns (none numeric found)"),
      chips);
    container.appendChild(exprBlock);
  }

  function collectMapping(prefix, roles) {
    const m = {};
    roles.forEach((r) => {
      const s = $(`#${prefix}_${r.key}`);
      m[r.key] = s && s.value ? s.value : null;
    });
    const roleCols = new Set(roles.map((r) => m[r.key]).filter(Boolean));
    m.expression = $$(`#${prefix}_expr input:checked`).map((c) => c.value)
      .filter((v) => !roleCols.has(v));
    return m;
  }

  function renderPreview(table, inspect) {
    table.innerHTML = "";
    const cols = inspect.columns;
    table.appendChild(el("thead", {}, el("tr", {}, cols.map((c) => el("th", {}, c)))));
    const tb = el("tbody", {});
    (inspect.preview || []).forEach((row) => {
      tb.appendChild(el("tr", {}, cols.map((c) => el("td", {}, String(row[c] ?? "")))));
    });
    table.appendChild(tb);
  }

  async function handleFile(inputEl, opts) {
    const file = inputEl.files[0];
    if (!file) return;
    opts.nameEl.textContent = file.name;
    opts.store(file);
    try {
      const insp = await window.API.inspect(file);
      opts.setInspect(insp);
      $(opts.rowCountSel).textContent = `${insp.n_rows.toLocaleString()} rows, ${insp.columns.length} columns`;
      renderPreview($(opts.previewSel), insp);
      renderMapping($(opts.mappingSel), insp, opts.roles, opts.prefix);
      $(opts.inspectSel).classList.remove("hidden");
    } catch (e) {
      alert("Could not read file: " + e.message);
    }
  }

  $("#clFile").addEventListener("change", (e) => handleFile(e.target, {
    nameEl: $("#clFileName"), store: (f) => (state.clFile = f),
    setInspect: (i) => (state.clInspect = i),
    rowCountSel: "#clRowCount", previewSel: "#clPreview",
    mappingSel: "#clMapping", inspectSel: "#clInspect",
    roles: CL_ROLES, prefix: "clm",
  }));
  $("#guFile").addEventListener("change", (e) => handleFile(e.target, {
    nameEl: $("#guFileName"), store: (f) => (state.guFile = f),
    setInspect: (i) => (state.guInspect = i),
    rowCountSel: "#guRowCount", previewSel: "#guPreview",
    mappingSel: "#guMapping", inspectSel: "#guInspect",
    roles: GU_ROLES, prefix: "gum",
  }));

  // ── Job runner + live log ───────────────────────────────────────────────
  function jobBox(host) {
    host.innerHTML = "";
    host.classList.remove("hidden", "error");
    const stage = el("span", { class: "job-stage" }, "starting…");
    const log = el("pre", { class: "log" });
    host.appendChild(el("div", { class: "job-head" }, el("span", { class: "spinner" }), stage));
    host.appendChild(log);
    return { stage, log, host };
  }

  async function pollJob(jobId, ui) {
    while (true) {
      const j = await window.API.job(jobId);
      ui.stage.textContent = j.stage || j.status;
      ui.log.textContent = (j.log_tail || []).join("\n");
      ui.log.scrollTop = ui.log.scrollHeight;
      if (j.status === "done") { $(".spinner", ui.host)?.remove(); return j; }
      if (j.status === "error") {
        ui.host.classList.add("error");
        $(".spinner", ui.host)?.remove();
        throw new Error(j.error || "job failed");
      }
      await sleep(1200);
    }
  }

  function fail(ui, msg) {
    ui.host.classList.remove("hidden");
    ui.host.classList.add("error");
    ui.host.innerHTML = `<div class="err-banner">Error</div><pre class="log">${esc(msg)}</pre>`;
  }

  // ── Downloads / figures ─────────────────────────────────────────────────
  function dl(jobId, name, label) {
    return el("a", { class: "btn btn-sm", href: window.API.fileUrl(jobId, name), download: name }, label);
  }
  function figure(jobId, name, title, caption) {
    const img = el("img", { src: window.API.fileUrl(jobId, name), alt: title, loading: "lazy" });
    img.addEventListener("click", () => lightbox(img.src));
    return el("div", { class: "figure" }, el("h3", {}, title),
      caption ? el("p", { class: "cap" }, caption) : null, img);
  }
  // Figure with an embedding switcher (UMAP / t-SNE / PCA).
  const EMB_LABELS = { umap: "UMAP", tsne: "t-SNE", pca: "PCA" };
  function embeddingFigure(jobId, embMap, title, caption) {
    const order = ["umap", "tsne", "pca"].filter((k) => embMap[k]);
    const img = el("img", { src: window.API.fileUrl(jobId, embMap[order[0]]), alt: title, loading: "lazy" });
    img.addEventListener("click", () => lightbox(img.src));
    const btns = el("div", { class: "btn-row" });
    order.forEach((k, i) => {
      const b = el("button", { class: "btn btn-sm" + (i === 0 ? " btn-primary" : "") }, EMB_LABELS[k]);
      b.addEventListener("click", () => {
        img.src = window.API.fileUrl(jobId, embMap[k]);
        $$(".btn", btns).forEach((x) => x.classList.remove("btn-primary"));
        b.classList.add("btn-primary");
      });
      btns.appendChild(b);
    });
    return el("div", { class: "figure" }, el("h3", {}, title),
      caption ? el("p", { class: "cap" }, caption) : null,
      order.length > 1 ? btns : null, img);
  }
  function lightbox(src) {
    const box = el("div", { class: "lightbox" }, el("img", { src }));
    box.addEventListener("click", () => box.remove());
    document.body.appendChild(box);
  }
  function kpi(val, label, sub) {
    return el("div", { class: "kpi" },
      el("div", { class: "k-val" }, String(val)),
      el("div", { class: "k-label" }, label),
      sub ? el("div", { class: "k-sub" }, sub) : null);
  }
  // Sortable table. cols: [{key,label,cls,get,fmt}]. Click any header to sort by
  // that column's value (numeric when the column is all-numeric, else text);
  // click again to reverse. `fmt` is a fallback {key: fn} formatter map.
  function dataTable(cols, rows, fmt) {
    fmt = fmt || {};
    const data = rows.slice();
    let sortKey = null, sortDir = 1;
    const rawOf = (c, r) => (c.get ? c.get(r) : r[c.key]);
    const fmtOf = (c, v) => {
      const f = c.fmt || fmt[c.key];
      return f ? f(v) : (v == null ? "" : String(v));
    };
    const table = el("table", { class: "data" });
    const wrap = el("div", { class: "table-wrap" }, table);

    function render() {
      table.innerHTML = "";
      const htr = el("tr", {});
      cols.forEach((c) => {
        const arrow = sortKey === c.key ? (sortDir > 0 ? " ▲" : " ▼") : "";
        const th = el("th", { class: "sortable", title: "Sort by " + c.label }, c.label + arrow);
        th.addEventListener("click", () => {
          if (sortKey === c.key) sortDir = -sortDir;
          else { sortKey = c.key; sortDir = 1; }
          const numeric = data.every((r) => {
            const x = rawOf(c, r);
            return x == null || x === "" || !isNaN(Number(x));
          });
          data.sort((a, b) => {
            const x = rawOf(c, a), y = rawOf(c, b);
            if (numeric) return ((Number(x) || 0) - (Number(y) || 0)) * sortDir;
            return String(x).localeCompare(String(y)) * sortDir;
          });
          render();
        });
        htr.appendChild(th);
      });
      table.appendChild(el("thead", {}, htr));
      const tb = el("tbody", {});
      data.forEach((r) => {
        tb.appendChild(el("tr", {}, cols.map((c) =>
          el("td", c.cls ? { class: c.cls } : {}, fmtOf(c, rawOf(c, r))))));
      });
      table.appendChild(tb);
    }
    render();
    return wrap;
  }

  // ── Run: Clustering ─────────────────────────────────────────────────────
  $("#btnCluster").addEventListener("click", runCluster);
  async function runCluster() {
    const ui = jobBox($("#clJob"));
    $("#clResults").classList.add("hidden");
    try {
      const fd = new FormData();
      const mode = activeMode("clInputMode");
      fd.append("mode", mode);
      let familyLabel = "UPLOAD";
      if (mode === "upload") {
        if (!state.clFile) throw new Error("Choose a CSV/XLSX file first.");
        const map = collectMapping("clm", CL_ROLES);
        if (!map.sequence) throw new Error("Pick the Sequence column.");
        fd.append("file", state.clFile);
        fd.append("mapping", JSON.stringify(map));
      } else {
        familyLabel = $("#clFamily").value.trim();
        if (!familyLabel) throw new Error("Enter a family name.");
        fd.append("family", familyLabel);
        fd.append("assembly", $("#clAssembly").value);
        fd.append("source", $("#clSource").value);
        fd.append("max_loci", $("#clMaxLoci").value || "0");
      }
      fd.append("kmer", $("#clKmer").value);
      fd.append("min_cluster_size", $("#clMinCluster").value);
      fd.append("n_neighbors", $("#clNeighbors").value);
      fd.append("min_dist", $("#clMinDist").value);
      fd.append("min_samples", $("#clMinSamples").value);
      fd.append("compute_tsne", $("#clTsne").checked ? "true" : "false");

      const { job_id } = await window.API.cluster(fd);
      const j = await pollJob(job_id, ui);
      state.clusterJobId = job_id;
      state.clusterFamily = familyLabel;
      renderClusterResults(job_id, j.result);
      refreshGuidesLink();
    } catch (e) {
      fail(ui, e.message);
    }
  }

  function renderClusterResults(jobId, r) {
    const host = $("#clResults");
    host.innerHTML = "";
    host.classList.remove("hidden");

    const kpis = el("div", { class: "kpis" },
      kpi(r.n_loci.toLocaleString(), "Loci"),
      kpi(r.n_clusters, "Clusters (excl. noise)"),
      kpi(r.has_expression ? "Yes" : "No", "Expression data"));
    host.appendChild(kpis);

    if (r.sequences_cached) {
      host.appendChild(el("p", { class: "hint" },
        "↺ Sequences reused from cache — no re-download. Re-clustering with different parameters stays instant; change the family / assembly / source / max-loci to fetch fresh."));
    }

    if (r.files && r.files.viz_html) {
      const frame = el("iframe", { class: "viz-frame", src: window.API.fileUrl(jobId, r.files.viz_html), loading: "lazy" });
      const openLink = el("a", { class: "btn btn-sm", target: "_blank", rel: "noopener",
        href: window.API.fileUrl(jobId, r.files.viz_html) }, "Open in new tab ↗");
      const controls = el("div", { class: "btn-row" }, openLink);
      if (r.files.viz_html_expr) {
        const toggle = el("button", { class: "btn btn-sm" }, "Size points by expression");
        let onExpr = false;
        toggle.addEventListener("click", () => {
          onExpr = !onExpr;
          const f = onExpr ? r.files.viz_html_expr : r.files.viz_html;
          frame.src = window.API.fileUrl(jobId, f);
          openLink.href = frame.src;
          toggle.textContent = onExpr ? "Uniform point size" : "Size points by expression";
        });
        controls.appendChild(toggle);
      }
      host.appendChild(el("div", { class: "figure" },
        el("h3", {}, "Sequence-similarity embedding"),
        el("p", { class: "cap" }, "PCA, UMAP and t-SNE projections of k-mer profiles; points colored by HDBSCAN cluster. Hover for locus details."),
        controls, frame));
    }

    if (Array.isArray(r.cluster_summary) && r.cluster_summary.length) {
      host.appendChild(el("h3", { class: "section-title" }, "Cluster summary"));
      host.appendChild(dataTable([
        { key: "cluster", label: "Cluster" },
        { key: "n_total", label: "Loci" },
        { key: "n_plus", label: "+ strand" },
        { key: "n_minus", label: "− strand" },
        { key: "pct_plus", label: "% +" },
        { key: "pct_minus", label: "% −" },
      ], r.cluster_summary));
    }

    const row = el("div", { class: "btn-row" });
    if (r.files.clustered_csv) row.appendChild(dl(jobId, r.files.clustered_csv, "⬇ Clustered CSV"));
    if (r.files.coords_csv) row.appendChild(dl(jobId, r.files.coords_csv, "⬇ Coordinates"));
    if (r.files.summary_csv) row.appendChild(dl(jobId, r.files.summary_csv, "⬇ Summary"));
    const go = el("button", { class: "btn btn-primary btn-sm" }, "Design guides from this result →");
    go.addEventListener("click", () => {
      switchTab("guides");
      $("#guInputMode .seg[data-mode='job']").click();
      $("#guFamily").value = state.clusterFamily || "FAMILY";
      refreshGuidesLink();
    });
    row.appendChild(go);
    host.appendChild(row);
  }

  function refreshGuidesLink() {
    const box = $("#guLinked");
    if (state.clusterJobId) {
      box.innerHTML = "";
      box.appendChild(el("p", {},
        "Using clustering result ",
        el("code", {}, state.clusterFamily || "result"),
        " (job ", el("code", {}, state.clusterJobId.slice(0, 8)), "). ",
        "Coordinates and any expression columns carry over automatically."));
    } else {
      box.innerHTML = '<p class="muted">No clustering result yet. Run the <strong>Cluster</strong> tab first, or switch to upload.</p>';
    }
  }

  // ── Run: Guides ─────────────────────────────────────────────────────────
  $("#btnGuides").addEventListener("click", runGuides);
  async function runGuides() {
    const ui = jobBox($("#guJob"));
    $("#guResults").classList.add("hidden");
    try {
      const fd = new FormData();
      const mode = activeMode("guInputMode");
      fd.append("mode", mode);
      if (mode === "job") {
        if (!state.clusterJobId) throw new Error("No clustering result — run the Cluster tab or upload a CSV.");
        fd.append("source_job_id", state.clusterJobId);
      } else {
        if (!state.guFile) throw new Error("Choose a clustered CSV/XLSX first.");
        const map = collectMapping("gum", GU_ROLES);
        if (!map.sequence) throw new Error("Pick the Sequence column.");
        fd.append("file", state.guFile);
        fd.append("mapping", JSON.stringify(map));
      }
      fd.append("family", $("#guFamily").value.trim() || "FAMILY");
      fd.append("cas", $("#guCas").value);
      fd.append("grna_len", $("#guLen").value);
      fd.append("n_greedy", $("#guGreedy").value);
      fd.append("n_workers", $("#guWorkers").value);
      fd.append("global_mm_index", $("#guMm").value);

      const { job_id } = await window.API.guides(fd);
      const j = await pollJob(job_id, ui);
      renderGuidesResults(job_id, j.result);
    } catch (e) {
      fail(ui, e.message);
    }
  }

  const pct = (v) => (v == null ? "" : Number(v).toFixed(1) + "%");
  const pct0 = (v) => (v == null || v === "" ? "" : Number(v).toFixed(0) + "%");
  const f2 = (v) => (v == null ? "" : Number(v).toFixed(2));
  const f3 = (v) => (v == null ? "" : Number(v).toFixed(3));

  // One sortable column per cluster, reading its % from each row's `clusters` map.
  function clusterCols(ids) {
    return (ids || []).map((cid) => ({
      key: "C" + cid, label: "C" + cid,
      get: (row) => (row.clusters ? row.clusters[String(cid)] : null),
      fmt: pct0,
    }));
  }

  function renderGuidesResults(jobId, r) {
    const host = $("#guResults");
    host.innerHTML = "";
    host.classList.remove("hidden");
    const R = r.results;
    const hasExpr = R.has_expression;
    const cCols = clusterCols(r.cluster_ids);

    const kpis = [
      kpi(R.n_candidates.toLocaleString(), "gRNA candidates"),
      kpi(R.n_pam_sites.toLocaleString(), "PAM sites"),
      kpi(pct(hasExpr ? R.top_expr_cov_pct : R.top_seq_cov_pct),
        hasExpr ? "Best expr. coverage" : "Best coverage", R.best_expr_grna),
      kpi(pct(R.greedy5_loci_pct), `${r.greedy.length}-guide panel · loci`),
    ];
    if (hasExpr) kpis.push(kpi(pct(R.greedy5_expr_pct), `${r.greedy.length}-guide panel · expression`));
    host.appendChild(el("div", { class: "kpis" }, kpis));

    if (!hasExpr) {
      host.appendChild(el("p", { class: "hint" },
        "No expression columns were detected, so every locus is weighted equally, the panel is optimized for locus coverage, and expression columns are omitted below."));
    }

    // Optimal panel table
    if (r.greedy && r.greedy.length) {
      host.appendChild(el("h3", { class: "section-title" }, `Optimal ${r.greedy.length}-guide panel (greedy set-cover)`));
      host.appendChild(el("p", { class: "cap" },
        "Guides are added one at a time, each chosen to cover the most still-uncovered " +
        (hasExpr ? "expression" : "loci") + ". Cumulative reach grows down the table; C0…Cn are the % of each cluster's loci the guide hits. Click any header to sort."));
      const gCols = [
        { key: "rank", label: "#" },
        { key: "grna", label: "Guide 5′→3′ (fwd)", cls: "seq" },
        { key: "grna_rc", label: "Rev-comp 5′→3′", cls: "seq" },
        { key: "marginal_loci", label: "+ loci" },
        { key: "cumulative_loci", label: "Σ loci" },
        { key: "pct_loci", label: "% loci", fmt: pct },
      ];
      if (hasExpr) gCols.push({ key: "pct_expr", label: "% expr", fmt: pct });
      host.appendChild(dataTable(gCols.concat(cCols), r.greedy));
    }

    // Figures — the "why".
    host.appendChild(el("h3", { class: "section-title" }, "Why these guides"));
    if (r.figures.coverage) host.appendChild(figure(jobId, r.figures.coverage,
      "Coverage analysis",
      hasExpr
        ? ("(a) how many candidates reach each % of loci; (b) sequence vs expression coverage with correlation r = " +
           f3(R.seq_expr_corr) + "; (c) mismatch tolerance of the best guide; (d) per-cluster coverage of the top 5 guides.")
        : "(a) mismatch tolerance of the best guide; (b) per-cluster coverage of the top 5 guides. Sequence-vs-expression panels are omitted — there is no expression data."));
    if (r.figures.candidates) host.appendChild(figure(jobId, r.figures.candidates,
      "Top candidates",
      hasExpr
        ? "Left: best guides by raw sequence coverage. Right: best by expression coverage. OTS = on-target quality score (0–1)."
        : "Best guides ranked by sequence coverage. OTS = on-target quality score (0–1)."));
    if (r.figures.positional) host.appendChild(figure(jobId, r.figures.positional,
      "PAM-site positional density",
      "Where usable PAM sites fall along the element (0 = 5′, 1 = 3′), separately for + and − strand guides."));
    if (r.figures.embedding && Object.keys(r.figures.embedding).length)
      host.appendChild(embeddingFigure(jobId, r.figures.embedding,
        "Reagent coverage on the embedding",
        "The optimal panel projected onto the embedding: each covered locus colored by the guide that reaches it; grey = not covered. Exact protospacer match. Switch between UMAP / t-SNE / PCA above."));

    // Candidate table
    if (r.top_candidates && r.top_candidates.length) {
      host.appendChild(el("h3", { class: "section-title" }, "Top 25 candidates"));
      host.appendChild(el("p", { class: "cap" }, "C0…Cn = % of each cluster's loci this guide hits. Click any header to sort."));
      const tCols = [
        { key: "grna", label: "Guide 5′→3′ (fwd)", cls: "seq" },
        { key: "grna_rc", label: "Rev-comp 5′→3′", cls: "seq" },
        { key: "gc", label: "GC", fmt: f2 },
        { key: "on_target_score", label: "OTS", fmt: f2 },
        { key: "exact_cov", label: "Loci" },
      ];
      if (hasExpr) tCols.push({ key: "exact_expr_cov", label: "Expr", fmt: f2 });
      tCols.push({ key: "pct_seq_cov", label: "% loci", fmt: pct });
      if (hasExpr) tCols.push({ key: "pct_expr_cov", label: "% expr", fmt: pct });
      host.appendChild(dataTable(tCols.concat(cCols), r.top_candidates));
    }

    const row = el("div", { class: "btn-row" });
    if (r.files.candidates_csv) row.appendChild(dl(jobId, r.files.candidates_csv, "⬇ Candidates CSV"));
    if (r.files.greedy_csv) row.appendChild(dl(jobId, r.files.greedy_csv, "⬇ Panel CSV"));
    if (r.files.covered_csv) row.appendChild(dl(jobId, r.files.covered_csv, "⬇ Covered loci CSV"));
    host.appendChild(row);
  }

  // ── Boot ────────────────────────────────────────────────────────────────
  populateAssemblies();
  refreshGuidesLink();
  checkHealth();
})();
