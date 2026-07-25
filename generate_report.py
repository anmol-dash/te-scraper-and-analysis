#!/usr/bin/env python3
"""
generate_report.py — Comprehensive HTML analysis report for a GAMECA TE family folder.
Usage: python generate_report.py <family_folder> [<family_folder2> ...]
"""

import argparse, base64, csv, json, os, re, subprocess, sys, tempfile
from pathlib import Path
from datetime import datetime

# ── Helpers ──────────────────────────────────────────────────────────────────

def b64_img(path):
    p = Path(path)
    if not p.exists():
        return None
    ext = p.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
    try:
        data = base64.b64encode(p.read_bytes()).decode()
        return f"data:{mime};base64,{data}"
    except Exception:
        return None

def pdf_to_b64(pdf_path):
    """Convert first page of PDF → PNG via sips, return base64 data URI."""
    p = Path(pdf_path)
    if not p.exists():
        return None
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        res = subprocess.run(
            ["sips", "-s", "format", "png", "-Z", "2000", str(p), "--out", tmp_path],
            capture_output=True, timeout=60
        )
        if res.returncode == 0 and Path(tmp_path).exists():
            return b64_img(tmp_path)
    except Exception:
        pass
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass
    return None

def get_img(folder, stem):
    """Return base64 URI for a figure: try .png first, then .pdf via sips."""
    f = Path(folder)
    for ext in (".png", ".jpg", ".jpeg"):
        p = f / (stem + ext)
        if p.exists():
            return b64_img(p)
    pdf = f / (stem + ".pdf")
    if pdf.exists():
        return pdf_to_b64(pdf)
    return None

def _alignment_images_dir(d):
    """Return the directory holding CIAlign plots.

    Stage 7 writes them to 04_alignments/images/; older result trees used a
    top-level cialign_plots/.
    """
    d = Path(d)
    new = d / "04_alignments" / "images"
    if new.is_dir() and any(new.glob("*.png")):
        return new
    return d / "cialign_plots"


def _cluster_image_stems(img_dir):
    """Discover per-cluster CIAlign plot stems present in *img_dir*, in order."""
    img_dir = Path(img_dir)
    if not img_dir.is_dir():
        return []
    stems = set()
    for p in img_dir.glob("cluster_*_alignment*.png"):
        name = p.stem
        cut = name.find("_alignment")
        if cut > 0:
            stems.add(name[:cut] + "_alignment")

    def _key(s):
        tok = s.replace("_alignment", "").replace("cluster_", "")
        try:
            return (0, int(tok))
        except ValueError:
            return (1, tok)

    return sorted(stems, key=_key)


def read_txt(path):
    p = Path(path)
    return p.read_text(errors="replace").strip() if p.exists() else ""

def read_csv_rows(path, max_rows=None):
    p = Path(path)
    if not p.exists():
        return [], []
    with open(p, newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    headers = list(rows[0].keys()) if rows else []
    if max_rows:
        rows = rows[:max_rows]
    return headers, rows

def parse_report(text):
    """Extract key: value pairs from GAMECA report text files."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("=", "-", "─", "GAMECA", "Generated", "Family", "Assembly")):
            pass
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip().lstrip("─ ")
            v = v.strip()
            if k and v and len(k) < 60:
                out[k] = v
    return out

def parse_stats(text):
    """Parse overall_statistics.txt into a dict."""
    out = {"family": "", "n_copies": "", "mean_len": "", "median_len": "",
           "std_len": "", "range_len": "", "gc_mean": "", "gc_std": "",
           "gc_range": "", "expr_cols": "", "expr_mean": "", "expr_max": ""}
    for line in text.splitlines():
        m = re.search(r"STATISTICS\s*[—-]\s*(.+)", line)
        if m:
            out["family"] = m.group(1).strip()
        if "Dataset size:" in line:
            m2 = re.search(r"([\d,]+)\s+sequences", line)
            if m2: out["n_copies"] = m2.group(1).replace(",", "")
        if re.match(r"\s*Mean:\s+[\d.]+\s+Median:", line):
            parts = re.findall(r"([\d.]+)", line)
            if len(parts) >= 2:
                out["mean_len"] = parts[0]; out["median_len"] = parts[1]
        elif re.match(r"\s*Std:\s+[\d.]+\s+Range:", line):
            parts = re.findall(r"([\d.]+)", line)
            if parts: out["std_len"] = parts[0]
            m2 = re.search(r"Range:\s*\[(.+?)\]", line)
            if m2: out["range_len"] = m2.group(1)
        if re.match(r"\s*Mean:\s+[\d.]+\s+\([\d.]+%\)", line):
            m2 = re.search(r"Mean:\s+([\d.]+)\s+\(([\d.]+)%\)", line)
            if m2: out["gc_mean"] = m2.group(2)
        if "Total expression" in line:
            m2 = re.search(r"mean:\s*([\d.e+]+)", line)
            if m2: out["expr_mean"] = m2.group(1)
            m3 = re.search(r"max:\s*([\d.e+]+)", line)
            if m3: out["expr_max"] = m3.group(1)
        if "Columns:" in line:
            m2 = re.search(r"Columns:\s*(\d+)", line)
            if m2: out["expr_cols"] = m2.group(1)
    return out

# ── HTML primitives ───────────────────────────────────────────────────────────

def html_table(headers, rows, max_rows=None, cols=None):
    if not rows:
        return "<p class='muted'>No data available.</p>"
    if not headers:
        headers = list(rows[0].keys())
    if cols:
        headers = [h for h in headers if h in cols]
    if max_rows:
        rows = rows[:max_rows]
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for i, row in enumerate(rows):
        cells = ""
        for h in headers:
            v = str(row.get(h, ""))
            if len(v) > 70: v = v[:67] + "…"
            cells += f"<td>{v}</td>"
        body += f"<tr class='{'even' if i%2==0 else 'odd'}'>{cells}</tr>"
    return f"<div class='table-wrap'><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"

def fig(uri, caption=""):
    if not uri:
        return ""
    cap = f"<p class='fig-cap'>{caption}</p>" if caption else ""
    return f"<div class='fig-box'><img src='{uri}' loading='lazy'>{cap}</div>"

def stat_card(label, value, unit=""):
    return f"<div class='card'><div class='card-val'>{value}<span class='card-unit'>{unit}</span></div><div class='card-lbl'>{label}</div></div>"

def note(text):
    return f"<p class='note'>{text}</p>"

def section_html(sid, icon, title, body):
    return f"""
<section id="{sid}">
  <h2><span class='sec-icon'>{icon}</span>{title}</h2>
  {body}
</section>"""

# ── CSS + JS template ─────────────────────────────────────────────────────────

CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #f1f5f9; --sidebar: #0f172a; --accent: #3b82f6;
  --text: #1e293b; --muted: #64748b; --border: #e2e8f0;
  --card-bg: #ffffff; --header-bg: #1e40af;
}
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: var(--bg); color: var(--text); display: flex; min-height: 100vh; }
/* Sidebar */
#sidebar {
  width: 240px; min-width: 240px; background: var(--sidebar); color: #cbd5e1;
  padding: 24px 0; position: sticky; top: 0; height: 100vh;
  overflow-y: auto; flex-shrink: 0;
}
#sidebar h3 { font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase;
              color: #64748b; padding: 0 20px 8px; margin-top: 16px; }
#sidebar a { display: block; padding: 5px 20px; font-size: 0.82rem; color: #94a3b8;
             text-decoration: none; border-left: 3px solid transparent;
             transition: all 0.15s; }
#sidebar a:hover, #sidebar a.active { color: #f1f5f9; border-left-color: var(--accent);
             background: rgba(59,130,246,0.1); }
#sidebar .brand { padding: 0 20px 24px; border-bottom: 1px solid #1e293b; }
#sidebar .brand h2 { color: #f8fafc; font-size: 1.1rem; }
#sidebar .brand p { color: #64748b; font-size: 0.75rem; margin-top: 4px; }
/* Main */
#main { flex: 1; padding: 32px 40px; max-width: 1100px; }
.report-header { background: linear-gradient(135deg, #1e40af, #0f172a);
  color: white; border-radius: 12px; padding: 32px; margin-bottom: 32px; }
.report-header h1 { font-size: 1.8rem; font-weight: 700; }
.report-header .meta { font-size: 0.85rem; color: #93c5fd; margin-top: 8px; }
/* Sections */
section { margin-bottom: 40px; }
section h2 { font-size: 1.2rem; font-weight: 600; color: var(--header-bg);
  border-left: 4px solid var(--accent); padding: 6px 12px;
  margin-bottom: 16px; background: #eff6ff; border-radius: 0 6px 6px 0; }
.sec-icon { margin-right: 8px; }
/* Stat cards */
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }
.card { background: white; border: 1px solid var(--border); border-radius: 8px;
  padding: 14px 18px; min-width: 130px; }
.card-val { font-size: 1.5rem; font-weight: 700; color: var(--header-bg); }
.card-unit { font-size: 0.8rem; font-weight: 400; color: var(--muted); margin-left: 3px; }
.card-lbl { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }
/* Tables */
.table-wrap { overflow-x: auto; margin: 12px 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.82rem; background: white;
  border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.07); }
thead tr { background: #1e40af; color: white; }
th { padding: 9px 12px; text-align: left; font-weight: 600; white-space: nowrap; }
td { padding: 7px 12px; border-bottom: 1px solid var(--border); }
tr.even td { background: #f8fafc; }
/* Figures */
.fig-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 16px 0; }
.fig-box { background: white; border: 1px solid var(--border); border-radius: 8px;
  padding: 12px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.fig-box img { max-width: 100%; border-radius: 4px; }
.fig-cap { font-size: 0.78rem; color: var(--muted); margin-top: 8px; font-style: italic; }
/* Text */
.note { background: #fefce8; border-left: 3px solid #eab308;
  padding: 10px 14px; border-radius: 0 6px 6px 0; font-size: 0.85rem;
  color: #713f12; margin: 12px 0; }
.muted { color: var(--muted); font-size: 0.85rem; font-style: italic; }
.skipped { background: #f1f5f9; border-left: 3px solid #94a3b8;
  padding: 10px 14px; border-radius: 0 6px 6px 0; font-size: 0.85rem; color: var(--muted); margin: 12px 0; }
pre { background: #f8fafc; border: 1px solid var(--border); border-radius: 6px;
  padding: 14px; font-size: 0.78rem; overflow-x: auto; white-space: pre-wrap; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 900px) { .two-col { grid-template-columns: 1fr; }
  #sidebar { display: none; } #main { padding: 16px; } }
"""

JS = """
document.addEventListener('DOMContentLoaded', () => {
  const links = document.querySelectorAll('#sidebar a[href^="#"]');
  const observer = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        links.forEach(l => l.classList.toggle('active', l.getAttribute('href') === '#' + e.target.id));
      }
    });
  }, { rootMargin: '-30% 0px -60% 0px' });
  document.querySelectorAll('section[id]').forEach(s => observer.observe(s));
});
"""

# ── Report sections ───────────────────────────────────────────────────────────

def sec_overview(d, stats, cl_rows):
    cards = "".join([
        stat_card("Copies", stats["n_copies"]),
        stat_card("Mean Length", stats["mean_len"], "bp"),
        stat_card("Median Length", stats["median_len"], "bp"),
        stat_card("GC Content", stats["gc_mean"], "%"),
        stat_card("Clusters", str(len(cl_rows))),
        stat_card("Expression cols", stats["expr_cols"]),
    ])

    cluster_desc = """
    <p style='margin:12px 0 6px;font-size:0.85rem;color:#64748b'>
      Copies analysed per cluster (cluster −1 = noise / unassigned by HDBSCAN):
    </p>"""

    cl_h, cl_r = read_csv_rows(d / "03_clustering" / "cluster_summary.csv")
    cl_tbl = html_table(cl_h, cl_r, cols=["cluster","n_total","n_plus","n_minus","pct_plus","pct_minus"])

    return f"""
<div class='cards'>{cards}</div>
{cluster_desc}
{cl_tbl}
<p style='margin-top:10px;font-size:0.82rem;color:#64748b'>
  <strong>n_plus / n_minus</strong>: copies on the forward / reverse strand.
  A near-even split is typical for TEs without strong strand bias.
</p>"""

def sec_clustering(d):
    reports_dir = d / "reports"
    imgs = []
    img_dir = _alignment_images_dir(d)

    # Whole family: not-cleaned vs CIAlign-cleaned. "whole_family_*" is the
    # current stem; "global_*" is only tried if the new one is absent, so
    # pre-restructure result trees still render.
    for label, stem, legacy_stem in [
        ("Raw alignment (before CIAlign cleaning)",
         "whole_family_alignment_input",  "global_alignment_input"),
        ("Cleaned alignment (after CIAlign)",
         "whole_family_alignment_output", "global_alignment_output"),
        ("Markup (removed positions highlighted)",
         "whole_family_alignment_markup", "global_alignment_markup"),
    ]:
        uri = get_img(img_dir, stem) or get_img(img_dir, legacy_stem)
        if uri:
            imgs.append(fig(uri, label))

    # Per-cluster — discovered from the images actually present rather than a
    # hardcoded 0/1/2/-1 list, so families with more clusters aren't truncated.
    cluster_imgs = []
    for stem_prefix in _cluster_image_stems(img_dir):
        for suffix, what in (("_output", "cleaned alignment"),
                             ("_input", "alignment, not cleaned")):
            uri = get_img(img_dir, f"{stem_prefix}{suffix}")
            if uri:
                label = stem_prefix.replace("_alignment", "").replace("cluster_", "Cluster ")
                cluster_imgs.append(fig(uri, f"{label} — {what}"))

    align_stats = read_txt(Path(d) / "04_alignments" / "alignment_stats.txt")
    pre_block = f"<pre>{align_stats}</pre>" if align_stats else ""

    grid_global = f"<div class='fig-grid'>{''.join(imgs)}</div>" if imgs else ""
    grid_clusters = f"<div class='fig-grid'>{''.join(cluster_imgs)}</div>" if cluster_imgs else ""

    desc = """
<p style='font-size:0.85rem;margin-bottom:14px'>
  Sequences are aligned globally with MAFFT and cleaned with <strong>CIAlign</strong>,
  which removes insertions, gap-heavy columns, and divergent sequences to produce
  a representative multiple sequence alignment. The alignment is used for consensus
  generation and divergence estimation.
</p>"""

    return f"""{desc}{pre_block}{grid_global}
{'<h3 style="margin:16px 0 8px;font-size:0.95rem">Per-cluster alignments</h3>' + grid_clusters if grid_clusters else ''}"""

def sec_motif(d):
    body = []
    body.append("""<p style='font-size:0.85rem;margin-bottom:14px'>
      Transcription factor binding sites (TFBS) from <strong>JASPAR 2022 vertebrate core</strong>
      are intersected with TE loci using tabix-indexed BED files. The top overlapping
      motifs genome-wide are shown below.</p>""")

    # Overall top motifs
    h, rows = read_csv_rows(d / "motif_analysis" / "overall_motif_counts.csv", max_rows=20)
    if rows:
        body.append("<h3 style='margin:12px 0 6px;font-size:0.95rem'>Top 20 JASPAR motifs (genome-wide)</h3>")
        body.append(html_table(h, rows))

    # Overall top motifs figure
    uri = get_img(d / "motif_analysis", "overall_top_motifs")
    if uri:
        body.append(fig(uri, "Top JASPAR motifs across all TE loci"))

    # TFBS heatmap (per-cluster binding counts)
    h2, rows2 = read_csv_rows(d / "05_tfbs" / "cluster_tf_binding_counts.csv", max_rows=20)
    if rows2:
        body.append("<h3 style='margin:16px 0 6px;font-size:0.95rem'>Per-cluster TF binding site counts (top 20 rows)</h3>")
        body.append(html_table(h2, rows2))
        body.append("<p class='muted'>Full interactive heatmap: <code>05_tfbs/cluster_tf_binding_heatmap.html</code></p>")

    # Enrichment results
    enrichment_uri = get_img(d / "enrichment_results", "enrichment_heatmap")
    if enrichment_uri:
        body.append("<h3 style='margin:16px 0 6px;font-size:0.95rem'>Cluster-level motif enrichment heatmap</h3>")
        body.append(fig(enrichment_uri, "Fisher's exact test enrichment of JASPAR motifs per cluster"))

    return "".join(body) if any(body) else "<p class='skipped'>TFBS/motif analysis not available for this family.</p>"

def sec_motif_gains(d):
    reports = d / "reports"
    h, rows = read_csv_rows(reports / "motif_gain_summary.csv", max_rows=20)
    if not rows:
        return "<p class='skipped'>Motif-gain analysis not run for this family.</p>"

    body = ["""<p style='font-size:0.85rem;margin-bottom:14px'>
      Motif-gain analysis compares which JASPAR TF binding sites are <em>gained</em>
      relative to a shuffled background. A high <strong>pct_gained</strong> means most
      binding sites for that TF emerged de novo on this TE family, suggesting the TE
      may have contributed regulatory potential genome-wide.</p>"""]
    body.append(f"<h3 style='margin:12px 0 6px;font-size:0.95rem'>Top gained motifs (top 20)</h3>")
    body.append(html_table(h, rows, cols=["motif_name","n_gained","n_total","pct_gained","copies_with_gain","pct_copies","enriched_clusters"]))

    grid = []
    for stem, cap in [("fig_motif_gains_bar", "Motif gains bar chart"),
                      ("fig_motif_gains_heatmap", "Motif gains heatmap (per-cluster)")]:
        uri = get_img(reports, stem)
        if uri:
            grid.append(fig(uri, cap))
    if grid:
        body.append(f"<div class='fig-grid'>{''.join(grid)}</div>")
    return "".join(body)

def sec_go(d):
    ga = d / "go_annotations"
    skipped = read_txt(d / "go_annotations" / "GO_SKIPPED.txt") or read_txt(d / "go_annotations" / "go_skipped.txt")
    if skipped or not ga.exists():
        return "<p class='skipped'>GO annotation analysis was skipped or not applicable.</p>"

    body = ["""<p style='font-size:0.85rem;margin-bottom:14px'>
      Nearby genes (within ±50 kb) of each cluster's TE loci are retrieved and annotated
      with <strong>Gene Ontology (GO) Biological Process</strong> terms via PANTHER/QuickGO.
      Enriched terms suggest potential regulatory influence of the TE on nearby gene function.</p>"""]

    h, rows = read_csv_rows(ga / "go_bp_top_terms.csv", max_rows=20)
    if rows:
        body.append(html_table(h, rows))

    imgs = []
    for f_name in sorted(ga.glob("*.png")):
        uri = b64_img(f_name)
        if uri:
            label = f_name.stem.replace("_", " ").replace("go bp", "GO-BP")
            imgs.append(fig(uri, label))
    if imgs:
        body.append(f"<div class='fig-grid'>{''.join(imgs)}</div>")

    return "".join(body) if len(body) > 1 else "<p class='skipped'>No GO enrichment results found.</p>"

def sec_consensus(d):
    body = ["""<p style='font-size:0.85rem;margin-bottom:14px'>
      Consensus sequences are derived from the cleaned MAFFT alignment per cluster using
      the most-common nucleotide at each position (gaps excluded). They represent the
      "canonical" form of each cluster and are used for LTR structure annotation,
      ORF finding, and primer specificity scoring.</p>"""]

    h, rows = read_csv_rows(Path(d) / "04_alignments" / "consensus_summary.csv")
    for _legacy in ("04_alignments/cluster_consensus_summary.csv",
                    "05_consensus/cluster_consensus_summary.csv",
                    "cluster_alignments/cluster_consensus_summary.csv"):
        if rows:
            break
        h, rows = read_csv_rows(Path(d) / _legacy)
    if rows:
        body.append(html_table(h, rows))
    return "".join(body)

def sec_ltr_struct(d):
    reports = d / "reports"
    h, rows = read_csv_rows(reports / "ltr_struct_summary.csv")
    if not rows:
        return "<p class='skipped'>LTR structure annotation not available (LINE/SINE elements lack LTRs).</p>"

    body = ["""<p style='font-size:0.85rem;margin-bottom:14px'>
      LTR retrotransposons have flanking Long Terminal Repeats (LTRs) that are nearly
      identical upon insertion and accumulate mutations at a known rate. The pipeline
      annotates each copy for: LTR presence / identity, PBS (primer binding site),
      PPT (polypurine tract), and TSD (target site duplications). Solo-LTRs arise
      when the internal region is deleted by recombination between the two flanking LTRs.</p>"""]
    body.append(html_table(h, rows))

    uri = get_img(reports, "fig_ltr_struct")
    if uri:
        body.append(fig(uri, "LTR structure annotation summary per cluster"))
    return "".join(body)

def sec_divergence(d):
    reports = d / "reports"
    h, rows = read_csv_rows(reports / "divergence_stats.csv")

    body = ["""<p style='font-size:0.85rem;margin-bottom:14px'>
      Sequence divergence from the RepBase consensus is estimated with
      <strong>Kimura 2-parameter (K80)</strong> distance on the aligned sequences.
      Under a neutral molecular clock, divergence ≈ 2 × substitution rate × age.
      The repeat landscape histogram shows how copy-number varies with divergence,
      capturing bursts of transposition activity over evolutionary time.</p>"""]

    if rows:
        body.append(html_table(h, rows, cols=["cluster","n","mean_div_pct","median_div_pct","std_div_pct","p10_div_pct","p90_div_pct"]))

    grid = []
    for stem, cap in [
        ("fig_repeat_landscape", "Repeat landscape (copy count vs. divergence %)"),
        ("fig_phylo_divergence", "Per-copy divergence distribution"),
        ("fig_phylo_master", "Phylogenetic age estimate (master element)"),
    ]:
        uri = get_img(reports, stem)
        if uri: grid.append(fig(uri, cap))

    # Also try png in reports dir directly
    for p in (reports / "fig_repeat_landscape.png", reports / "fig_repeat_landscape_per_cluster.png"):
        uri = b64_img(p)
        if uri and not any(cap in g for g in grid for cap in ["landscape"]):
            grid.append(fig(uri, "Repeat landscape"))

    if grid:
        body.append(f"<div class='fig-grid'>{''.join(grid)}</div>")
    return "".join(body)

def sec_phylo(d):
    reports = d / "reports"
    txt = read_txt(reports / "phylo_report.txt")
    if not txt:
        return "<p class='skipped'>Phylogenetic / age analysis not run for this family.</p>"

    info = parse_report(txt)
    cards_keys = [
        ("Copies analysed", "copies"), ("Median divergence", "div%"),
        ("Median est. age", "Myr"), ("Intact (ORF≥thr)", "intact copies"),
    ]
    card_data = [
        ("Copies analysed", info.get("Copies analysed", ""), ""),
        ("Median divergence", info.get("Median divergence", ""), ""),
        ("Estimated age", info.get("Median est. age", ""), ""),
        ("Intact copies", info.get("Intact (ORF≥thr)", ""), ""),
    ]
    cards_html = "".join(stat_card(l, v, u) for l, v, u in card_data if v)

    subfamily_section = ""
    sub_h, sub_rows = read_csv_rows(reports / "subfamily_table.csv", max_rows=20)
    if sub_rows:
        subfamily_section = """<h3 style='margin:16px 0 6px;font-size:0.95rem'>Subfamily table</h3>"""
        subfamily_section += html_table(sub_h, sub_rows)
        for stem in ["fig_subfamily_tree", "fig_subfamily_divergence"]:
            uri = get_img(reports, stem)
            if uri:
                subfamily_section += fig(uri, stem.replace("fig_", "").replace("_", " ").title())

    grid = []
    for stem, cap in [
        ("fig_phylo_master", "Master/source element phylogenetic tree"),
        ("fig_phylo_divergence", "Divergence distribution by cluster"),
        ("fig_phylo_tree", "Subfamily phylogenetic tree"),
    ]:
        uri = get_img(reports, stem)
        if uri: grid.append(fig(uri, cap))

    body = [f"""<p style='font-size:0.85rem;margin-bottom:14px'>
      Phylogenetic age is estimated using the molecular clock model applied to K80
      divergence from the RepBase consensus (rate = {info.get('Assembly / rate', 'mm10 4.5e-9 subs/site/yr')}).
      Intact elements (containing a near-full-length ORF) are potential
      "master elements" — retrotranspositionally active source copies.</p>""",
      f"<div class='cards'>{cards_html}</div>" if cards_html else "",
      f"<div class='fig-grid'>{''.join(grid)}</div>" if grid else "",
      subfamily_section,
    ]
    return "".join(body)

def sec_transduction(d):
    reports = d / "reports"
    txt = read_txt(reports / "transduction_report.txt")
    if not txt:
        return "<p class='skipped'>3′ transduction analysis not run for this family.</p>"

    info = parse_report(txt)
    cards_html = "".join(stat_card(l, v) for l, v in [
        ("Copies", info.get("Copies", "")),
        ("Lineages", info.get("Transduction lineages", "")),
        ("Largest lineage", info.get("Largest lineage", "")),
        ("With poly-A signal", info.get("With poly-A signal", "")),
    ] if v)

    grid = []
    for stem, cap in [
        ("fig_transduction_groups", "Transduction lineage groups"),
        ("fig_transduction_polya", "Poly-A signal / tail length distribution"),
    ]:
        uri = get_img(reports, stem)
        if uri: grid.append(fig(uri, cap))

    return f"""<p style='font-size:0.85rem;margin-bottom:14px'>
      3′ transduction occurs when a TE retrotransposes together with downstream
      unique sequence. Copies sharing identical transduced sequence form a
      <strong>transduction lineage</strong> traceable back to a common source element.
      Poly-A signal and tail annotation helps identify full-length retrotransposition events.</p>
<div class='cards'>{cards_html}</div>
{'<div class="fig-grid">' + ''.join(grid) + '</div>' if grid else ''}"""

def sec_antisense(d):
    reports = d / "reports"
    txt = read_txt(reports / "antisense_report.txt")
    if not txt:
        return "<p class='skipped'>Antisense / bidirectional promoter analysis not run.</p>"

    info = parse_report(txt)
    cards_html = "".join(stat_card(l, v) for l, v in [
        ("Copies", info.get("Copies", "")),
        ("Bidirectional (≥0.4)", info.get("Bidirectional (>=0.4)", "")),
        ("Median bidir. score", info.get("Median bidir. score", "")),
    ] if v)

    grid = []
    for stem, cap in [
        ("fig_antisense_expr", "Expression on forward vs. reverse strand"),
        ("fig_antisense_motifs", "Motif content near antisense TSSs"),
    ]:
        uri = get_img(reports, stem)
        if uri: grid.append(fig(uri, cap))

    return f"""<p style='font-size:0.85rem;margin-bottom:14px'>
      Many TE families act as <strong>bidirectional promoters</strong>: they drive
      expression both on their own strand and, antisense, into adjacent host genes.
      A bidirectionality score ≥ 0.4 (where 0.5 = perfectly symmetric antisense
      expression) flags copies likely functioning as cryptic promoters.</p>
<div class='cards'>{cards_html}</div>
{'<div class="fig-grid">' + ''.join(grid) + '</div>' if grid else ''}"""

def sec_ctcf(d):
    reports = d / "reports"
    txt = read_txt(reports / "ctcf_tad_report.txt")
    if not txt:
        return "<p class='skipped'>CTCF / TAD proximity analysis not run.</p>"

    info = parse_report(txt)
    cards_html = "".join(stat_card(l, v) for l, v in [
        ("Copies", info.get("Copies", "")),
        ("CTCF core motif", info.get("CTCF core motif", "")),
        ("CTCF ChIP overlap", info.get("CTCF ChIP overlap", "n/a")),
        ("Boundary-proximal", info.get("Boundary-proximal", "n/a")),
    ] if v)

    grid = []
    for stem, cap in [
        ("fig_ctcf_overlap", "CTCF motif overlap per cluster"),
        ("fig_ctcf_tad_distance", "Distance to nearest TAD boundary"),
    ]:
        uri = get_img(reports, stem)
        if uri: grid.append(fig(uri, cap))

    return f"""<p style='font-size:0.85rem;margin-bottom:14px'>
      TEs containing a <strong>CTCF binding motif</strong> can rewire chromatin
      architecture by creating ectopic loop anchors and acting as TAD (Topologically
      Associating Domain) boundaries. Copies near existing boundaries may reinforce
      or disrupt gene regulatory domains.</p>
<div class='cards'>{cards_html}</div>
{'<div class="fig-grid">' + ''.join(grid) + '</div>' if grid else ''}"""

def sec_epigenetic(d):
    reports = d / "reports"
    txt = read_txt(reports / "epigenetic_report.txt")
    if not txt:
        return "<p class='skipped'>Epigenetic overlay analysis not run.</p>"

    info = parse_report(txt)
    tracks_used = info.get("Tracks used", "0")
    if tracks_used in ("0", "0  (none)"):
        return f"<p class='skipped'>Epigenetic overlay: no ChIP-seq / ATAC tracks provided ({info.get('Family','')}, {info.get('Copies','?')} copies). Run with --epigenetic-tracks to enable.</p>"

    grid = []
    for stem, cap in [
        ("fig_epigenetic_heatmap", "Epigenetic signal heatmap across TE copies"),
        ("fig_epigenetic_overlap", "Fraction of copies overlapping each track"),
    ]:
        uri = get_img(reports, stem)
        if uri: grid.append(fig(uri, cap))

    return f"""<p style='font-size:0.85rem;margin-bottom:14px'>
      Epigenetic signals (H3K27ac, H3K4me3, ATAC, etc.) overlaid on TE loci reveal
      which copies are in active chromatin. Cluster-level differences indicate
      functionally distinct subfamilies with different regulatory activity.</p>
<div class='cards'>{stat_card('Tracks used', tracks_used)}</div>
{'<div class="fig-grid">' + ''.join(grid) + '</div>' if grid else ''}"""

def sec_primers(d):
    h, rows = read_csv_rows(d / "06_primers" / "selected_primers_summary.csv", max_rows=10)
    if not rows:
        return "<p class='skipped'>Primer design not run for this family.</p>"

    top_h, top_rows = read_csv_rows(d / "06_primers" / "cluster_top_primers.csv", max_rows=10)

    return f"""<p style='font-size:0.85rem;margin-bottom:14px'>
      18-mer primers are designed to maximally cover TE copies while minimising
      off-target binding to non-TE genomic sequence. Coverage is computed as the
      number of TE copies containing an exact match; total expression weights
      high-expression copies. The selected primers are the Pareto-optimal set
      balancing <strong>coverage, specificity, Tm, and GC content</strong>.</p>
<h3 style='margin:12px 0 6px;font-size:0.95rem'>Selected primers</h3>
{html_table(h, rows, cols=['primer','coverage','total_expr','strategy'])}
{'<h3 style="margin:16px 0 6px;font-size:0.95rem">Top per-cluster primers</h3>' + html_table(top_h, top_rows) if top_rows else ''}"""

def sec_fold(d):
    reports = d / "reports"
    txt = read_txt(reports / "fold_report.txt")
    if not txt:
        return "<p class='skipped'>Protein fold prediction (ColabFold) not run for this family.</p>"

    info = parse_report(txt)
    cards_html = "".join(stat_card(l, v) for l, v in [
        ("Unique ORFs", info.get("Unique ORFs found", "")),
        ("Structures folded", info.get("Structures folded", "")),
        ("Best model", info.get("Name", "")),
        ("Best pLDDT", info.get("Mean pLDDT", "")),
        ("Best max PAE", info.get("Max PAE", "")),
        ("Mean pLDDT (all)", info.get("Mean pLDDT (all)", "")),
    ] if v)

    body = [f"""<p style='font-size:0.85rem;margin-bottom:14px'>
      Open reading frames (ORFs) in the consensus sequence are translated and folded
      with <strong>ColabFold (AlphaFold2)</strong>. <em>pLDDT</em> (0–100) is a
      per-residue confidence score: &gt;90 = very high, 70–90 = confident,
      50–70 = low/disordered, &lt;50 = unreliable. <em>PAE</em> (Predicted Aligned Error)
      reflects inter-domain positional uncertainty.</p>""",
      f"<div class='cards'>{cards_html}</div>",
    ]

    # Summary figures
    grid_summary = []
    for stem, cap in [
        ("fig_fold_summary", "pLDDT summary across all ORFs"),
        ("fig_fold_plddt", "Per-residue pLDDT profile"),
        ("fig_fold_orf_map", "ORF map on consensus"),
    ]:
        uri = get_img(reports, stem)
        if uri: grid_summary.append(fig(uri, cap))
    if grid_summary:
        body.append(f"<div class='fig-grid'>{''.join(grid_summary)}</div>")

    # Per-ORF ColabFold plots
    cf_dir = reports / "colabfold_output"
    if cf_dir.exists():
        body.append("<h3 style='margin:16px 0 8px;font-size:0.95rem'>Per-ORF ColabFold outputs</h3>")
        orf_pngs = {}
        for p in sorted(cf_dir.glob("*_coverage.png")):
            orf_name = p.name.replace("_coverage.png", "")
            orf_pngs.setdefault(orf_name, {})["coverage"] = p
        for p in sorted(cf_dir.glob("*_pae.png")):
            orf_name = p.name.replace("_pae.png", "")
            orf_pngs.setdefault(orf_name, {})["pae"] = p
        for p in sorted(cf_dir.glob("*_plddt.png")):
            orf_name = p.name.replace("_plddt.png", "")
            orf_pngs.setdefault(orf_name, {})["plddt"] = p

        for orf_name, figs_dict in orf_pngs.items():
            # Load scores JSON for pLDDT
            score_json = cf_dir / f"{orf_name}_scores_rank_001_alphafold2_ptm_model_1_seed_000.json"
            plddt_mean = ""
            if score_json.exists():
                try:
                    sc = json.loads(score_json.read_text())
                    vals = sc.get("plddt", [])
                    if vals:
                        plddt_mean = f"{sum(vals)/len(vals):.1f}"
                except Exception:
                    pass
            orf_label = orf_name.replace("_", " ")
            header = f"<h4 style='margin:12px 0 6px;font-size:0.88rem;color:#1e40af'>{orf_label}" + \
                     (f" — mean pLDDT: <strong>{plddt_mean}</strong>" if plddt_mean else "") + "</h4>"
            grid_orf = []
            for key, cap in [("coverage", "MSA coverage"), ("pae", "PAE matrix"), ("plddt", "pLDDT profile")]:
                if key in figs_dict:
                    uri = b64_img(figs_dict[key])
                    if uri: grid_orf.append(fig(uri, cap))
            body.append(header + f"<div class='fig-grid'>{''.join(grid_orf)}</div>")

    return "".join(body)

def sec_grna(d):
    reports = d / "reports"
    txt = read_txt(reports / "grna_offtarget_report.txt")
    if not txt:
        return "<p class='skipped'>gRNA off-target analysis not run for this family.</p>"

    info = parse_report(txt)
    cards_html = "".join(stat_card(l, v) for l, v in [
        ("Cas / PAM", info.get("Cas / PAM", "")),
        ("Family copies", info.get("Family copies", "")),
        ("Candidate guides", info.get("Candidate guides", "")),
        ("Pareto-frontier", info.get("Pareto-frontier", "")),
        ("Best guide", info.get("Guide", "")),
        ("Coverage", info.get("Coverage", "")),
        ("Off-targets", info.get("Off-target sites", "")),
    ] if v)

    h, rows = read_csv_rows(reports / "grna_offtarget_guides.csv", max_rows=10)

    grid = []
    for stem, cap in [
        ("fig_grna_offtarget_top", "Top guides: coverage vs. off-target"),
        ("fig_grna_offtarget_pareto", "Pareto frontier (coverage / specificity)"),
        ("fig_grna_offtarget_mm", "Mismatch tolerance profile"),
    ]:
        uri = get_img(reports, stem)
        if uri: grid.append(fig(uri, cap))

    return f"""<p style='font-size:0.85rem;margin-bottom:14px'>
      SpCas9 gRNAs are designed to target as many TE copies as possible
      (<strong>coverage</strong>) while minimising mismatched off-target cuts
      in the rest of the genome. The Pareto frontier contains guides where no
      other guide beats them on both dimensions simultaneously.</p>
<div class='cards'>{cards_html}</div>
{'<h3 style="margin:12px 0 6px;font-size:0.95rem">Top candidate guides</h3>' + html_table(h, rows, cols=['guide','n_targets','n_offtarget','coverage_pct']) if rows else ''}
{'<div class="fig-grid">' + ''.join(grid) + '</div>' if grid else ''}"""

def sec_multiassembly(d):
    reports = d / "reports"
    txt = read_txt(reports / "multiassembly_report.txt")
    if not txt:
        return "<p class='skipped'>Multi-assembly liftover not run for this family.</p>"
    info = parse_report(txt)
    targets = info.get("Target assemblies", "0  (none)")
    if targets.startswith("0"):
        return f"<p class='skipped'>Multi-assembly liftover: no target assemblies specified. ({info.get('Source assembly','mm10')} source, {info.get('Copies','?')} copies).</p>"

    grid = []
    for stem, cap in [
        ("fig_multiassembly_mapping", "Liftover mapping rate per assembly"),
        ("fig_multiassembly_coords", "Coordinate shifts across assemblies"),
    ]:
        uri = get_img(reports, stem)
        if uri: grid.append(fig(uri, cap))
    return f"""<p style='font-size:0.85rem;margin-bottom:14px'>
      Copies are lifted over across genome assemblies using UCSC liftOver chains,
      tracking insertion presence/absence to identify assembly-specific insertions
      and validate copy coordinates.</p>
<pre>{txt[:1000]}</pre>
{'<div class="fig-grid">' + ''.join(grid) + '</div>' if grid else ''}"""

def sec_ortholog(d):
    reports = d / "reports"
    txt = read_txt(reports / "ortholog_report.txt")
    if not txt:
        return "<p class='skipped'>Ortholog insertion analysis not run for this family.</p>"
    info = parse_report(txt)
    if info.get("Target species", "0  (none)").startswith("0"):
        return f"<p class='skipped'>Ortholog insertion: no target species specified. Lineage-specific insertions require cross-species chain files.</p>"

    grid = []
    for stem, cap in [("fig_ortholog_presence","Insertion presence across species"),
                       ("fig_ortholog_summary","Summary of lineage-specific copies")]:
        uri = get_img(reports, stem)
        if uri: grid.append(fig(uri, cap))
    return f"""<p style='font-size:0.85rem;margin-bottom:14px'>
      Ortholog insertion analysis uses UCSC net/chain files to determine whether each
      TE copy is <em>lineage-specific</em> (absent in the outgroup) or <em>shared</em>
      (present in both species, predating the split). Lineage-specific insertions are
      candidates for species-specific regulatory innovation.</p>
<pre>{txt[:1000]}</pre>
{'<div class="fig-grid">' + ''.join(grid) + '</div>' if grid else ''}"""

# ── TOC & page assembly ───────────────────────────────────────────────────────

TOC_ITEMS = [
    ("overview",      "Overview"),
    ("alignment",     "Alignment (CIAlign)"),
    ("motif",         "TFBS / Motif Analysis"),
    ("motif-gains",   "Motif Gains"),
    ("go",            "GO Annotations"),
    ("consensus",     "Consensus Sequences"),
    ("ltr-struct",    "LTR Structure"),
    ("divergence",    "Divergence & Landscape"),
    ("phylo",         "Phylogenetics"),
    ("transduction",  "3′ Transduction"),
    ("antisense",     "Antisense / Bidir. Promoter"),
    ("ctcf",          "CTCF / TAD Proximity"),
    ("epigenetic",    "Epigenetic Overlay"),
    ("primers",       "Primer Design"),
    ("fold",          "Protein Fold (ColabFold)"),
    ("grna",          "gRNA Off-target"),
    ("multiassembly", "Multi-assembly Liftover"),
    ("ortholog",      "Ortholog Insertion"),
]

SECTION_DEFS = [
    ("overview",      "📊", "Dataset Overview",               sec_overview),
    ("alignment",     "🔬", "Multiple Sequence Alignment",     sec_clustering),
    ("motif",         "🎯", "TFBS / Motif Analysis",           sec_motif),
    ("motif-gains",   "📈", "Motif Gains",                     sec_motif_gains),
    ("go",            "🔗", "GO Biological Process Enrichment",sec_go),
    ("consensus",     "🧬", "Consensus Sequences",             sec_consensus),
    ("ltr-struct",    "🏗️",  "LTR Structure Annotation",       sec_ltr_struct),
    ("divergence",    "📉", "Divergence & Repeat Landscape",   sec_divergence),
    ("phylo",         "🌳", "Phylogenetics & Age Estimation",  sec_phylo),
    ("transduction",  "➡️",  "3′ Transduction",                sec_transduction),
    ("antisense",     "↔️",  "Antisense / Bidirectional Promoter", sec_antisense),
    ("ctcf",          "🔒", "CTCF / TAD Proximity",            sec_ctcf),
    ("epigenetic",    "🎨", "Epigenetic Overlay",              sec_epigenetic),
    ("primers",       "🔧", "Primer Design",                   sec_primers),
    ("fold",          "🧪", "Protein Fold Prediction",         sec_fold),
    ("grna",          "✂️",  "gRNA Off-target Analysis",       sec_grna),
    ("multiassembly", "🗺️",  "Multi-assembly Liftover",        sec_multiassembly),
    ("ortholog",      "🐭", "Ortholog Insertion Analysis",     sec_ortholog),
]

def generate_report(family_dir_str):
    d = Path(family_dir_str).resolve()
    if not d.is_dir():
        print(f"ERROR: {d} is not a directory", file=sys.stderr)
        return

    family = d.name
    print(f"\n[{family}] Reading data…")

    stats_txt = read_txt(d / "02_statistics" / "overall_statistics.txt")
    stats = parse_stats(stats_txt)
    family_display = stats["family"] or family

    _, cl_rows = read_csv_rows(d / "03_clustering" / "cluster_summary.csv")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Detect TE class from family name heuristic
    fn_lower = family_display.lower()
    te_class = "LTR retrotransposon" if any(x in fn_lower for x in ["ltr","iap","mt2","ervk","l1ltr"]) \
               else "LINE retrotransposon" if "l1" in fn_lower or "line" in fn_lower \
               else "SINE" if "sine" in fn_lower or "b1" in fn_lower or "b2" in fn_lower or "alu" in fn_lower \
               else "Transposable Element"

    # Build sidebar
    toc_links = "\n".join(f"<a href='#{sid}'>{label}</a>" for sid, label in TOC_ITEMS)
    sidebar = f"""
<div id='sidebar'>
  <div class='brand'>
    <h2>GAMECA</h2>
    <p>{family_display}</p>
    <p style='font-size:0.7rem;margin-top:4px;color:#475569'>Report generated {now}</p>
  </div>
  <h3>Sections</h3>
  {toc_links}
</div>"""

    # Build header
    header = f"""
<div class='report-header'>
  <h1>{family_display}</h1>
  <div class='meta'>
    {te_class} · {stats['n_copies']} genomic copies · Assembly: mm10 ·
    Report generated {now}
  </div>
</div>"""

    # Build each section
    print(f"[{family}] Building sections…")
    sections_html = []
    for sid, icon, title, fn in SECTION_DEFS:
        try:
            if sid == "overview":
                body = fn(d, stats, cl_rows)
            else:
                body = fn(d)
        except Exception as e:
            body = f"<p class='note'>Error generating section: {e}</p>"
        sections_html.append(section_html(sid, icon, title, body))

    main_content = f"""
<div id='main'>
  {header}
  {''.join(sections_html)}
  <footer style='margin-top:40px;padding-top:20px;border-top:1px solid #e2e8f0;
    font-size:0.78rem;color:#94a3b8;text-align:center'>
    Generated by GAMECA · {family_display} · {now}
  </footer>
</div>"""

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>GAMECA Report — {family_display}</title>
<style>{CSS}</style>
</head>
<body>
{sidebar}
{main_content}
<script>{JS}</script>
</body>
</html>"""

    out_path = d / "report.html"
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"[{family}] ✓ Written: {out_path}  ({size_mb:.1f} MB)")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate GAMECA HTML report for one or more family folders.")
    ap.add_argument("folders", nargs="+", help="Family output folder(s)")
    args = ap.parse_args()

    for folder in args.folders:
        generate_report(folder)

if __name__ == "__main__":
    main()
