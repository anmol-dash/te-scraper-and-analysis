#!/usr/bin/env python3
"""
generate_latex_report.py — Formal LaTeX/PDF analysis report for GAMECA TE family folders.
Renders ColabFold PDB structures with matplotlib + BioPython.
Usage: python generate_latex_report.py mt2_mm iapltr1a_mm l1md_gf
"""

import argparse, csv, json, math, os, re, shutil, subprocess, sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from Bio.PDB import PDBParser

# Resolve pdflatex portably: explicit override, then whatever is on PATH
# (cluster/Linux), then the local TinyTeX install as a last resort. This keeps
# the report buildable on HPC nodes, not just the original macOS machine.
PDFLATEX = (os.environ.get("PDFLATEX")
            or shutil.which("pdflatex")
            or "/Users/anmol/Library/TinyTeX/bin/universal-darwin/pdflatex")
BASE = Path(__file__).parent

# ── Column display names ──────────────────────────────────────────────────────
COLNAME = {
    'cluster': 'Cluster', 'n_total': r'$N$', 'n': r'$N$',
    'n_plus': r'Strand $+$', 'n_minus': r'Strand $-$',
    'pct_plus': r'\% $+$', 'pct_minus': r'\% $-$',
    'mean_div_pct': r'Mean div.\ (\%)', 'median_div_pct': r'Median div.\ (\%)',
    'std_div_pct': r'SD (\%)', 'p10_div_pct': r'P10 (\%)', 'p90_div_pct': r'P90 (\%)',
    'n_full_length': 'Full-length', 'n_solo_ltr': 'Solo-LTR',
    'n_truncated': 'Truncated', 'n_pbs': 'PBS', 'n_ppt': 'PPT',
    'mean_ltr_identity': 'LTR identity',
    'Motif_name': 'Motif', 'motif_name': 'Motif', 'count': 'Count', 'Count': 'Count',
    'n_gained': r'$N$ gained', 'pct_gained': r'\% gained',
    'copies_with_gain': 'Copies w/ gain', 'pct_copies': r'\% copies',
    'enriched_clusters': 'Clusters',
    'primer': 'Primer (18-mer)', 'coverage': 'Coverage', 'strategy': 'Strategy',
    'total_expr': 'Total expr.', 'rows_covered': r'$N$ covered',
    'Guide': 'Guide (20-mer)', 'guide': 'Guide (20-mer)',
    'n_targets': 'Targets', 'n_offtarget': 'Off-targets', 'coverage_pct': r'\% coverage',
    'offtarget': 'Off-tgt.', 'cov_pct': r'Cov.\ \%', 'on_target': 'On-tgt.',
    'specificity': 'Spec.', 'pareto': 'Pareto',
    'GO_Term': 'GO term', 'go_term': 'GO term',
    'consensus_length': 'Length (bp)', 'gc_content': r'GC (\%)',
    'mean_length': 'Mean len.\ (bp)', 'median_length': 'Median len.\ (bp)',
    'n_sequences': 'Sequences',
    'orf_name': 'ORF', 'mean_plddt': 'Mean pLDDT', 'max_pae': r'Max PAE (\AA)',
    'n_residues': 'Residues', 'ptm': 'pTM',
    'Subfamily': 'Subfamily', 'N_copies': r'$N$ copies', 'Mean_div': r'Mean div.\ (\%)',
    # consensus summary
    'size': 'Size', 'consensus_length': 'Length (bp)', 'fallback': 'Fallback',
    'avg_gaps_per_col': r'Avg.\ gaps/col', 'cols_majority_gap': 'Gap-major cols',
    'cols_no_gaps': 'Gapless cols', 'alignment_length': 'Aln.\ len.',
    'consensus_length_no_gaps': 'Cons.\ len.',
    # subfamily table
    'n_loci': r'$N$ loci', 'consensus_length_bp': 'Length (bp)',
    'div_from_canonical_pct': r'Div.\ canon.\ (\%)', 'subfamily_label': 'Subfamily',
    'is_canonical_type': 'Canonical',
    # ltr per-locus
    'Chromosome': 'Chrom.', 'Start': 'Start', 'Stop': 'Stop', 'strand': 'Strand',
    'classification': 'Class', 'ltr_identity': 'LTR id.',
    'pbs_found': 'PBS', 'ppt_found': 'PPT',
    # transduction per-copy
    'TE_name': 'TE name', 'lineage_id': 'Lineage', 'transduced_seq_len': 'Transd.\ len.',
    'polya_signal': 'Poly-A sig.', 'polya_tail': 'Poly-A tail',
}

# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_txt(path):
    p = Path(path)
    return p.read_text(errors='replace').strip() if p.exists() else ''

def read_csv(path, max_rows=None):
    p = Path(path)
    if not p.exists(): return [], []
    with open(p, newline='', errors='replace') as f:
        rows = list(csv.DictReader(f))
    headers = list(rows[0].keys()) if rows else []
    return headers, (rows[:max_rows] if max_rows else rows)

def parse_report(text):
    out = {}
    for line in text.splitlines():
        line = line.strip().lstrip('─ ')
        if ':' in line and not line.startswith(('=', '-', 'GAMECA', 'Generated')):
            k, _, v = line.partition(':')
            k = k.strip(); v = v.strip()
            if k and v and len(k) < 55 and '=' not in k:
                out[k] = v
    return out

def parse_stats(text):
    out = {}
    for line in text.splitlines():
        m = re.search(r'STATISTICS\s*[—\-]+\s*(.+)', line)
        if m: out['family'] = m.group(1).strip()
        if 'Dataset size:' in line:
            m2 = re.search(r'(\d[\d,]*)\s+sequences', line)
            if m2: out['n_copies'] = m2.group(1).replace(',', '')
        if re.match(r'\s*Mean:\s+[\d.]+\s+Median:', line) and 'mean_len' not in out:
            nums = re.findall(r'[\d.]+', line)
            if len(nums) >= 2: out['mean_len'] = nums[0]; out['median_len'] = nums[1]
        if re.match(r'\s*Std:\s+[\d.]+\s+Range:', line) and 'std_len' not in out:
            m2 = re.search(r'Std:\s+([\d.]+)', line); m3 = re.search(r'Range:\s*\[(.+?)\]', line)
            if m2: out['std_len'] = m2.group(1)
            if m3: out['range_len'] = m3.group(1)
        if re.match(r'\s*Mean:\s+[\d.]+\s+\([\d.]+%\)', line) and 'gc_mean' not in out:
            m2 = re.search(r'\(([\d.]+)%\)', line)
            if m2: out['gc_mean'] = m2.group(1)
        if 'Total expression' in line:
            m2 = re.search(r'mean:\s*([\d.e+]+)', line)
            if m2: out['expr_mean'] = m2.group(1)
        if 'Columns:' in line and 'expr_cols' not in out:
            m2 = re.search(r'Columns:\s*(\d+)', line)
            if m2: out['expr_cols'] = m2.group(1)
    return out

def get_fig(folder, stem):
    """Return path to figure (PNG preferred over PDF), or None."""
    for ext in ('.png', '.pdf', '.jpg', '.jpeg'):
        p = Path(folder) / (stem + ext)
        if p.exists(): return p
    return None

def detect_assembly(d):
    """Infer assembly and organism from data files. Returns (assembly, organism_latex)."""
    for pat in ['01_data/*_clustered.csv', 'alusx1_sequences.csv', '*_sequences.csv']:
        found = list(Path(d).glob(pat))
        if not found:
            continue
        try:
            with open(found[0], newline='', errors='replace') as f:
                header = f.readline()
                line2 = f.readline()
            combined = (header + line2).lower()
            if 'hg38' in combined or 'grch38' in combined:
                return 'hg38', r'human (\textit{Homo sapiens})'
            if 'hg19' in combined or 'grch37' in combined:
                return 'hg19', r'human (\textit{Homo sapiens})'
            if 'mm10' in combined or 'grcm38' in combined:
                return 'mm10', r'mouse (\textit{Mus musculus})'
            if 'mm39' in combined or 'grcm39' in combined:
                return 'mm39', r'mouse (\textit{Mus musculus})'
        except Exception:
            pass
    return 'mm10', r'mouse (\textit{Mus musculus})'

def detect_te_class(d):
    """Read repClass/repFamily from clustered CSV to describe the TE family."""
    for pat in ['01_data/*_clustered.csv']:
        found = list(Path(d).glob(pat))
        if not found:
            continue
        try:
            with open(found[0], newline='', errors='replace') as f:
                reader = csv.DictReader(f)
                row = next(reader, {})
            rep_class  = row.get('repClass', row.get('repclass', '')).strip()
            rep_family = row.get('repFamily', row.get('repfamily', '')).strip()
            if rep_class and rep_family:
                return f'{rep_class}/{rep_family}'
            if rep_class:
                return rep_class
        except Exception:
            pass
    return ''

def pregenerate_seq_figures(d, out_dir):
    """Generate basic sequence-statistics PNG figures from the clustered CSV.
    Returns dict of label -> Path for figures successfully written."""
    d = Path(d); out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clustered = None
    for pat in ['01_data/*_clustered.csv']:
        found = list(d.glob(pat))
        if found:
            clustered = found[0]
            break
    if not clustered:
        return {}

    # Read all rows (116K rows ~50 MB — acceptable)
    with open(clustered, newline='', errors='replace') as f:
        rows_raw = list(csv.DictReader(f))
    if not rows_raw:
        return {}

    lengths, millidivs, swscores, chroms = [], [], [], []
    for r in rows_raw:
        try:
            lengths.append(int(r.get('Stop', r.get('stop', 0))) -
                           int(r.get('Start', r.get('start', 0))))
        except (ValueError, TypeError):
            pass
        try:
            millidivs.append(float(r['milliDiv']))
        except (KeyError, ValueError, TypeError):
            pass
        try:
            swscores.append(float(r['swScore']))
        except (KeyError, ValueError, TypeError):
            pass
        c = r.get('Chromosome', r.get('chr', '')).strip()
        if c:
            chroms.append(c)

    # GC from sequences — sample at most 15 000 rows for speed
    step = max(1, len(rows_raw) // 15000)
    gcs = []
    for r in rows_raw[::step]:
        seq = r.get('Seq', r.get('sequence', '')).upper()
        if seq:
            n = len(seq)
            gcs.append((seq.count('G') + seq.count('C')) / n)

    figs = {}

    # ── Panel 1: 2×2 sequence statistics ────────────────────────────
    n_panels = sum([bool(lengths), bool(gcs), bool(millidivs), bool(swscores)])
    if n_panels:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes = axes.flat
        panel_i = 0
        ax_list = list(axes)

        def _hist(ax, data, xlabel, title, color):
            ax.hist(data, bins=60, color=color, edgecolor='none', alpha=0.85)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=8)

        if lengths:
            _hist(ax_list[0], lengths, 'Sequence length (bp)',
                  'Sequence Length Distribution', '#2E86AB')
        else:
            ax_list[0].axis('off')
        if gcs:
            _hist(ax_list[1], gcs, 'GC content (fraction)',
                  'GC Content Distribution', '#A23B72')
        else:
            ax_list[1].axis('off')
        if millidivs:
            _hist(ax_list[2], millidivs, 'RepeatMasker milliDiv',
                  'Divergence from Consensus', '#F18F01')
        else:
            ax_list[2].axis('off')
        if swscores:
            _hist(ax_list[3], swscores, 'Smith-Waterman score',
                  'SW Alignment Score', '#C73E1D')
        else:
            ax_list[3].axis('off')

        fig.suptitle(f'{d.name} — Sequence Statistics ({len(rows_raw):,} copies)',
                     fontsize=12, fontweight='bold')
        fig.tight_layout()
        p = out_dir / 'fig_seq_stats_panel.png'
        fig.savefig(str(p), dpi=150, facecolor='white')
        plt.close(fig)
        figs['seq_stats_panel'] = p

    # ── Panel 2: Chromosome distribution ────────────────────────────
    if chroms:
        chrom_counts = {}
        for c in chroms:
            chrom_counts[c] = chrom_counts.get(c, 0) + 1

        def _chrom_key(c):
            c2 = c.replace('chr', '').replace('Chr', '')
            try:
                return (0, int(c2), '')
            except ValueError:
                if c2 in ('X',): return (1, 0, c2)
                if c2 in ('Y',): return (1, 1, c2)
                if c2 in ('M', 'MT', 'Mt'): return (2, 0, c2)
                return (3, 0, c2)

        sorted_chroms = sorted(chrom_counts.keys(), key=_chrom_key)
        vals = [chrom_counts[c] for c in sorted_chroms]
        fig, ax = plt.subplots(figsize=(max(8, len(sorted_chroms) * 0.45), 4))
        ax.bar(range(len(sorted_chroms)), vals, color='#2E86AB', edgecolor='none', alpha=0.85)
        ax.set_xticks(range(len(sorted_chroms)))
        ax.set_xticklabels(sorted_chroms, rotation=55, ha='right', fontsize=7)
        ax.set_xlabel('Chromosome', fontsize=9)
        ax.set_ylabel('Copy count', fontsize=9)
        ax.set_title(f'{d.name} — Chromosomal Distribution of TE Copies', fontsize=11, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        p = out_dir / 'fig_chrom_distribution.png'
        fig.savefig(str(p), dpi=150, facecolor='white')
        plt.close(fig)
        figs['chrom_distribution'] = p

    return figs

# ── LaTeX text helpers ────────────────────────────────────────────

def esc(s):
    """Escape LaTeX special characters in plain text, replace common unicode."""
    if not isinstance(s, str): s = str(s)
    # 1. Escape backslash first
    s = s.replace('\\', r'\textbackslash{}')
    # 2. Escape other LaTeX special chars (including $)
    for ch, rep in [('&', r'\&'), ('%', r'\%'), ('$', r'\$'), ('#', r'\#'),
                    ('^', r'\textasciicircum{}'), ('~', r'\textasciitilde{}'),
                    ('{', r'\{'), ('}', r'\}'), ('_', r'\_')]:
        s = s.replace(ch, rep)
    # 3. Replace unicode chars (these may add $..$ so must come AFTER $ escaping)
    unicode_subs = [
        ('\u2265', r'$\geq$'), ('\u2264', r'$\leq$'), ('\u2192', r'$\to$'),
        ('\u00b1', r'$\pm$'), ('\u00d7', r'$\times$'), ('\u2260', r'$\neq$'),
        ('\u2248', r'$\approx$'), ('\u221e', r'$\infty$'),
        ('\u03b1', r'$\alpha$'), ('\u03b2', r'$\beta$'),
        ('\u03bc', r'$\mu$'), ('\u03c3', r'$\sigma$'), ('\u00b0', r'${}^\circ$'),
        ('\u2018', "'"), ('\u2019', "'"), ('\u201c', "``"), ('\u201d', "''"),
        ('\u2013', '--'), ('\u2014', '---'), ('\u00a0', ' '),
    ]
    for code, rep in unicode_subs:
        s = s.replace(code, rep)
    return s

def fmtcol(name):
    return COLNAME.get(name, esc(name.replace('_', ' ').title()))

def _is_number(s):
    try:
        float(s); return True
    except (ValueError, TypeError):
        return False

def fmtnum(s):
    """Render a numeric string compactly: thousands separators for integers,
    3-significant-figure rounding for floats, scientific notation for very
    large/small magnitudes. Returns a LaTeX-ready string."""
    f = float(s)
    if f != f:                       # NaN
        return 'n/a'
    a = abs(f)
    # integers (within float tolerance)
    if a < 1e15 and float(int(round(f))) == round(f, 6):
        return f'{int(round(f)):,}'.replace(',', '\\,')
    # very large / very small -> scientific with LaTeX exponent
    if a >= 1e6 or (a < 1e-3 and a > 0):
        m = f'{f:.3g}'
        if 'e' in m:
            mant, exp = m.split('e')
            return f'${mant}\\times10^{{{int(exp)}}}$'
        return m
    # moderate float -> up to 3 decimals, strip trailing zeros
    out = f'{f:.3f}'.rstrip('0').rstrip('.')
    return out if out not in ('', '-') else '0'

def fmtval(val):
    """Format a cell value: round numbers, truncate over-long strings."""
    s = str(val).strip()
    if s == '':
        return ''
    if _is_number(s):
        return fmtnum(s)
    if len(s) > 60:
        s = s[:57] + '...'
    return esc(s)

def _col_spec(headers, rows):
    """Choose a per-column alignment that keeps the table inside \\textwidth:
    numeric columns are right-aligned (narrow), short text left-aligned, and
    long free-text columns wrapped in a p{} box sized to the widest cell."""
    specs = []
    for h in headers:
        raw = [str(r.get(h, '')).strip() for r in rows]
        nonempty = [v for v in raw if v != '']
        is_num = bool(nonempty) and all(_is_number(v) for v in nonempty)
        maxlen = max([len(h)] + [len(v) for v in raw]) if raw else len(h)
        has_space = any(' ' in v for v in nonempty)
        if is_num:
            specs.append('r')
        elif maxlen <= 24 or not has_space:
            # short text, or an unbreakable token (e.g. a DNA primer/guide):
            # set at natural width — a p{} box would force an overfull line.
            specs.append('l')
        else:
            # genuinely long prose that can wrap on spaces (e.g. GO terms)
            width = min(7.5, max(3.0, maxlen * 0.16))
            specs.append(f'p{{{width:.1f}cm}}')
    return ''.join(specs)

def longtable(headers, rows, caption, label, cols=None, note='', small=False):
    if not rows: return f'% No data for table {label}\n'
    if cols: headers = [h for h in headers if h in cols]
    rows_f = [{h: fmtval(r.get(h, '')) for h in headers} for r in rows]
    n = len(headers)
    # For >=3 columns, lead with \extracolsep{\fill} so the table stretches
    # evenly to the full text width instead of sitting narrow. Two-column
    # tables look cleaner centred at natural width than with one huge gap.
    base_spec = _col_spec(headers, rows)
    col_spec = ('@{\\extracolsep{\\fill}}' + base_spec) if n >= 3 else base_spec
    hdr = ' & '.join(fmtcol(h) for h in headers) + r' \\'
    body = ''.join('  ' + ' & '.join(r[h] for h in headers) + r' \\' + '\n' for r in rows_f)
    # Footnote spans the full table width in a wrapping p-box so long notes
    # never overrun the page margin.
    note_line = (fr'  \multicolumn{{{n}}}{{@{{}}p{{0.95\linewidth}}@{{}}}}{{\small\textit{{{esc(note)}}}}}\\' + '\n') if note else ''
    use_small = small or n > 5
    open_grp = '\\begingroup\n\\small\n' if use_small else ''
    close_grp = '\\endgroup\n' if use_small else ''
    # Ensure the table never starts with too little room (otherwise a heading's
    # no-break glue can force its first rows past the bottom margin).
    return (
        '\\Needspace*{0.2\\textheight}\n'
        f'{open_grp}'
        f'\\begin{{longtable}}{{{col_spec}}}\n'
        f'  \\caption{{{esc(caption)}}}\\label{{{label}}}\\\\\n'
        f'  \\toprule\n  {hdr}\n  \\midrule\n  \\endfirsthead\n'
        f'  \\multicolumn{{{n}}}{{l}}{{\\small\\textit{{Table \\thetable\\ -- continued}}}}\\\\\n'
        f'  \\toprule\n  {hdr}\n  \\midrule\n  \\endhead\n'
        f'  \\midrule\\multicolumn{{{n}}}{{r}}{{\\small\\textit{{Continued on next page}}}}\\\\\n'
        f'  \\endfoot\n  \\bottomrule\n{note_line}  \\endlastfoot\n'
        f'{body}\\end{{longtable}}\n'
        f'{close_grp}'
    )

def fig_latex(path, caption, label, width=r'0.88\linewidth', placement='H', short=None):
    p = Path(path)
    if not p.exists(): return f'% Figure missing: {path}\n'
    cap = (f'\\caption[{esc(short)}]{{{esc(caption)}}}' if short
           else f'\\caption{{{esc(caption)}}}')
    # Place exactly here ([H], non-floating, so figures never pile up and dump
    # onto later pages), but cap the image height and require enough vertical
    # room first (\Needspace) so the figure + caption can never run off the
    # bottom of the page; if it does not fit, the page breaks cleanly.
    return (
        '\\Needspace*{0.66\\textheight}\n'
        f'\\begin{{figure}}[{placement}]\n'
        f'  \\centering\n'
        f'  \\includegraphics[width={width},height=0.6\\textheight,keepaspectratio]{{{str(p)}}}\n'
        f'  {cap}\n'
        f'  \\label{{{label}}}\n'
        f'\\end{{figure}}\n'
    )

def twofigs(p1, c1, l1, p2, c2, l2, w=r'0.48\linewidth'):
    e1, e2 = Path(p1).exists() if p1 else False, Path(p2).exists() if p2 else False
    if not e1 and not e2: return ''
    if not e1: return fig_latex(p2, c2, l2)
    if not e2: return fig_latex(p1, c1, l1)
    # Place exactly here ([H], non-floating) with a height cap and a \Needspace
    # guard, so the pair can never pile up or run off the bottom of the page.
    g = r'\includegraphics[width=\linewidth,height=0.42\textheight,keepaspectratio]'
    return (
        '\\Needspace*{0.5\\textheight}\n'
        f'\\begin{{figure}}[H]\n  \\centering\n'
        f'  \\begin{{subfigure}}{{{w}}}\n    \\centering\n'
        f'    {g}{{{str(Path(p1))}}}\n'
        f'    \\caption{{{esc(c1)}}}\n    \\label{{{l1}}}\n  \\end{{subfigure}}\n  \\hfill\n'
        f'  \\begin{{subfigure}}{{{w}}}\n    \\centering\n'
        f'    {g}{{{str(Path(p2))}}}\n'
        f'    \\caption{{{esc(c2)}}}\n    \\label{{{l2}}}\n  \\end{{subfigure}}\n'
        f'\\end{{figure}}\n'
    )

def notavail(analysis):
    return f'\\textit{{The {esc(analysis)} analysis was not performed for this family, or no data are available.}}\n\n'

def kv_table(items, caption, label):
    """Key-value summary table spanning the full text width (Parameter left,
    Value right-aligned at the margin) so it does not sit narrow and centred."""
    body = ''.join(f'  {esc(k)} & {fmtval(v)} \\\\\n' for k, v in items if v)
    return (
        '\\Needspace*{0.16\\textheight}\n'
        f'\\begin{{table}}[H]\n  \\centering\n  \\caption{{{esc(caption)}}}\n  \\label{{{label}}}\n'
        f'  \\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}l r}}\\toprule\n'
        f'  \\textbf{{Parameter}} & \\textbf{{Value}} \\\\\n'
        f'  \\midrule\n{body}  \\bottomrule\\end{{tabular*}}\n\\end{{table}}\n'
    )

# ── PDB / ColabFold rendering ─────────────────────────────────────────────────

PLDDT_COLORS = [(0, '#d13b27'), (0.5, '#f0c27f'), (0.7, '#65cbf3'), (0.9, '#003f97'), (1.0, '#003f97')]
PLDDT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'plddt', [(t, c) for t, c in PLDDT_COLORS])

def _plddt_color(v):
    if v >= 90: return '#003f97'
    elif v >= 70: return '#65cbf3'
    elif v >= 50: return '#f0c27f'
    else: return '#d13b27'

def render_colabfold(pdb_path, scores_json_path, pae_json_path, out_png, title=''):
    """
    Render a ColabFold structure to a 4-panel PNG:
      [3D front] [3D side] | [pLDDT profile] [PAE heatmap]
    pLDDT is read from the B-factor column (AlphaFold convention).
    """
    # Parse PDB
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('protein', str(pdb_path))
    ca_coords, plddt_vals = [], []
    for model in struct:
        for chain in model:
            for res in chain:
                if 'CA' in res:
                    at = res['CA']
                    ca_coords.append(at.coord)
                    plddt_vals.append(at.bfactor)
        break
    ca = np.array(ca_coords)
    plddt = np.array(plddt_vals)
    n_res = len(plddt)

    # Load PAE from scores JSON (has both plddt & pae keys)
    pae = None
    if scores_json_path and Path(scores_json_path).exists():
        try:
            d = json.loads(Path(scores_json_path).read_text())
            raw = d.get('pae', None)
            if raw: pae = np.array(raw)
        except Exception: pass
    if pae is None and pae_json_path and Path(pae_json_path).exists():
        try:
            d = json.loads(Path(pae_json_path).read_text())
            raw = d.get('predicted_aligned_error', None)
            if raw: pae = np.array(raw)
        except Exception: pass

    norm = mcolors.Normalize(vmin=0, vmax=100)
    n_panels = 4 if pae is not None and pae.size > 0 else 3
    fig = plt.figure(figsize=(14, 7))
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    # ── 3D structure panels ──────────────────────────────────────────────────
    for pi, (elev, azim) in enumerate([(25, 30), (25, 120)]):
        if n_panels == 4:
            ax = fig.add_subplot(2, 2, pi + 1, projection='3d')
        else:
            ax = fig.add_subplot(1, 3, pi + 1, projection='3d')

        # Centre coordinates
        centre = ca.mean(axis=0)
        coords = ca - centre

        # Draw backbone segments colored by local pLDDT
        for i in range(n_res - 1):
            mid = (plddt[i] + plddt[i + 1]) / 2
            col = _plddt_color(mid)
            ax.plot(coords[i:i+2, 0], coords[i:i+2, 1], coords[i:i+2, 2],
                    color=col, linewidth=2.0, solid_capstyle='round')
        # Scatter for residue points
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        c=plddt, cmap=PLDDT_CMAP, norm=norm,
                        s=12, zorder=5, depthshade=True)

        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([1, 1, 1])
        for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
            spine.set_ticklabels([]); spine._axinfo['grid']['color'] = (0, 0, 0, 0.05)
        ax.set_title(['Front view', 'Side view'][pi], fontsize=10, pad=4)

    # ── pLDDT per-residue bar chart ──────────────────────────────────────────
    if n_panels == 4:
        ax3 = fig.add_subplot(2, 2, 3)
    else:
        ax3 = fig.add_subplot(1, 3, 3)

    bar_cols = [_plddt_color(v) for v in plddt]
    ax3.bar(np.arange(1, n_res + 1), plddt, color=bar_cols, width=1.0, linewidth=0)
    for thresh, color in [(90, '#003f97'), (70, '#65cbf3'), (50, '#f0c27f')]:
        ax3.axhline(y=thresh, color=color, linestyle='--', linewidth=0.8, alpha=0.7)
    ax3.set_xlim(0.5, n_res + 0.5); ax3.set_ylim(0, 100)
    ax3.set_xlabel('Residue', fontsize=9); ax3.set_ylabel('pLDDT', fontsize=9)
    ax3.set_title('Per-residue pLDDT', fontsize=10)
    ax3.spines[['top', 'right']].set_visible(False)
    mean_plddt = plddt.mean()
    ax3.axhline(y=mean_plddt, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
    ax3.text(n_res * 0.98, mean_plddt + 1, f'mean={mean_plddt:.1f}',
             ha='right', va='bottom', fontsize=8, color='black')
    from matplotlib.patches import Patch
    ax3.legend(handles=[
        Patch(facecolor='#003f97', label='Very high (>90)'),
        Patch(facecolor='#65cbf3', label='Confident (70–90)'),
        Patch(facecolor='#f0c27f', label='Low (50–70)'),
        Patch(facecolor='#d13b27', label='Very low (<50)'),
    ], fontsize=7, loc='lower right', framealpha=0.8)

    # ── PAE heatmap ──────────────────────────────────────────────────────────
    if pae is not None and pae.size > 0:
        ax4 = fig.add_subplot(2, 2, 4)
        im = ax4.imshow(pae, cmap='Greens_r', vmin=0, vmax=30, aspect='auto', origin='upper')
        cbar = plt.colorbar(im, ax=ax4, shrink=0.85)
        cbar.set_label(r'PAE ($\AA$)', fontsize=9)
        ax4.set_xlabel('Scored residue', fontsize=9)
        ax4.set_ylabel('Aligned residue', fontsize=9)
        ax4.set_title('Predicted Aligned Error', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95] if title else [0, 0, 1, 1])
    plt.savefig(str(out_png), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Rendered: {out_png.name}')

def render_cluster_scatter(d):
    """Render static UMAP / t-SNE / PCA scatter plots coloured by HDBSCAN
    cluster from 03_clustering/clustering_coordinates.csv. The pipeline only
    saves interactive HTML, so the PDF previously had no embedded projection.
    Returns an ordered list of (projection_name, png_path)."""
    src = Path(d) / '03_clustering' / 'clustering_coordinates.csv'
    if not src.exists():
        return []
    with open(src, newline='', errors='replace') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    # Subsample for very large datasets to keep scatter plots readable
    MAX_SCATTER = 30000
    if len(rows) > MAX_SCATTER:
        step = len(rows) // MAX_SCATTER
        rows = rows[::step]
    clusters = sorted({r.get('Cluster', '') for r in rows},
                      key=lambda c: (str(c) == '-1', _safe_float_key(c)))
    palette = plt.get_cmap('tab10')
    colour, ci = {}, 0
    for c in clusters:
        if str(c) == '-1':
            colour[c] = (0.62, 0.62, 0.62)
        else:
            colour[c] = palette(ci % 10); ci += 1
    out = []
    for xk, yk, name in [('umap_x', 'umap_y', 'UMAP'),
                         ('tsne_x', 'tsne_y', 't-SNE'),
                         ('pca_x', 'pca_y', 'PCA')]:
        if xk not in rows[0]:
            continue
        try:
            series = {c: ([], []) for c in clusters}
            for r in rows:
                series[r.get('Cluster', '')][0].append(float(r[xk]))
                series[r.get('Cluster', '')][1].append(float(r[yk]))
        except (ValueError, KeyError):
            continue
        fig, ax = plt.subplots(figsize=(6.2, 5.0))
        for c in clusters:
            xs, ys = series[c]
            lbl = 'noise ($-1$)' if str(c) == '-1' else f'cluster {c}'
            ax.scatter(xs, ys, s=9, color=colour[c], label=lbl, alpha=0.75, linewidths=0)
        ax.set_xlabel(f'{name}\\,1'.replace('\\,', ' '), fontsize=10)
        ax.set_ylabel(f'{name} 2', fontsize=10)
        ax.set_title(f'{Path(d).name} — {name} projection (HDBSCAN clusters)', fontsize=11)
        ax.legend(markerscale=2, fontsize=8, frameon=False, loc='best')
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        p = Path(d) / '03_clustering' / f"fig_{name.lower().replace('-', '')}.png"
        fig.savefig(str(p), dpi=140, facecolor='white')
        plt.close(fig)
        out.append((name, p))
    return out

def _safe_float_key(c):
    try:
        return float(c)
    except (ValueError, TypeError):
        return 0.0

def render_all_pdbs(d, out_dir):
    """Find all ColabFold PDB files in d, render them, return list of (name, png_path, plddt_mean, max_pae, n_res, ptm)."""
    cf_dir = Path(d)
    if not cf_dir.is_dir(): return []
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    pdbs = sorted(cf_dir.glob('*_unrelaxed_rank_001*.pdb'))
    if not pdbs:
        pdbs = sorted(cf_dir.glob('*.pdb'))
    for pdb in pdbs:
        stem = pdb.stem.replace('_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000', '')
        scores_json = cf_dir / (stem + '_scores_rank_001_alphafold2_ptm_model_1_seed_000.json')
        pae_json = cf_dir / (stem + '_predicted_aligned_error_v1.json')
        out_png = out_dir / (stem + '_structure.png')

        # Read scores
        plddt_mean = max_pae = n_res = ptm = None
        if scores_json.exists():
            try:
                sc = json.loads(scores_json.read_text())
                plddt_arr = sc.get('plddt', [])
                if plddt_arr:
                    plddt_mean = np.mean(plddt_arr)
                    n_res = len(plddt_arr)
                max_pae = sc.get('max_pae', None)
                ptm = sc.get('ptm', None)
            except Exception: pass

        # Render
        title = esc(stem.replace('_', ' '))
        try:
            render_colabfold(pdb, scores_json if scores_json.exists() else None,
                             pae_json if pae_json.exists() else None,
                             out_png, title=title)
        except Exception as e:
            print(f'  WARNING: PDB render failed for {pdb.name}: {e}')
            out_png = None

        results.append({
            'name': stem,
            'pdb': pdb,
            'png': out_png if out_png and out_png.exists() else None,
            'plddt_mean': f'{plddt_mean:.1f}' if plddt_mean is not None else 'n/a',
            'max_pae': f'{max_pae:.1f}' if max_pae is not None else 'n/a',
            'n_res': str(n_res) if n_res is not None else 'n/a',
            'ptm': f'{ptm:.3f}' if ptm is not None else 'n/a',
        })
    return results

# ── Chapter builders ──────────────────────────────────────────────────────────

def ch_summary(d, stats, cl_rows, family_display):
    """Executive summary chapter."""
    reports = d / 'reports'
    phylo = parse_report(read_txt(reports / 'phylo_report.txt'))
    trans = parse_report(read_txt(reports / 'transduction_report.txt'))
    ctcf  = parse_report(read_txt(reports / 'ctcf_tad_report.txt'))
    anti  = parse_report(read_txt(reports / 'antisense_report.txt'))
    grna  = parse_report(read_txt(reports / 'grna_offtarget_report.txt'))
    fold  = parse_report(read_txt(reports / 'fold_report.txt'))

    fn = family_display.lower()
    raw_class = detect_te_class(d)
    if raw_class:
        rc = raw_class.lower()
        if 'ltr' in rc and ('ervk' in rc or 'iap' in rc): te_class = 'LTR retrotransposon (ERVK superfamily)'
        elif 'ltr' in rc and ('ervl' in rc or 'maln' in rc or 'mal' in rc): te_class = 'LTR retrotransposon (ERVL-MaLR superfamily)'
        elif 'line' in rc or 'l1' in rc: te_class = 'LINE retrotransposon (L1 superfamily)'
        elif 'sine' in rc and 'alu' in rc: te_class = 'SINE (Alu superfamily)'
        elif 'sine' in rc: te_class = 'SINE'
        else: te_class = raw_class
    else:
        if any(x in fn for x in ['iap', 'ervk', 'ltr1']): te_class = 'LTR retrotransposon (ERVK superfamily)'
        elif any(x in fn for x in ['mt2', 'mal', 'ervl']): te_class = 'LTR retrotransposon (ERVL-MaLR superfamily)'
        elif 'l1md' in fn or 'line' in fn: te_class = 'LINE retrotransposon (L1 superfamily)'
        elif 'alu' in fn or 'b1' in fn or 'b2' in fn or 'sine' in fn: te_class = 'SINE (Alu superfamily)' if 'alu' in fn else 'SINE'
        else: te_class = 'transposable element'

    assembly, organism_latex = detect_assembly(d)

    n_clusters = len(cl_rows)
    total_copies = stats.get('n_copies', '?')
    median_div = phylo.get('Median divergence', 'n/a')
    est_age = phylo.get('Median est. age', 'n/a')
    n_lineages = trans.get('Transduction lineages', 'n/a')
    ctcf_n = ctcf.get('CTCF core motif', 'n/a')
    bidir = anti.get('Bidirectional (>=0.4)', 'n/a')
    best_guide = grna.get('Guide', '')
    guide_cov = grna.get('Coverage', '')
    best_plddt = fold.get('Mean pLDDT', '')
    intact = phylo.get('Intact (ORF≥thr)', phylo.get('Intact (ORF>=thr)', 'n/a'))

    tex = r'\chapter{Executive Summary}' + '\n\n'
    tex += (
        f'This report presents a comprehensive GAMECA (Genome-wide Analysis of Mobile Element Clusters and Activity) '
        f'analysis of the \\textit{{{esc(family_display)}}} family, a {te_class} annotated in the {organism_latex} '
        f'{esc(assembly)} reference genome. A total of {esc(total_copies)} genomic copies were retrieved from RepeatMasker annotations '
        f'and subjected to multi-stage computational analysis spanning sequence clustering, multiple sequence alignment, '
        f'transcription factor binding site prediction, structural annotation, evolutionary age estimation, '
        f'and functional genomics overlays.\n\n'
    )

    tex += r'\section*{Key Findings}' + '\n\\begin{itemize}\n'
    tex += f'  \\item \\textbf{{Copy number and clustering:}} {esc(total_copies)} genomic copies partitioned into {n_clusters} cluster(s) by HDBSCAN on a UMAP embedding of pairwise sequence distances.\n'
    if median_div != 'n/a':
        tex += f'  \\item \\textbf{{Evolutionary age:}} Median K80 divergence from the RepBase consensus is {esc(median_div)}, corresponding to an estimated median insertion age of {esc(est_age)} (molecular clock rate 4.5\\,$\\times$\\,10\\textsuperscript{{$-9$}} substitutions/site/year).\n'
    if n_lineages != 'n/a':
        tex += f"  \\item \\textbf{{3' transduction:}} {esc(n_lineages)} transduction lineage(s) identified, indicating retrotranspositionally active source element(s) that co-mobilised flanking unique sequence.\n"
    if ctcf_n not in ('n/a', '0', ''):
        tex += f'  \\item \\textbf{{CTCF binding:}} {esc(ctcf_n)} copies harbour a CTCF core-binding motif (MA0139.1), implicating this family in chromatin loop anchoring and TAD boundary formation.\n'
    if bidir not in ('n/a', '0', ''):
        tex += f'  \\item \\textbf{{Bidirectional promoter activity:}} {esc(bidir)} copies (bidirectionality score\\,$\\geq$\\,0.4) are predicted to act as bidirectional promoters, potentially driving antisense transcription into adjacent host genes.\n'
    if intact not in ('n/a', '0', ''):
        tex += f'  \\item \\textbf{{Intact elements:}} {esc(intact)} copies retain a near-complete open reading frame and represent candidates for ongoing or recent transpositional activity.\n'
    if best_guide:
        tex += f'  \\item \\textbf{{gRNA design:}} Best SpCas9 guide \\texttt{{{esc(best_guide)}}} achieves coverage {esc(guide_cov)} with minimal off-target activity.\n'
    if best_plddt:
        tex += f'  \\item \\textbf{{Protein fold prediction:}} ColabFold (AlphaFold2) structural modelling of consensus ORFs yielded a mean pLDDT of {esc(best_plddt)} (best model), consistent with predominantly disordered or low-confidence regions typical of TE-encoded proteins.\n'
    tex += '\\end{itemize}\n\n'
    return tex

def ch_overview(d, stats, cl_rows, seq_figs=None):
    h_cl, rows_cl = read_csv(d / '03_clustering' / 'cluster_summary.csv')
    h_per, _ = [], []
    per_cluster_stats = []
    per_dir = d / '02_statistics' / 'per_cluster'
    if per_dir.exists():
        for f in sorted(per_dir.glob('cluster_*_statistics.txt')):
            s = parse_stats(read_txt(f))
            m = re.search(r'cluster_(-?\d+)', f.stem)
            s['cluster'] = m.group(1) if m else '?'
            per_cluster_stats.append(s)

    assembly, _ = detect_assembly(d)

    tex = r'\chapter{Dataset Overview and Statistics}' + '\n\n'
    tex += (
        r'\section{Genomic Copy Statistics}' + '\n'
        f'The \\textit{{{esc(stats.get("family","?"))}}} dataset comprises {esc(stats.get("n_copies","?"))} genomic copies '
        f'recovered from the {esc(assembly)} RepeatMasker annotation. Sequence length is '
        f'{esc(stats.get("mean_len","?"))}\\,bp on average '
        f'(median {esc(stats.get("median_len","?"))}\\,bp, SD {esc(stats.get("std_len","?"))}\\,bp); '
        f'the distribution is right-skewed by a small number of full-length insertions, so the mean, '
        f'median and SD summarise the family better than the raw min--max span, which is dominated by outliers. '
        f'Mean GC content is {esc(stats.get("gc_mean","?"))}\\%. '
        f'Per-copy expression across {esc(stats.get("expr_cols","?"))} user-designated samples '
        f'(named and ordered before this run) is analysed in the Gene Expression chapter.\n\n'
    )

    if seq_figs and seq_figs.get('seq_stats_panel'):
        tex += fig_latex(seq_figs['seq_stats_panel'],
                         'Sequence statistics: (top-left) length distribution, '
                         '(top-right) GC content, '
                         '(bottom-left) RepeatMasker divergence score (milliDiv), '
                         '(bottom-right) Smith-Waterman alignment score.',
                         'fig:seq_stats', width=r'0.92\linewidth',
                         short='Sequence statistics panel.')

    tex += r'\section{Cluster Composition}' + '\n'
    tex += (
        'HDBSCAN clustering on the UMAP low-dimensional embedding partitioned the copies as shown in '
        'Table~\\ref{tab:cluster_summary}. Cluster~$-1$ contains sequences that could not be assigned to '
        'any dense group (noise points).\n\n'
    )
    if rows_cl:
        tex += longtable(h_cl, rows_cl, 'Cluster composition: strand distribution of genomic copies.',
                         'tab:cluster_summary', note='n+ / n-: copies on + / - strand.')

    if seq_figs and seq_figs.get('chrom_distribution'):
        tex += r'\section{Chromosomal Distribution}' + '\n'
        tex += (
            'Figure~\\ref{fig:chrom_dist} shows the number of TE copies per chromosome, '
            'reflecting the genome-wide insertion pattern of this family.\n\n'
        )
        tex += fig_latex(seq_figs['chrom_distribution'],
                         'Chromosomal distribution of genomic copies.',
                         'fig:chrom_dist', width=r'\linewidth',
                         short='Chromosomal distribution of copies.')

    if per_cluster_stats:
        tex += r'\section{Per-cluster Sequence Statistics}' + '\n'
        tex += 'Sequence length and GC content statistics per cluster are presented in Table~\\ref{tab:per_cluster_stats}.\n\n'
        hdr = ['cluster', 'n_copies', 'mean_len', 'median_len', 'std_len', 'range_len', 'gc_mean']
        pcs_rows = [{'cluster': s.get('cluster','?'), 'n_copies': s.get('n_copies',''),
                     'mean_len': s.get('mean_len',''), 'median_len': s.get('median_len',''),
                     'std_len': s.get('std_len',''), 'range_len': s.get('range_len',''),
                     'gc_mean': s.get('gc_mean','')} for s in per_cluster_stats]
        tex += longtable(hdr, pcs_rows, 'Per-cluster sequence length and GC content statistics.',
                         'tab:per_cluster_stats',
                         note='gc_mean: mean GC content (%). range_len: [min, max] in bp.')
    return tex

def ch_clustering(d):
    tex = r'\chapter{Sequence Clustering}' + '\n\n'
    tex += (
        r'\section{UMAP and HDBSCAN}' + '\n'
        'Pairwise sequence distances were computed with a $k$-mer frequency (MASH-distance-inspired) approach '
        'and projected to two dimensions using Uniform Manifold Approximation and Projection (UMAP; '
        r'\textit{McInnes et al.}, 2018). Density-based clustering was then performed with HDBSCAN '
        '(minimum cluster size 15), which identifies arbitrary-shape clusters and designates low-density '
        'copies as noise (cluster $-1$). Interactive UMAP visualisations are available in '
        '\\texttt{03\\_clustering/clustering\\_visualization.html}.\n\n'
    )
    # Static projection scatter plots (rendered from clustering_coordinates.csv)
    scatters = render_cluster_scatter(d)
    if scatters:
        sd = dict(scatters)
        umap = sd.get('UMAP')
        if umap:
            tex += fig_latex(umap,
                             'UMAP projection of pairwise k-mer sequence distances, coloured by '
                             'HDBSCAN cluster. Each point is one genomic copy; well-separated point '
                             'clouds correspond to sequence-distinct subtypes, and grey noise points '
                             '(cluster -1) lie in low-density regions.',
                             'fig:umap', width=r'0.72\linewidth',
                             short='UMAP projection coloured by HDBSCAN cluster.')
        tsne, pca = sd.get('t-SNE'), sd.get('PCA')
        if tsne or pca:
            tex += twofigs(tsne, 't-SNE projection (HDBSCAN clusters).', 'fig:tsne',
                           pca, 'PCA projection (first two components).', 'fig:pca')
    tex += r'\section{Multiple Sequence Alignment (CIAlign)}' + '\n'
    tex += (
        'All copies were globally aligned with MAFFT (\\texttt{--auto} mode) and subsequently cleaned with '
        'CIAlign, which removes gap-heavy columns, insertion-rich sequences, and poorly aligning outliers '
        'to produce a representative cleaned alignment for consensus derivation and divergence estimation. '
        'For very large alignments, CIAlign\\textquotesingle{}s cleaning and plotting pass can fail outright '
        'or exceed a 1-hour timeout; when this happens the pipeline falls back to cleaning the alignment with '
        'trimAl (\\texttt{-automated1}) and retries CIAlign once on the trimAl-cleaned alignment, capped at '
        '1{,}000 displayed sequences, rather than omitting the alignment visualization entirely. '
        'Figure~\\ref{fig:align_global} shows the global alignment before and after CIAlign processing.\n\n'
    )
    align_stats = read_txt(d / '04_alignments' / 'alignment_stats.txt')
    if align_stats:
        kv = {}
        for line in align_stats.splitlines():
            if ':' in line:
                k, _, v = line.partition(':'); kv[k.strip()] = v.strip()
        if kv:
            tex += kv_table(list(kv.items()), 'Global alignment statistics.', 'tab:align_stats')

    p_in = get_fig(d / 'cialign_plots', 'global_alignment_input')
    p_out = get_fig(d / 'cialign_plots', 'global_alignment_output')
    tex += twofigs(p_in, 'Raw global alignment', 'fig:align_in',
                   p_out, 'Cleaned global alignment (CIAlign)', 'fig:align_global')

    # Per-cluster alignments
    found_cluster_figs = []
    for stem_prefix in ['cluster_-1_alignment', 'cluster_0_alignment',
                         'cluster_1_alignment', 'cluster_2_alignment']:
        p = get_fig(d / 'cialign_plots', f'{stem_prefix}_output')
        if p:
            label = stem_prefix.replace('_alignment', '').replace('cluster_', 'Cluster ')
            found_cluster_figs.append((p, label))

    if found_cluster_figs:
        tex += r'\section{Per-cluster Alignments}' + '\n'
        tex += 'Cleaned per-cluster alignments are shown below.\n\n'
        for i in range(0, len(found_cluster_figs), 2):
            if i + 1 < len(found_cluster_figs):
                p1, l1 = found_cluster_figs[i]; p2, l2 = found_cluster_figs[i+1]
                tex += twofigs(p1, f'Cleaned alignment, {l1}', f'fig:calign_{i}',
                               p2, f'Cleaned alignment, {l2}', f'fig:calign_{i+1}')
            else:
                p1, l1 = found_cluster_figs[i]
                tex += fig_latex(p1, f'Cleaned alignment, {l1}', f'fig:calign_{i}')
    return tex

def ch_consensus(d):
    tex = r'\chapter{Consensus Sequences}' + '\n\n'
    tex += (
        'Cluster consensus sequences were derived from the CIAlign-cleaned MAFFT alignments '
        'by taking the modal nucleotide at each column position (gap characters excluded). '
        'The resulting consensus sequences serve as reference templates for LTR structure annotation, '
        'open reading frame (ORF) prediction, and primer specificity scoring. '
        'Table~\\ref{tab:consensus_summary} summarises the length and composition of each cluster consensus.\n\n'
    )
    h, rows = read_csv(d / '05_consensus' / 'cluster_consensus_summary.csv')
    if not rows:
        h, rows = read_csv(d / 'cluster_alignments' / 'cluster_consensus_summary.csv')
    if rows:
        keep = ['cluster', 'n_sequences', 'consensus_length', 'avg_gaps_per_col',
                'cols_majority_gap']
        tex += longtable(h, rows, 'Cluster consensus sequence summary.', 'tab:consensus_summary',
                         cols=[c for c in keep if c in h],
                         note='Length: consensus length (bp); avg. gaps/col: mean gap count per '
                              'alignment column; gap-major cols: columns where gaps are the majority '
                              'character.')
    else:
        tex += notavail('consensus summary')
    return tex

def ch_motif(d, stats=None):
    stats = stats or {}
    tex = r'\chapter{Transcription Factor Binding Site Analysis}' + '\n\n'
    tex += (
        r'\section{JASPAR 2022 Motif Intersection}' + '\n'
        'JASPAR 2022 vertebrate core position frequency matrices (PFMs; Fornes \\textit{et al.}, 2020) '
        'were scanned against the genome using MOODS (Korhonen \\textit{et al.}, 2017) and intersected '
        'with TE loci via tabix-indexed BED files. Only significant hits ($p < 10^{-4}$) within TE '
        'boundaries were retained. The top 25 overlapping JASPAR motifs genome-wide are listed in '
        'Table~\\ref{tab:top_motifs}.\n\n'
    )
    h, rows = read_csv(d / 'motif_analysis' / 'overall_motif_counts.csv', max_rows=25)
    if rows:
        tex += longtable(h, rows, 'Top 25 JASPAR motifs overlapping TE loci (ranked by site count).',
                         'tab:top_motifs')
    p = get_fig(d / 'motif_analysis', 'overall_top_motifs')
    if p: tex += fig_latex(p, 'Top JASPAR motifs overlapping TE loci.', 'fig:top_motifs')

    # Explain why site counts can exceed the number of copies, and give the
    # per-motif breakdown that separates sites from insertions.
    tex += (
        '\\paragraph{Interpreting the counts.} These values count \\emph{binding sites}, not '
        'insertions. A single TE copy frequently contains several occurrences of the same motif, '
        'so the site count for a motif — and the total in Figure~\\ref{fig:top_motifs} — can and '
        'usually does exceed the number of ' + esc(stats.get('family', '?')) + ' loci '
        '(' + esc(stats.get('n_copies', '?')) + '). To make this explicit, '
        'Table~\\ref{tab:motif_multiplicity} reports, for each motif, the total number of sites '
        '(\\texttt{n\\_sites}), the number of distinct insertions carrying at least one site '
        '(\\texttt{n\\_loci}, which never exceeds the copy count), the number of insertions carrying '
        '\\emph{more than one} site (\\texttt{n\\_loci\\_multisite}), and the mean sites per occupied '
        'insertion. The JASPAR matrix identifier is shown alongside each TF name where the source '
        'annotation provides it.\n\n'
    )
    hm, rowsm = read_csv(d / '05_tfbs' / 'motif_site_multiplicity.csv', max_rows=25)
    if rowsm:
        tex += longtable(hm, rowsm,
                         'Per-motif site multiplicity: sites vs.\\ occupied insertions vs.\\ '
                         'insertions with more than one site (top 25 by site count).',
                         'tab:motif_multiplicity',
                         note='n\\_sites: total binding sites. n\\_loci: distinct insertions with '
                              '$\\geq$1 site. n\\_loci\\_multisite: insertions with $>$1 site for that motif.')

    tex += r'\section{Per-cluster TF Binding Site Counts}' + '\n'
    tex += (
        'Binding site counts per cluster were aggregated to identify differential TFBS enrichment '
        'across sequence subtypes. The top 20 rows of the per-cluster binding count matrix are shown in '
        'Table~\\ref{tab:tfbs_counts}. The full interactive heatmap is available in '
        '\\texttt{05\\_tfbs/cluster\\_tf\\_binding\\_heatmap.html}. As in the previous section, each '
        'cell counts binding \\emph{sites}, so a cluster\'s value for a motif can exceed the number '
        'of copies in that cluster whenever insertions carry multiple sites of the same motif; see '
        'Table~\\ref{tab:motif_multiplicity} for the sites-per-insertion breakdown.\n\n'
    )
    h2, rows2 = read_csv(d / '05_tfbs' / 'cluster_tf_binding_counts.csv', max_rows=20)
    if rows2:
        tex += longtable(h2, rows2, 'Per-cluster JASPAR TF binding site counts (top 20 motifs).',
                         'tab:tfbs_counts', note='Full matrix available in 05\_tfbs/.')

    enrich_png = get_fig(d / 'enrichment_results', 'enrichment_heatmap')
    if enrich_png:
        tex += r'\section{Cluster Enrichment Analysis}' + '\n'
        tex += (
            "Motif enrichment within each cluster relative to the full copy set was assessed by "
            "Fisher's exact test. The enrichment heatmap (Figure~\\ref{fig:enrich}) shows $-\\log_{10}$ "
            "$p$-values for significantly enriched motifs.\n\n"
        )
        tex += fig_latex(enrich_png, "JASPAR motif enrichment heatmap (Fisher's exact test, per cluster).",
                         'fig:enrich')
    return tex

def ch_motif_gains(d):
    reports = d / 'reports'
    h, rows = read_csv(reports / 'motif_gain_summary.csv', max_rows=25)
    tex = r'\chapter{Motif Gain Analysis}' + '\n\n'
    if not rows:
        return tex + notavail('motif gain')
    tex += (
        'Motif gain analysis identifies JASPAR TF binding sites that are significantly over-represented '
        'in TE copies relative to a dinucleotide-shuffled background, indicating that the TE sequence '
        'itself encodes regulatory information for those TFs. The fraction of binding sites classified '
        'as "gained" (absent in shuffled controls) is reported as \\textit{pct\\_gained}. '
        'The top 25 gained motifs are listed in Table~\\ref{tab:motif_gains}.\n\n'
        'Here too, \\texttt{n\\_gained} and \\texttt{n\\_total} count individual binding \\emph{sites}, '
        'not insertions: because a single copy can contribute several sites of the same motif, both '
        'values may exceed the number of family loci. The \\texttt{copies\\_with\\_gain} column gives '
        'the complementary insertion-level figure — the number of distinct copies with at least one '
        'gained site — which never exceeds the copy count.\n\n'
    )
    tex += longtable(h, rows, 'Top 25 gained JASPAR TF binding motifs (ranked by $N$ gained).',
                     'tab:motif_gains',
                     cols=['motif_name','n_gained','n_total','pct_gained',
                           'copies_with_gain','pct_copies','enriched_clusters'])
    p1 = get_fig(reports, 'fig_motif_gains_bar')
    p2 = get_fig(reports, 'fig_motif_gains_heatmap')
    tex += twofigs(p1, 'Motif gains bar chart (top motifs).', 'fig:gains_bar',
                   p2, 'Motif gains heatmap (per-cluster distribution).', 'fig:gains_heat')
    return tex

def ch_go(d):
    ga = d / 'go_annotations'
    tex = r'\chapter{Gene Ontology Enrichment}' + '\n\n'
    skipped = read_txt(ga / 'GO_SKIPPED.txt')
    if skipped or not ga.exists() or not any(ga.glob('*.csv')):
        return tex + notavail('GO Biological Process enrichment')
    tex += (
        'Protein-coding genes located within 50\\,kb of each cluster\'s TE insertion loci were '
        'retrieved from Ensembl mm10 gene annotations and submitted to Gene Ontology (GO) '
        'Biological Process overrepresentation analysis. The top enriched terms are listed in '
        'Table~\\ref{tab:go_terms}.\n\n'
    )
    h, rows = read_csv(ga / 'go_bp_top_terms.csv', max_rows=20)
    if rows:
        tex += longtable(h, rows, 'Top 20 GO Biological Process terms enriched near TE loci.',
                         'tab:go_terms')
    pngs = sorted(ga.glob('*.png'))[:4]
    labels = [p.stem.replace('_', ':').replace('go:', 'GO:') for p in pngs]
    for i in range(0, len(pngs), 2):
        if i + 1 < len(pngs):
            tex += twofigs(pngs[i], f'GO BP enrichment: {labels[i]}.', f'fig:go_{pngs[i].stem}',
                           pngs[i+1], f'GO BP enrichment: {labels[i+1]}.', f'fig:go_{pngs[i+1].stem}')
        else:
            tex += fig_latex(pngs[i], f'GO Biological Process enrichment: {labels[i]}.',
                             f'fig:go_{pngs[i].stem}', width=r'0.7\linewidth')
    return tex

def ch_expression(d):
    ed = d / 'expression_plots'
    tex = r'\chapter{Gene Expression Analysis}' + '\n\n'
    stats_csv = ed / 'expression_stats.csv'
    h, rows = read_csv(stats_csv, max_rows=30) if stats_csv.exists() else ([], [])
    box  = get_fig(ed, 'boxplot_all')
    prof = get_fig(ed, 'stage_profile')
    peak = get_fig(ed, 'peak_stage_summary')
    chrm = get_fig(ed, 'chromosomal_expression')
    if not rows and not any([box, prof, peak, chrm]):
        return tex + notavail('gene expression')

    tex += (
        'Per-copy expression values supplied with the input were summarised per cluster and per '
        'sample/stage, using the expression columns and figure order designated by the user '
        'before this run (not auto-detected). Unlike a single family-wide mean, this exposes '
        'whether particular sequence subtypes (clusters) drive the family\'s expression and in '
        'which sample or developmental stage that expression peaks. All values below are computed '
        'directly from the designated expression columns; no values are imputed.\n\n'
    )
    if box:
        tex += fig_latex(box, 'Per-cluster expression distribution across all samples '
                              '(box = interquartile range, whiskers = full range).', 'fig:expr_box')
    if prof:
        tex += fig_latex(prof, 'Mean expression per sample/stage (bars, all copies) with '
                               'per-cluster profiles overlaid (lines).', 'fig:expr_profile')
    if peak or chrm:
        tex += twofigs(peak, 'Peak-expression stage per cluster.', 'fig:expr_peak',
                       chrm, 'Mean expression per chromosome.', 'fig:expr_chrom')
    if rows:
        tex += r'\section{Per-cluster Expression Statistics}' + '\n'
        tex += ('Table~\\ref{tab:expr_stats} reports, for each cluster and expression column, the '
                'mean, median, standard deviation, coefficient of variation, and the fraction of '
                'zero-expression copies.\n\n')
        tex += longtable(h, rows, 'Per-cluster expression statistics (first 30 rows).',
                         'tab:expr_stats',
                         cols=['Cluster', 'Label', 'n', 'mean', 'median', 'std', 'cv', 'pct_zero'],
                         note='cv: coefficient of variation (std/mean). pct\\_zero: percent of copies '
                              'with zero expression in that column.')
    return tex

def ch_ltr_struct(d):
    reports = d / 'reports'
    tex = r'\chapter{LTR Structure Annotation}' + '\n\n'
    h, rows = read_csv(reports / 'ltr_struct_summary.csv')
    if not rows:
        return tex + (
            r'\textit{LTR structure annotation was not performed, or this family lacks LTR elements '
            '(e.g., LINE or SINE families).}\n\n'
        )
    tex += (
        'LTR retrotransposons are flanked by Long Terminal Repeats that are nearly identical upon '
        'insertion and diverge at the neutral substitution rate thereafter. Each copy was annotated for: '
        '(i) presence of 5\\textquotesingle\\ and 3\\textquotesingle\\ LTRs and their sequence identity; '
        '(ii) primer binding site (PBS) immediately downstream of the 5\\textquotesingle\\ LTR; '
        '(iii) polypurine tract (PPT) upstream of the 3\\textquotesingle\\ LTR; and '
        '(iv) target site duplications (TSDs). Solo-LTRs arise by homologous recombination between the '
        'two flanking LTRs, excising the internal sequence. '
        'Structural statistics per cluster are presented in Table~\\ref{tab:ltr_struct}.\n\n'
    )
    tex += longtable(h, rows, 'LTR structure annotation summary per cluster.', 'tab:ltr_struct',
                     note='PBS: primer binding site; PPT: polypurine tract; LTR identity: mean pairwise identity between 5\' and 3\' LTR.')
    p = get_fig(reports, 'fig_ltr_struct')
    if p: tex += fig_latex(p, 'LTR structure annotation summary.', 'fig:ltr_struct')

    h2, rows2 = read_csv(reports / 'ltr_struct_annotated.csv', max_rows=20)
    if rows2:
        safe_cols = ['Chromosome', 'Start', 'Stop', 'strand', 'Cluster',
                     'classification', 'ltr_identity', 'pbs_found', 'ppt_found']
        tex += r'\FloatBarrier' + '\n'
        tex += longtable(h2, rows2, 'LTR structure annotation per locus (first 20 entries).',
                         'tab:ltr_annotated', cols=[c for c in safe_cols if c in h2],
                         note='First 20 rows shown; full data in reports/ltr\_struct\_annotated.csv.')
    return tex

def ch_divergence(d):
    reports = d / 'reports'
    tex = r'\chapter{Sequence Divergence and Repeat Landscape}' + '\n\n'
    tex += (
        'Sequence divergence from the RepBase consensus was estimated using the Kimura two-parameter '
        "(K80) substitution model applied to each copy's CIAlign-cleaned alignment. "
        'Under a neutral molecular clock ($\\mu = 4.5 \\times 10^{-9}$ substitutions/site/year for '
        'the Mus lineage), insertion age $T$ is estimated as $T = d_{\\text{K80}} / (2\\mu)$, '
        'where $d_{\\text{K80}}$ is the K80 distance between the copy and the consensus sequence. '
        'The repeat landscape histogram displays copy-number density as a function of divergence, '
        'capturing historical bursts of transpositional activity.\n\n'
    )
    h, rows = read_csv(reports / 'divergence_stats.csv')
    if rows:
        tex += longtable(h, rows, 'K80 divergence statistics per cluster.', 'tab:div_stats',
                         note='div\_pct: percentage divergence from RepBase consensus.')

    p1 = get_fig(reports, 'fig_repeat_landscape')
    p2 = get_fig(reports, 'fig_phylo_divergence')
    tex += twofigs(p1, 'Repeat landscape: copy count per divergence bin.', 'fig:landscape',
                   p2, 'Per-copy K80 divergence distribution.', 'fig:div_dist')
    p3 = get_fig(reports, 'fig_repeat_landscape_per_cluster')
    if p3: tex += fig_latex(p3, 'Repeat landscape stratified by cluster.', 'fig:landscape_cluster',
                            width=r'0.75\linewidth')
    return tex

def ch_phylo(d):
    reports = d / 'reports'
    txt = read_txt(reports / 'phylo_report.txt')
    tex = r'\chapter{Phylogenetics and Age Estimation}' + '\n\n'
    if not txt:
        return tex + notavail('phylogenetic age estimation')
    info = parse_report(txt)
    tex += (
        'Phylogenetic age was estimated by applying the neutral molecular clock to K80 divergence '
        'values. Intact elements (those retaining an ORF of sufficient length) are candidates for '
        'active retrotransposition and represent potential "master" or source elements. '
        'MAFFT \\texttt{{--addfragments}} was used for per-cluster phylogenetic alignment, with a '
        'saturation guard to exclude highly diverged copies that would compromise clock accuracy. '
        'Key phylogenetic parameters are summarised in Table~\\ref{tab:phylo_params}.\n\n'
    )
    items = [(k, v) for k, v in info.items() if k not in ('Family',)]
    if items:
        tex += kv_table(items, 'Phylogenetic and age estimation parameters.', 'tab:phylo_params')

    p1 = get_fig(reports, 'fig_phylo_master')
    p2 = get_fig(reports, 'fig_phylo_tree')
    if p1 or p2:
        tex += twofigs(p1, 'Master/source element phylogenetic context.', 'fig:phylo_master',
                       p2, 'Subfamily phylogenetic tree.', 'fig:phylo_tree')

    # Subfamily table
    h, rows = read_csv(reports / 'subfamily_table.csv', max_rows=30)
    if rows:
        tex += r'\section{Subfamily Analysis}' + '\n'
        tex += (
            'Subfamilies were defined by hierarchical clustering of pairwise K80 distances between '
            'cluster consensus sequences. The resulting dendrogram partitions copies into discrete '
            'evolutionary lineages (Table~\\ref{tab:subfamily}).\n\n'
        )
        tex += longtable(h, rows, 'Subfamily assignments and divergence statistics.', 'tab:subfamily')
        p_sdiv = get_fig(reports, 'fig_subfamily_divergence')
        p_stree = get_fig(reports, 'fig_subfamily_tree')
        tex += twofigs(p_sdiv, 'Subfamily divergence distribution.', 'fig:sub_div',
                       p_stree, 'Subfamily phylogenetic tree.', 'fig:sub_tree')
    return tex

def ch_transduction(d):
    reports = d / 'reports'
    txt = read_txt(reports / 'transduction_report.txt')
    tex = r"\chapter{3\textquotesingle\ Transduction}" + '\n\n'
    if not txt:
        return tex + notavail("3' transduction")
    info = parse_report(txt)
    tex += (
        "3\\textquotesingle\\ transduction occurs when a retrotransposon mobilises together with "
        "downstream unique flanking sequence, creating a diagnostic genomic signature: two or more "
        "insertions sharing an identical unique 3\\textquotesingle\\ sequence. Such copies form a "
        "\\textit{transduction lineage} traceable back to a common source element. Poly-A signals "
        "and poly-A tail lengths were annotated on each copy to identify full-length retrotransposition "
        "events (Table~\\ref{tab:transduction}).\n\n"
    )
    items = [(k, v) for k, v in info.items()]
    if items:
        tex += kv_table(items, "3' transduction summary statistics.", 'tab:transduction')
    p1 = get_fig(reports, 'fig_transduction_groups')
    p2 = get_fig(reports, 'fig_transduction_polya')
    tex += twofigs(p1, 'Transduction lineage group sizes.', 'fig:trans_groups',
                   p2, "Poly-A signal and tail-length distribution.", 'fig:trans_polya')
    h, rows = read_csv(reports / 'transduction_per_copy.csv', max_rows=20)
    if rows:
        safe_cols = [c for c in h if c in ['TE_name','Cluster','lineage_id','transduced_seq_len','polya_signal','polya_tail']]
        if safe_cols:
            tex += longtable(h, rows, "3' transduction annotations per copy (first 20 entries).",
                             'tab:trans_per_copy', cols=safe_cols,
                             note='First 20 rows; full data in reports/transduction\_per\_copy.csv.')
    return tex

def ch_antisense(d):
    reports = d / 'reports'
    txt = read_txt(reports / 'antisense_report.txt')
    tex = r'\chapter{Antisense and Bidirectional Promoter Activity}' + '\n\n'
    if not txt:
        return tex + notavail('antisense / bidirectional promoter')
    info = parse_report(txt)
    tex += (
        'Many TE families function as \\textit{bidirectional promoters}, driving expression both '
        'on the sense strand and, antisense, into neighbouring host genes. The bidirectionality score '
        'quantifies the symmetry of expression between the two orientations: a score of 0.5 '
        'indicates perfectly balanced bidirectional expression, while 0.0 or 1.0 indicate fully '
        'unidirectional expression. Copies with a bidirectionality score $\\geq 0.4$ are classified '
        'as putative bidirectional promoters (Table~\\ref{tab:antisense}).\n\n'
    )
    tex += kv_table(list(info.items()), 'Antisense / bidirectional promoter summary.', 'tab:antisense')
    p1 = get_fig(reports, 'fig_antisense_expr')
    p2 = get_fig(reports, 'fig_antisense_motifs')
    tex += twofigs(p1, 'Sense vs.\ antisense expression per copy.', 'fig:anti_expr',
                   p2, 'JASPAR motifs near antisense TSSs.', 'fig:anti_motifs')
    return tex

def ch_ctcf(d):
    reports = d / 'reports'
    txt = read_txt(reports / 'ctcf_tad_report.txt')
    tex = r'\chapter{CTCF Binding and TAD Proximity}' + '\n\n'
    if not txt:
        return tex + notavail('CTCF / TAD proximity')
    info = parse_report(txt)
    ctcf_n = info.get('CTCF core motif', '0')
    total = info.get('Copies', '?')
    tex += (
        f'CTCF (CCCTC-binding factor) is the primary insulator protein organising mammalian '
        f'chromatin into topologically associating domains (TADs). TEs harbouring a CTCF core '
        f'binding motif (JASPAR MA0139.1) can create ectopic loop anchors and disrupt or reinforce '
        f'existing TAD boundaries. Among the {esc(total)} copies analysed, {esc(ctcf_n)} carry a '
        f'CTCF core motif, equivalent to '
        f'{float(ctcf_n)/float(total)*100:.1f}\\% of the family '
        f'(Table~\\ref{{tab:ctcf}}).\n\n'
        if ctcf_n.isdigit() and total.isdigit() else
        f'CTCF binding motif and TAD proximity statistics are summarised in Table~\\ref{{tab:ctcf}}.\n\n'
    )
    tex += kv_table(list(info.items()), 'CTCF binding and TAD proximity summary.', 'tab:ctcf')
    p1 = get_fig(reports, 'fig_ctcf_overlap')
    p2 = get_fig(reports, 'fig_ctcf_tad_distance')
    tex += twofigs(p1, 'CTCF motif overlap per cluster.', 'fig:ctcf_overlap',
                   p2, 'Distance to nearest TAD boundary.', 'fig:ctcf_tad')
    return tex

def ch_epigenetic(d):
    reports = d / 'reports'
    txt = read_txt(reports / 'epigenetic_report.txt')
    tex = r'\chapter{Epigenetic Overlay}' + '\n\n'
    if not txt:
        return tex + notavail('epigenetic overlay')
    info = parse_report(txt)
    if info.get('Tracks used', '0').startswith('0'):
        return tex + (
            r'\textit{No epigenetic tracks were supplied for this analysis. '
            'Epigenetic overlay requires ChIP-seq or ATAC-seq signal tracks in bigWig format.}' + '\n\n'
        )
    tex += (
        'Epigenetic signal tracks (histone modifications and/or chromatin accessibility) were '
        'intersected with TE loci to identify copies in active chromatin states. '
        'Cluster-level differences in epigenetic signal reveal functionally distinct subfamilies '
        'with contrasting regulatory activity (Figure~\\ref{fig:epi_heatmap}).\n\n'
    )
    tex += kv_table(list(info.items()), 'Epigenetic overlay summary.', 'tab:epigenetic')
    p1 = get_fig(reports, 'fig_epigenetic_heatmap')
    p2 = get_fig(reports, 'fig_epigenetic_overlap')
    tex += twofigs(p1, 'Epigenetic signal heatmap across TE copies.', 'fig:epi_heatmap',
                   p2, 'Fraction of copies overlapping each track.', 'fig:epi_overlap')
    return tex

def ch_primers(d):
    tex = r'\chapter{Primer Design}' + '\n\n'
    tex += (
        'Family-specific 18-mer primers were designed to maximise coverage of TE copies while '
        'minimising off-target binding to non-repetitive genomic sequence. Candidate 18-mers were '
        'evaluated for: (i) exact-match coverage across all family copies; (ii) weighted expression '
        'coverage (sum of read counts for covered copies); (iii) thermodynamic parameters (melting '
        'temperature, GC content, self-complementarity); and (iv) genomic uniqueness. '
        'The Pareto-optimal set balancing coverage and specificity is reported in '
        'Table~\\ref{tab:primers_selected}.\n\n'
    )
    h, rows = read_csv(d / '06_primers' / 'selected_primers_summary.csv', max_rows=15)
    if rows:
        tex += longtable(h, rows, 'Selected 18-mer primers: coverage and expression statistics.',
                         'tab:primers_selected',
                         cols=[c for c in h if c in ['primer','coverage','strategy']],
                         note='coverage: number of TE copies matched.')
    h2, rows2 = read_csv(d / '06_primers' / 'cluster_top_primers.csv', max_rows=15)
    if rows2:
        tex += longtable(h2, rows2, 'Top primers per cluster.', 'tab:primers_cluster',
                         cols=[c for c in h2 if c in ['cluster','primer','coverage','strategy']])
    return tex

def ch_fold(d, pdb_renders):
    reports = d / 'reports'
    tex = r'\chapter{Protein Structure Prediction}' + '\n\n'
    if not pdb_renders:
        return tex + notavail('ColabFold protein structure prediction')

    tex += (
        'Open reading frames (ORFs) in the cluster consensus sequences were translated and submitted '
        'to \\textbf{ColabFold} (Mirdita \\textit{et al.}, 2022), which applies the AlphaFold2 neural '
        'network (Jumper \\textit{et al.}, 2021) to predict 3D protein structure. '
        'Per-residue confidence is quantified by the predicted local distance difference test score '
        '(pLDDT; 0--100): pLDDT $> 90$ (very high), 70--90 (confident), 50--70 (low/disordered), '
        '$< 50$ (very low confidence). The predicted aligned error (PAE; \\AA) measures positional '
        'uncertainty between residue pairs, with low PAE indicating reliable relative domain placement. '
        'The pTM score ($\\in [0,1]$) summarises global structural confidence. '
        'All structures were rendered from the rank-1 model (unrelaxed). '
        'The 3D backbone trace is coloured by pLDDT using the standard AlphaFold2 colour scheme: '
        '\\textbf{\\textcolor{blue}{dark blue}} ($>90$), '
        '\\textbf{\\textcolor{cyan}{light blue}} (70--90), '
        '\\textbf{\\textcolor{orange}{yellow-orange}} (50--70), '
        '\\textbf{\\textcolor{red}{red-orange}} ($<50$).\n\n'
    )

    # Summary table of all ORFs (strip the redundant family-folder prefix from
    # the ORF label so the name column stays inside the page margin)
    fam_pre = d.name.lower() + '_'
    rows_summary = []
    for r in pdb_renders:
        short = r['name']
        if short.lower().startswith(fam_pre):
            short = short[len(fam_pre):]
        rows_summary.append({
            'orf_name': short, 'n_res': r['n_res'],
            'mean_plddt': r['plddt_mean'], 'max_pae': r['max_pae'], 'ptm': r['ptm']
        })
    if rows_summary:
        h = ['orf_name', 'n_res', 'mean_plddt', 'max_pae', 'ptm']
        tex += longtable(h, rows_summary,
                         'ColabFold structure prediction summary (rank-1 model per ORF).',
                         'tab:fold_summary', small=True)

    # Summary figures from reports/
    p_sum = get_fig(reports, 'fig_fold_summary')
    p_plddt = get_fig(reports, 'fig_fold_plddt')
    p_orf = get_fig(reports, 'fig_fold_orf_map')
    if p_sum or p_plddt:
        tex += twofigs(p_sum, 'pLDDT summary across all ORFs.', 'fig:fold_summary',
                       p_plddt, 'Per-residue pLDDT profile (all ORFs).', 'fig:fold_plddt')
    if p_orf:
        tex += fig_latex(p_orf, 'ORF map on the consensus sequence.', 'fig:fold_orf_map',
                         width=r'0.80\linewidth')

    # Per-ORF rendered structures
    has_orf_png = any(r['png'] and Path(r['png']).exists() for r in pdb_renders)
    if has_orf_png:
        tex += r'\section{Per-ORF Structure Panels}' + '\n'
        tex += (
            'Each panel below shows (top row) two orthogonal views of the backbone C$\\alpha$ trace '
            'coloured by pLDDT, and (bottom row) the per-residue pLDDT profile and the predicted '
            'aligned error (PAE) matrix.\n\n'
        )
    for i, r in enumerate(pdb_renders):
        # short label: drop the redundant family-folder prefix
        short = r['name']
        if short.lower().startswith(fam_pre):
            short = short[len(fam_pre):]
        if r['png'] and Path(r['png']).exists():
            tex += r'\FloatBarrier' + '\n'
            tex += fig_latex(r['png'],
                             f"ColabFold structure: {short} "
                             f"({r['n_res']} residues, mean pLDDT = {r['plddt_mean']}).",
                             f'fig:fold_{i}', width=r'\linewidth')
        # Also include original ColabFold pLDDT/PAE PNGs if available
        stem = r['name']
        cf_dir = (d / 'reports' / 'colabfold_output')
        p_cov = get_fig(cf_dir, f'{stem}_coverage')
        p_pae_png = get_fig(cf_dir, f'{stem}_pae')
        p_plt_png = get_fig(cf_dir, f'{stem}_plddt')
        if p_cov or p_pae_png:
            tex += twofigs(p_cov, f'MSA coverage: {short}.', f'fig:cf_cov_{i}',
                           p_pae_png if p_pae_png else p_plt_png,
                           f'PAE matrix: {short}.', f'fig:cf_pae_{i}')
    return tex

def ch_grna(d):
    reports = d / 'reports'
    txt = read_txt(reports / 'grna_offtarget_report.txt')
    tex = r'\chapter{gRNA Off-target Analysis}' + '\n\n'
    if not txt:
        return tex + notavail('gRNA off-target analysis')
    info = parse_report(txt)
    tex += (
        'SpCas9 guide RNAs (gRNAs) were designed to target as many TE copies as possible '
        '(\\textit{coverage}) while minimising mismatched cleavage of non-target genomic loci '
        '(\\textit{off-target sites}). Candidate 20-mers were enumerated from all GG PAM sites '
        'within TE copies and evaluated by Cas-OFFinder (Bae \\textit{et al.}, 2014) with up to '
        'three mismatches. The Pareto frontier of coverage-vs.-specificity trade-off is reported in '
        'Table~\\ref{tab:grna_summary}.\n\n'
    )
    tex += kv_table(list(info.items()), 'gRNA off-target analysis summary.', 'tab:grna_summary')
    h, rows = read_csv(reports / 'grna_offtarget_guides.csv', max_rows=15)
    if rows:
        pref = ['guide', 'coverage', 'cov_pct', 'coverage_pct', 'offtarget',
                'n_offtarget', 'specificity', 'pareto']
        safe_cols = [c for c in pref if c in h]
        tex += longtable(h, rows, 'Top candidate SpCas9 gRNAs (coverage and off-target profile).',
                         'tab:grna_guides', cols=safe_cols if safe_cols else None,
                         note='coverage: TE copies targeted; off-tgt.: genomic sites with up to 3 '
                              'mismatches outside the family; spec.: on-target / (on-target + off-target).')
    p1 = get_fig(reports, 'fig_grna_offtarget_top')
    p2 = get_fig(reports, 'fig_grna_offtarget_pareto')
    p3 = get_fig(reports, 'fig_grna_offtarget_mm')
    tex += twofigs(p1, 'Top gRNAs: coverage vs.\ off-target count.', 'fig:grna_top',
                   p2, 'Pareto frontier of gRNA candidates.', 'fig:grna_pareto')
    if p3: tex += fig_latex(p3, 'Mismatch tolerance profile for selected gRNAs.', 'fig:grna_mm',
                            width=r'0.65\linewidth')
    return tex

def ch_multiassembly_ortholog(d):
    reports = d / 'reports'
    tex = r'\chapter{Multi-assembly Liftover and Ortholog Insertion}' + '\n\n'
    multi_txt = read_txt(reports / 'multiassembly_report.txt')
    orth_txt = read_txt(reports / 'ortholog_report.txt')
    if not multi_txt and not orth_txt:
        return tex + notavail('multi-assembly liftover and ortholog insertion')
    multi_info = parse_report(multi_txt) if multi_txt else {}
    orth_info = parse_report(orth_txt) if orth_txt else {}

    if multi_txt:
        tex += r'\section{Multi-assembly Coordinate Liftover}' + '\n'
        tex += (
            'Genomic coordinates of TE insertions were mapped across alternative mm10 assembly '
            'versions using UCSC liftOver chain files, enabling validation of copy coordinates '
            'and identification of assembly-specific insertions.\n\n'
        )
        tex += kv_table(list(multi_info.items()), 'Multi-assembly liftover parameters.', 'tab:multi')
        p1 = get_fig(reports, 'fig_multiassembly_mapping')
        p2 = get_fig(reports, 'fig_multiassembly_coords')
        tex += twofigs(p1, 'Liftover mapping rate per target assembly.', 'fig:multi_map',
                       p2, 'Coordinate shift distribution across assemblies.', 'fig:multi_coords')

    if orth_txt and not orth_info.get('Target species', '0  (none)').startswith('0'):
        tex += r'\section{Ortholog Insertion Analysis}' + '\n'
        tex += (
            'Cross-species net/chain alignments were used to determine whether each insertion is '
            'lineage-specific (absent in the outgroup) or shared (present before the speciation event). '
            'Lineage-specific insertions are candidates for species-specific regulatory innovation.\n\n'
        )
        tex += kv_table(list(orth_info.items()), 'Ortholog insertion analysis summary.', 'tab:ortholog')
        p1 = get_fig(reports, 'fig_ortholog_presence')
        p2 = get_fig(reports, 'fig_ortholog_summary')
        tex += twofigs(p1, 'Insertion presence/absence across species.', 'fig:orth_pres',
                       p2, 'Lineage-specific vs.\ shared insertions.', 'fig:orth_summary')
    return tex

# ── LaTeX preamble / document envelope ───────────────────────────────────────

def preamble(family_display, date_str):
    return rf"""
\documentclass[12pt,a4paper]{{report}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[a4paper, top=2.5cm, bottom=2.5cm, left=3cm, right=2.5cm, headheight=15pt]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{graphicx}}
\usepackage[table,xcdraw]{{xcolor}}
\usepackage[hidelinks, colorlinks=true, linkcolor=blue!60!black, citecolor=green!50!black, urlcolor=blue]{{hyperref}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\usepackage{{fancyhdr}}
\usepackage{{microtype}}
\usepackage{{parskip}}
\usepackage[explicit]{{titlesec}}
\usepackage{{float}}
\usepackage{{placeins}}
\usepackage{{enumitem}}
\usepackage{{amsmath}}
\usepackage{{array}}
\usepackage{{multirow}}
\usepackage{{textcomp}}
\usepackage{{etoolbox}}
\usepackage{{needspace}}

% ── Density: let chapters flow continuously rather than each forcing a new
% page (18 short chapters otherwise leave most pages nearly half empty), but
% break to a new page when too little room remains so a heading + its first
% lines can never overflow the bottom margin into the footer.
\makeatletter
\patchcmd{{\chapter}}
  {{\if@openright\cleardoublepage\else\clearpage\fi}}
  {{\par\addvspace{{1.5\baselineskip}}\FloatBarrier\Needspace*{{0.22\textheight}}}}
  {{}}{{}}
\makeatother
\raggedbottom

% ── Float placement: pack figures with text; avoid near-empty float pages.
\renewcommand{{\topfraction}}{{0.85}}
\renewcommand{{\bottomfraction}}{{0.5}}
\renewcommand{{\textfraction}}{{0.12}}
\renewcommand{{\floatpagefraction}}{{0.7}}
\setcounter{{topnumber}}{{3}}
\setcounter{{bottomnumber}}{{2}}
\setcounter{{totalnumber}}{{4}}

% Chapter title styling
\definecolor{{chapterblue}}{{RGB}}{{15,52,96}}
\titleformat{{\chapter}}[display]
  {{\normalfont\huge\bfseries\color{{chapterblue}}}}
  {{Chapter \thechapter}}{{6pt}}{{\huge\raggedright #1}}
\titlespacing*{{\chapter}}{{0pt}}{{4pt}}{{10pt}}

\titleformat{{\section}}{{\normalfont\large\bfseries\color{{chapterblue!80!black}}}}{{\thesection}}{{1em}}{{#1}}

% Header / footer (drop the right-side title so long chapter names in the
% left mark cannot collide with it).
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\small\itshape\nouppercase{{\leftmark}}}}
\fancyhead[R]{{}}
\fancyfoot[C]{{\thepage}}
\fancyfoot[R]{{\small GAMECA Analysis Report}}
\renewcommand{{\headrulewidth}}{{0.4pt}}
\fancypagestyle{{plain}}{{\fancyhf{{}}\fancyfoot[C]{{\thepage}}\fancyfoot[R]{{\small GAMECA Analysis Report}}\renewcommand{{\headrulewidth}}{{0pt}}}}

% Table row colour
\rowcolors{{2}}{{blue!4}}{{white}}

% Caption font
\captionsetup{{font=small, labelfont=bf, width=0.92\textwidth}}
\captionsetup[subfigure]{{font=small, labelfont=bf}}

\setlist[itemize]{{itemsep=2pt, topsep=4pt}}
\setlist[enumerate]{{itemsep=2pt, topsep=4pt}}

\begin{{document}}

\begin{{titlepage}}
  \centering
  \vspace*{{\fill}}
  {{\color{{chapterblue}}\rule{{\linewidth}}{{3pt}}}}\\[12pt]
  {{\Huge\bfseries\color{{chapterblue}} GAMECA Analysis Report}}\\[8pt]
  {{\rule{{\linewidth}}{{1pt}}}}\\[20pt]
  {{\huge\itshape {esc(family_display)}}}\\[10pt]
  {{\large Genome-wide Analysis of Mobile Element Clusters and Activity}}\\[40pt]
  {{\normalsize Assembly: mm10 (GRCm38)\\[6pt]
  Date: {date_str}\\[6pt]
  Generated by GAMECA pipeline}}
  \vspace{{\fill}}
  {{\color{{chapterblue}}\rule{{\linewidth}}{{1.5pt}}}}
\end{{titlepage}}

\tableofcontents
\listoffigures
\listoftables

\clearpage
"""

CLOSING = r"""
\end{document}
"""

# ── Main report generator ─────────────────────────────────────────────────────

def generate_report(family_dir_str):
    d = Path(family_dir_str).resolve()
    if not d.is_dir():
        print(f'ERROR: {d} is not a directory.', file=sys.stderr); return

    family = d.name
    print(f'\n{"="*60}')
    print(f'[{family}] Generating LaTeX report…')

    reports_dir = d / 'reports'
    out_dir = d / 'report_latex'
    out_dir.mkdir(exist_ok=True)

    stats = parse_stats(read_txt(d / '02_statistics' / 'overall_statistics.txt'))
    family_display = stats.get('family', family)
    _, cl_rows = read_csv(d / '03_clustering' / 'cluster_summary.csv')
    date_str = datetime.now().strftime('%d %B %Y')

    # ── Render PDB structures ────────────────────────────────────────────────
    pdb_renders = []
    cf_dir = reports_dir / 'colabfold_output' if reports_dir.exists() else None
    if cf_dir and cf_dir.is_dir() and list(cf_dir.glob('*.pdb')):
        print(f'[{family}] Rendering ColabFold structures…')
        pdb_renders = render_all_pdbs(cf_dir, out_dir / 'pdb_renders')

    # ── Pre-generate basic sequence figures (for families without reports/) ──
    seq_figs = pregenerate_seq_figures(d, out_dir)

    # ── Build chapters ───────────────────────────────────────────────────────
    print(f'[{family}] Building chapters…')
    body = ''
    body += ch_summary(d, stats, cl_rows, family_display)
    body += ch_overview(d, stats, cl_rows, seq_figs=seq_figs)
    body += ch_clustering(d)
    body += ch_consensus(d)
    body += ch_motif(d, stats)
    body += ch_motif_gains(d)
    body += ch_go(d)
    body += ch_expression(d)
    body += ch_ltr_struct(d)
    body += ch_divergence(d)
    body += ch_phylo(d)
    body += ch_transduction(d)
    body += ch_antisense(d)
    body += ch_ctcf(d)
    body += ch_epigenetic(d)
    body += ch_primers(d)
    body += ch_fold(d, pdb_renders)
    body += ch_grna(d)
    body += ch_multiassembly_ortholog(d)

    # Guard every section heading so it (and its first lines) cannot land at the
    # very bottom of a page and spill over the footer. \FloatBarrier first
    # places any pending figures (otherwise deferred floats land during \output
    # and \Needspace cannot account for them); \Needspace then breaks to a new
    # page if fewer than ~6 lines remain.
    body = body.replace('\\section{', '\\Needspace*{6\\baselineskip}\\section{')

    tex_content = preamble(family_display, date_str) + body + CLOSING

    tex_path = out_dir / 'report.tex'
    tex_path.write_text(tex_content, encoding='utf-8')
    print(f'[{family}] Written: {tex_path}')

    # ── Compile ──────────────────────────────────────────────────────────────
    print(f'[{family}] Compiling (pass 1)…')
    env = os.environ.copy()
    # Prepend the TinyTeX bin only if it exists (macOS dev box); on the cluster
    # pdflatex is found via PATH / $PDFLATEX instead.
    _tinytex = '/Users/anmol/Library/TinyTeX/bin/universal-darwin'
    if os.path.isdir(_tinytex):
        env['PATH'] = _tinytex + ':' + env.get('PATH', '')
    if not (os.path.isabs(PDFLATEX) and os.path.exists(PDFLATEX)) and shutil.which(PDFLATEX) is None:
        print(f'[{family}] ✗ pdflatex not found (looked for "{PDFLATEX}"). '
              f'report.tex was written to {tex_path}; install TeX or set $PDFLATEX '
              f'to build the PDF.')
        return
    # Run three passes (no halt-on-error) so cross-references, the ToC and the
    # lists of figures/tables fully resolve; a transient first-pass error does
    # not abort the later passes that produce the final PDF.
    for pass_n in range(3):
        result = subprocess.run(
            [PDFLATEX, '-interaction=nonstopmode', 'report.tex'],
            cwd=str(out_dir), capture_output=True, text=True, env=env, timeout=300
        )
        if result.returncode != 0:
            log = result.stdout[-3000:] if result.stdout else ''
            print(f'  WARNING: pdflatex pass {pass_n+1} exited {result.returncode}')
            if pass_n == 2:
                print(f'  Last output:\n{log}')
                (out_dir / 'compile_error.log').write_text(result.stdout + '\n' + result.stderr)
        else:
            print(f'[{family}] Pass {pass_n+1} OK.')

    pdf_path = out_dir / 'report.pdf'
    if pdf_path.exists():
        size_mb = pdf_path.stat().st_size / 1_048_576
        print(f'[{family}] ✓ PDF: {pdf_path}  ({size_mb:.1f} MB)')
        try: subprocess.Popen(['open', str(pdf_path)])
        except Exception: pass
    else:
        print(f'[{family}] ✗ PDF not produced — check {out_dir}/compile_error.log')

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folders', nargs='+', help='Family analysis folder(s)')
    args = ap.parse_args()
    for folder in args.folders:
        generate_report(folder)

if __name__ == '__main__':
    main()
