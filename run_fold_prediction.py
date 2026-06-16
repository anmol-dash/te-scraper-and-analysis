#!/usr/bin/env python3
"""
run_fold_prediction.py --- GAMECA Fold Prediction
ORF prediction → protein translation → ColabFold → pLDDT analysis + figures.

Pipeline:
  1. Load locus sequences from --input CSV (Seq column required)
  2. Select top --source-seqs loci (by total expression if available, else by length)
  3. Find ORFs in all 6 reading frames across those loci (filter by --min-aa)
  4. Deduplicate by exact protein sequence; keep top --top-n by length
  5. Write combined FASTA → colabfold_batch → PDB + scores JSON per model
  6. Parse pLDDT per residue from JSON; extract Cα backbone from PDB
  7. Generate three figures:
       fig_fold_orf_map.pdf    --- ORF positions/lengths in representative loci
       fig_fold_plddt.pdf      --- per-residue pLDDT + Cα backbone trace per model
       fig_fold_summary.pdf    --- mean pLDDT + PAE summary across all models
  8. Write fold_measured_values.tex + fold_report.txt

Usage:
    python run_fold_prediction.py \\
        --input /path/mt2_mm_ultracombo_countsv4.csv \\
        --reports-dir /home/amodz/anmol/reports4 \\
        [--family MT2_Mm] \\
        [--min-aa 100] \\
        [--top-n 5] \\
        [--source-seqs 100] \\
        [--colabfold-cmd colabfold_batch] \\
        [--num-recycles 3] \\
        [--num-models 1]
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────────

_CODON_TABLE = {
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
    "AAT":"N","AAC":"N","AAA":"K","AAG":"K","GAT":"D","GAC":"D","GAA":"E","GAG":"E",
    "TGT":"C","TGC":"C","TGA":"*","TGG":"W","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
    "AGT":"S","AGC":"S","AGA":"R","AGG":"R","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}

# AlphaFold/ColabFold pLDDT confidence tiers
_PLDDT_TIERS = [
    (90, 101, "#0053D6", "Very high (>90)"),
    (70,  90, "#65CBF3", "Confident (70–90)"),
    (50,  70, "#FFDB13", "Low (50–70)"),
    ( 0,  50, "#FF7D45", "Very low (<50)"),
]

_NON_EXPR = {
    "chr","chromosome","chrom","#chrom","genoname","genostart","genoend",
    "start","chromstart","stop","end","chromend","strand","te_name",
    "repname","seq","cluster","cluster_id","umap_x","umap_y","pca_x","pca_y",
    "length","gc_content","te_id","repclass","repfamily","swscore","millidiv",
    "name","score","locus","unnamed: 0","_total_expr","tx_chr","tx_start",
    "tx_stop","tx_strand","te_chromosome","te_start","te_end","te_strand","te_length",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _pp(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _texesc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def _auto_expr_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c.lower().strip() not in _NON_EXPR
            and pd.api.types.is_numeric_dtype(df[c])]


# ── sequence utilities ─────────────────────────────────────────────────────────

def _revcomp(seq: str) -> str:
    comp = str.maketrans("ACGTN", "TGCAN")
    return seq.upper().translate(comp)[::-1]


def _translate(seq: str) -> str:
    return "".join(_CODON_TABLE.get(seq[i:i+3], "X") for i in range(0, len(seq)-2, 3))


def find_orfs(seq: str, min_aa: int = 100) -> list:
    """Return list of dicts with keys: start, stop, strand, frame, length_aa, protein."""
    seq = seq.upper().replace(" ", "")
    orfs = []
    for strand, s in (("+", seq), ("-", _revcomp(seq))):
        for frame in range(3):
            i = frame
            orf_start = None
            while i + 3 <= len(s):
                codon = s[i:i+3]
                aa = _CODON_TABLE.get(codon, "X")
                if aa == "M" and orf_start is None:
                    orf_start = i
                elif aa == "*" and orf_start is not None:
                    length_aa = (i - orf_start) // 3
                    if length_aa >= min_aa:
                        protein = _translate(s[orf_start:i])
                        if strand == "+":
                            orfs.append({"start": orf_start, "stop": i + 3,
                                         "strand": strand, "frame": frame + 1,
                                         "length_aa": length_aa, "protein": protein})
                        else:
                            orfs.append({"start": len(seq) - (i + 3),
                                         "stop":  len(seq) - orf_start,
                                         "strand": strand, "frame": -(frame + 1),
                                         "length_aa": length_aa, "protein": protein})
                    orf_start = None
                i += 3
    return orfs


# ── locus + ORF selection ──────────────────────────────────────────────────────

def select_top_loci(df: pd.DataFrame, n: int, expr_cols: list) -> pd.DataFrame:
    """Return up to n loci ranked by total expression (or by Seq length if no expr)."""
    if expr_cols:
        df = df.copy()
        df["_total_expr"] = df[expr_cols].sum(axis=1)
        return df.nlargest(n, "_total_expr")
    return df.assign(_seq_len=df["Seq"].str.len()).nlargest(n, "_seq_len")


def collect_unique_orfs(df: pd.DataFrame, min_aa: int, top_n: int) -> list:
    """Find ORFs in all loci in df, deduplicate by protein sequence, return top_n by length."""
    seen_proteins: set = set()
    unique_orfs = []
    total_loci = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 50 == 0 and idx > 0:
            _pp(f"    Scanning ORFs in locus {idx}/{total_loci}...")
        seq = str(row.get("Seq", ""))
        if len(seq) < min_aa * 3 + 3:
            continue
        for orf in find_orfs(seq, min_aa):
            p = orf["protein"]
            if p not in seen_proteins:
                seen_proteins.add(p)
                unique_orfs.append(orf)
    unique_orfs.sort(key=lambda o: o["length_aa"], reverse=True)
    return unique_orfs[:top_n]


# ── consensus-sequence folding (#11) ─────────────────────────────────────────────

def _read_nt_fasta(path: Path) -> list:
    """Return list of (name, seq) from a nucleotide FASTA."""
    records, name, buf = [], None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(buf)))
            name, buf = line[1:].strip().split()[0], []
        else:
            buf.append(line.strip())
    if name is not None:
        records.append((name, "".join(buf)))
    return records


def _majority_consensus(seqs: list) -> str:
    """Position-wise majority consensus over padded sequences (approximate)."""
    seqs = [s.upper() for s in seqs if s]
    if not seqs:
        return ""
    L = max(len(s) for s in seqs)
    out = []
    for i in range(L):
        counts = {}
        for s in seqs:
            if i < len(s) and s[i] in "ACGT":
                counts[s[i]] = counts.get(s[i], 0) + 1
        out.append(max(counts, key=counts.get) if counts else "N")
    return "".join(out)


def consensus_orfs(named_seqs: list, family: str, min_aa: int) -> list:
    """One longest ORF per consensus sequence (>= min_aa). Returns ORF dicts."""
    out = []
    seen = set()
    for cname, seq in named_seqs:
        seq = str(seq).upper().replace("-", "")
        if len(seq) < min_aa * 3 + 3:
            continue
        found = find_orfs(seq, min_aa)
        if not found:
            continue
        best = max(found, key=lambda o: o["length_aa"])
        if best["protein"] in seen:
            continue
        seen.add(best["protein"])
        best["consensus_name"] = cname
        out.append(best)
    out.sort(key=lambda o: o["length_aa"], reverse=True)
    return out


def collect_consensus_orfs(df, consensus_fasta, per_cluster, family, min_aa) -> list:
    """Gather consensus-derived ORFs from a FASTA and/or per-cluster majority consensus."""
    named_seqs = []
    if consensus_fasta and Path(consensus_fasta).exists():
        recs = _read_nt_fasta(consensus_fasta)
        named_seqs.extend(recs)
        _pp(f"  Loaded {len(recs)} consensus sequences from {Path(consensus_fasta).name}")
    elif consensus_fasta:
        _pp(f"  WARNING: --consensus-fasta '{consensus_fasta}' not found.")
    if per_cluster and "Cluster" in df.columns:
        for cl, sub in df.groupby("Cluster"):
            cons = _majority_consensus([str(s) for s in sub["Seq"]])
            named_seqs.append((f"cluster_{cl}_n{len(sub)}", cons))
        _pp(f"  Built {df['Cluster'].nunique()} per-cluster majority consensuses")
    elif per_cluster:
        _pp("  WARNING: --per-cluster set but no 'Cluster' column in input.")
    return consensus_orfs(named_seqs, family, min_aa)


# ── FASTA I/O ─────────────────────────────────────────────────────────────────

def write_fasta(orfs: list, family: str, fasta_path: Path):
    lines = []
    for i, orf in enumerate(orfs):
        if orf.get("consensus_name"):
            tag = orf["consensus_name"].replace(" ", "_")[:30]
            name = f"{family}_{tag}_aa{orf['length_aa']}"
        else:
            name = f"{family}_orf{i+1}_aa{orf['length_aa']}_fr{orf['frame']}"
        orf["name"] = name
        lines.append(f">{name}")
        lines.append(orf["protein"])
    fasta_path.write_text("\n".join(lines) + "\n")
    _pp(f"  Wrote {len(orfs)} ORFs to {fasta_path.name}")


# ── ColabFold runner ───────────────────────────────────────────────────────────

def find_colabfold(cmd_override: str = "") -> str:
    """Return the colabfold_batch executable path, or '' if not found."""
    if cmd_override:
        return shutil.which(cmd_override) or cmd_override
    candidates = [
        "colabfold_batch",
        os.path.expanduser("~/localcolabfold/colabfold-conda/bin/colabfold_batch"),
        os.path.expanduser("~/.local/bin/colabfold_batch"),
        "/usr/local/bin/colabfold_batch",
        "/opt/colabfold/bin/colabfold_batch",
    ]
    for c in candidates:
        found = shutil.which(c) or (os.path.isfile(c) and os.access(c, os.X_OK) and c)
        if found:
            return found
    # Try conda envs
    conda_base = os.environ.get("CONDA_PREFIX", "")
    if conda_base:
        p = os.path.join(conda_base, "bin", "colabfold_batch")
        if os.path.isfile(p):
            return p
    return ""


def _alphafold_importable() -> bool:
    """True if the `alphafold` module can be imported (required by colabfold_batch)."""
    try:
        r = subprocess.run([sys.executable, "-c", "import alphafold"],
                           capture_output=True, text=True)
        return r.returncode == 0
    except Exception:               # noqa: BLE001
        return False


def _pip_install_colabfold_alphafold() -> bool:
    """pip install colabfold WITH the full alphafold extra. Returns True on success."""
    _pp("Installing colabfold[alphafold] via pip (this pulls in the alphafold module)...")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade",
             "colabfold[alphafold]>=1.5.5"])
        return r.returncode == 0
    except Exception as exc:        # noqa: BLE001
        _pp(f"  pip install colabfold[alphafold] failed: {exc}")
        return False


def _auto_install_colabfold() -> str:
    """Install colabfold_batch if missing. Tries pip first, then localColabFold installer.

    Installs the FULL `colabfold[alphafold]` extra (not alphafold-minus-data) so the
    `alphafold` module colabfold_batch imports is actually present.
    """
    # ── 1. pip install colabfold[alphafold] (fast, ~500 MB) ───────────────────
    _pp("colabfold_batch not found — installing via pip (colabfold[alphafold])...")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "colabfold[alphafold]>=1.5.5"])
        if r.returncode == 0:
            found = shutil.which("colabfold_batch")
            if found:
                _pp(f"  pip install OK: {found}")
                return found
            # entry point may be in the same Scripts/bin as this interpreter
            scripts = Path(sys.executable).parent
            candidate = scripts / "colabfold_batch"
            if candidate.exists() and os.access(candidate, os.X_OK):
                _pp(f"  pip install OK: {candidate}")
                return str(candidate)
        _pp("  pip install finished but colabfold_batch not found — trying localColabFold installer")
    except Exception as exc:
        _pp(f"  pip install failed ({exc}) — trying localColabFold installer")

    # ── 2. localColabFold bash installer fallback (~20 GB) ────────────────────
    cf_bin = os.path.expanduser("~/localcolabfold/colabfold-conda/bin/colabfold_batch")
    if os.path.isfile(cf_bin) and os.access(cf_bin, os.X_OK):
        return cf_bin

    _pp("Downloading localColabFold installer (~20 GB, may take 20-30 min)...")
    installer_url = "https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabfold_linux.sh"
    home = os.path.expanduser("~")
    installer = os.path.join(home, "install_colabfold_linux.sh")
    try:
        r = subprocess.run(["wget", "-q", "-O", installer, installer_url])
        if r.returncode != 0:
            _pp("ERROR: failed to download localColabFold installer")
            return ""
        os.chmod(installer, 0o755)
        r = subprocess.run(["bash", installer], cwd=home)
        if r.returncode != 0:
            _pp("ERROR: localColabFold installer exited with non-zero status")
            return ""
    except Exception as exc:
        _pp(f"ERROR: localColabFold install failed: {exc}")
        return ""
    finally:
        try:
            os.remove(installer)
        except OSError:
            pass

    if os.path.isfile(cf_bin) and os.access(cf_bin, os.X_OK):
        _pp(f"  localColabFold installed: {cf_bin}")
        return cf_bin
    _pp("ERROR: installer finished but colabfold_batch binary not found at expected path")
    return ""


# ── Singularity / Apptainer image management ────────────────────────────────────

# Official ColabFold container (ships colabfold_batch + a CUDA-enabled JAX/alphafold
# stack). Override with --singularity-source if your cluster needs a different tag
# or you maintain your own .def file.
DEFAULT_SIF_SOURCE = "docker://ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2"


def _singularity_runtime() -> str:
    """Return 'apptainer' or 'singularity' (whichever is on PATH), or '' if neither."""
    for exe in ("apptainer", "singularity"):
        if shutil.which(exe):
            return exe
    return ""


def ensure_singularity_image(sif_path: str, source: str = DEFAULT_SIF_SOURCE,
                             force: bool = False) -> str:
    """Make sure `sif_path` exists, building it from `source` if needed.

    `source` may be a docker:// (or library://) URI, or a path to a .def file —
    `<runtime> build <sif> <source>` handles both. Returns the .sif path on
    success, or '' on failure.
    """
    sif = Path(sif_path).expanduser()
    if sif.is_file() and not force:
        _pp(f"  Singularity image present: {sif}")
        return str(sif)

    runtime = _singularity_runtime()
    if not runtime:
        _pp("FATAL: neither `apptainer` nor `singularity` is on PATH — cannot "
            "build or run the ColabFold container.")
        return ""

    if force and sif.is_file():
        _pp(f"  --build-singularity: rebuilding existing image {sif}")
    else:
        _pp(f"  Singularity image not found at {sif} — building from {source}")

    sif.parent.mkdir(parents=True, exist_ok=True)
    # `build` from a .def file needs --fakeroot on unprivileged HPC; building from
    # a docker:// URI does not. Add --fakeroot only for .def sources.
    build_cmd = [runtime, "build"]
    if str(source).endswith(".def"):
        build_cmd.append("--fakeroot")
    build_cmd += [str(sif), str(source)]
    _pp(f"  $ {' '.join(build_cmd)}")
    _pp("  (first build pulls a multi-GB image and can take 10-30 min)")
    try:
        result = subprocess.run(build_cmd)
    except FileNotFoundError as exc:
        _pp(f"FATAL: failed to invoke {runtime}: {exc}")
        return ""
    if result.returncode != 0 or not sif.is_file():
        _pp(f"FATAL: {runtime} build failed (exit {result.returncode}); no image at {sif}.")
        _pp("  Check that the source URI/tag exists and that you have build "
            "permissions (some clusters require `--fakeroot` or a build node).")
        return ""
    _pp(f"  Built ColabFold image: {sif}")
    return str(sif)


def _aligned_fasta_to_a3m(records: list, query_idx: int) -> str:
    """Convert a list of (name, aligned_seq) pairs to a3m format for one query.

    Query columns (non-gap in query): uppercase target residue or '-'.
    Insertion columns (gap in query): lowercase target residue, skipped if target also gaps.
    """
    qseq = records[query_idx][1]
    lines = [f">{records[query_idx][0]}", qseq.replace("-", "")]
    for j, (hname, hseq) in enumerate(records):
        if j == query_idx:
            continue
        out = []
        for q, t in zip(qseq, hseq):
            if q == "-":
                if t != "-":
                    out.append(t.lower())   # insertion relative to query
            else:
                out.append(t if t != "-" else "-")
        lines += [f">{hname}", "".join(out)]
    return "\n".join(lines) + "\n"


def run_mafft(fasta_path: Path, a3m_dir: Path) -> bool:
    """Align ORF protein sequences with MAFFT; write per-query a3m files for ColabFold."""
    mafft = shutil.which("mafft")
    if not mafft:
        _pp("  ERROR: mafft not found on PATH — install with: conda install -c bioconda mafft")
        return False

    aligned_path = fasta_path.with_suffix(".mafft.fasta")
    cmd = [mafft, "--auto", "--amino", "--quiet", str(fasta_path)]
    _pp(f"  $ {' '.join(cmd)} > {aligned_path.name}")
    with open(aligned_path, "w") as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        _pp(f"  ERROR: MAFFT failed:\n{result.stderr.strip()}")
        return False

    records, name, buf = [], None, []
    for line in aligned_path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(buf)))
            name, buf = line[1:].strip(), []
        else:
            buf.append(line.strip().upper())
    if name is not None:
        records.append((name, "".join(buf)))

    if not records:
        _pp("  ERROR: MAFFT produced no output.")
        return False

    a3m_dir.mkdir(parents=True, exist_ok=True)
    for i, (qname, _) in enumerate(records):
        (a3m_dir / f"{qname}.a3m").write_text(_aligned_fasta_to_a3m(records, i))
    _pp(f"  Wrote {len(records)} a3m files to {a3m_dir.name}/")
    return True


def run_colabfold(fasta_path: Path, cf_dir: Path, cmd: str,
                  num_recycles: int = 3, num_models: int = 1,
                  singularity_image: str = "", use_gpu: bool = True,
                  a3m_dir: Path = None) -> bool:
    """Run colabfold_batch, optionally with a pre-computed MAFFT MSA or inside Singularity."""
    cf_dir.mkdir(parents=True, exist_ok=True)
    # ColabFold ≥1.5 renamed the MSA modes (old: "MMseqs2-UniRef-Environmental"/
    # "custom"; new: "mmseqs2_uniref_env"/"mmseqs2_uniref"/"single_sequence").
    if a3m_dir is not None:
        # Pre-computed MAFFT MSA: ColabFold reads the .a3m files straight from the
        # input directory. --msa-mode only governs *generating* an MSA, and "custom"
        # no longer exists — so omit it entirely (also keeps the run fully offline).
        input_arg = str(a3m_dir)
        msa_args  = []
    else:
        input_arg = str(fasta_path)
        msa_args  = ["--msa-mode", "mmseqs2_uniref_env"]
    colabfold_args = [
        cmd, input_arg, str(cf_dir),
        "--num-recycle", str(num_recycles),
        "--num-models",  str(num_models),
    ] + msa_args
    if singularity_image:
        sing_prefix = ["singularity", "exec"]
        if use_gpu:
            sing_prefix.append("--nv")
        sing_prefix += ["--bind", f"{fasta_path.parent}:{fasta_path.parent}",
                        singularity_image]
        args = sing_prefix + colabfold_args
    else:
        args = colabfold_args
    _pp(f"  $ {' '.join(args)}")
    result = subprocess.run(args, text=True)
    return result.returncode == 0


# ── PDB / JSON parsers ─────────────────────────────────────────────────────────

def _parse_pdb_ca(pdb_path: Path):
    """Return (coords N×3, plddt N) arrays from Cα ATOM records (B-factor = pLDDT)."""
    coords, plddt = [], []
    with open(pdb_path) as f:
        for line in f:
            if line[:4] == "ATOM" and line[12:16].strip() == "CA":
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    plddt.append(float(line[60:66]))
                except ValueError:
                    continue
    if not coords:
        return np.zeros((0, 3)), np.array([])
    return np.array(coords), np.array(plddt)


def _pca_project(coords: np.ndarray) -> np.ndarray:
    """Project 3D Cα coords to 2D via SVD (best-fit plane). Returns N×2 array."""
    c = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(c, full_matrices=False)
    return c @ vh[:2].T


def parse_colabfold_results(cf_dir: Path, orfs: list) -> dict:
    """Match ColabFold JSON + PDB outputs back to ORF names. Returns dict keyed by orf name."""
    results = {}
    for orf in orfs:
        name = orf.get("name", "")
        plddt_per_res, max_pae, pdb_path = [], None, None

        # Find best-rank JSON (rank_001 preferred)
        json_hits = sorted(cf_dir.glob(f"{name}_scores_rank_001*.json"))
        if not json_hits:
            json_hits = sorted(cf_dir.glob(f"{name}_scores*.json"))
        if json_hits:
            data = json.loads(json_hits[0].read_text())
            plddt_per_res = data.get("plddt", [])
            max_pae       = data.get("max_pae", None)

        # Find best-rank PDB
        pdb_hits = sorted(cf_dir.glob(f"{name}_unrelaxed_rank_001*.pdb"))
        if not pdb_hits:
            pdb_hits = sorted(cf_dir.glob(f"{name}*.pdb"))
        if pdb_hits:
            pdb_path = pdb_hits[0]

        if plddt_per_res or pdb_path:
            results[name] = {
                "orf":     orf,
                "plddt":   np.array(plddt_per_res) if plddt_per_res else np.array([]),
                "max_pae": max_pae,
                "pdb":     pdb_path,
            }
    return results


# ── figures ────────────────────────────────────────────────────────────────────

def plot_orf_map(source_df: pd.DataFrame, orfs: list, family: str, out_path: Path):
    """Horizontal ORF map for the longest representative locus."""
    if source_df.empty or not orfs:
        return
    rep = source_df.loc[source_df["Seq"].str.len().idxmax()]
    seq_len = len(str(rep.get("Seq", "")))

    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.5 + len(orfs) * 0.55)))

    ax.axhline(0, color="#cccccc", linewidth=6, solid_capstyle="round", zorder=1)
    ax.text(seq_len, 0, f"{seq_len} bp", va="center", ha="left",
            fontsize=7.5, color="#888888", style="italic")

    colors = {"+":[f"#4C9BE8","#2980b9"], "-":["#E8604C","#c0392b"]}
    used_labels: set = set()
    for i, orf in enumerate(orfs):
        strand = orf["strand"]
        col    = colors[strand][i % 2]
        ypos   = (i + 1) * 0.85
        label  = f"{strand} strand" if strand not in used_labels else None
        used_labels.add(strand)
        ax.broken_barh([(orf["start"], orf["stop"] - orf["start"])],
                       (ypos - 0.3, 0.6), facecolors=col, edgecolors="white",
                       linewidth=0.5, zorder=2, label=label)
        ax.text((orf["start"] + orf["stop"]) / 2, ypos,
                f"ORF{i+1} ({orf['length_aa']} aa, frame {orf['frame']})",
                ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")

    ax.set_xlim(0, seq_len * 1.07)
    ax.set_ylim(-0.6, (len(orfs) + 1) * 0.85)
    ax.set_xlabel("Position in locus (bp)", fontsize=9)
    ax.set_title(f"{family} --- predicted ORFs (representative locus, {seq_len:,} bp)",
                 fontweight="bold", fontsize=10)
    ax.set_yticks([])
    ax.spines[["top","right","left"]].set_visible(False)
    handles = [mpatches.Patch(color=colors["+"][0], label="+ strand"),
               mpatches.Patch(color=colors["-"][0], label="− strand")]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    _pp(f"  Saved {out_path}")


def plot_plddt(fold_results: dict, out_path: Path):
    """Per-residue pLDDT line plots with Cα backbone trace insets."""
    if not fold_results:
        return
    n = len(fold_results)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols * 2,
                             figsize=(ncols * 6.5, nrows * 3.2),
                             gridspec_kw={"width_ratios": [2.5, 1] * ncols})
    if n == 1:
        axes = axes[np.newaxis, :]
    axes = axes.reshape(nrows, ncols, 2) if nrows > 1 else axes.reshape(1, ncols, 2)

    for idx, (name, res) in enumerate(fold_results.items()):
        row, col = divmod(idx, ncols)
        ax_plddt   = axes[row, col, 0]
        ax_struct  = axes[row, col, 1]
        plddt      = res["plddt"]
        orf        = res["orf"]

        # ── pLDDT line plot ──
        if len(plddt) > 0:
            for lo, hi, col_hex, _ in _PLDDT_TIERS:
                ax_plddt.axhspan(lo, hi, alpha=0.10, color=col_hex, linewidth=0)
            x = np.arange(1, len(plddt) + 1)
            # Color line segments by tier
            for i in range(len(plddt) - 1):
                p = (plddt[i] + plddt[i+1]) / 2
                c = next((h for lo, hi, h, _ in _PLDDT_TIERS if lo <= p < hi), "#0053D6")
                ax_plddt.plot([x[i], x[i+1]], [plddt[i], plddt[i+1]],
                              color=c, linewidth=1.4)
            ax_plddt.axhline(plddt.mean(), color="black", linewidth=0.8,
                             linestyle="--", label=f"mean={plddt.mean():.1f}")
            ax_plddt.set_ylim(0, 105)
            ax_plddt.set_ylabel("pLDDT", fontsize=8)
            ax_plddt.legend(fontsize=7.5, loc="lower right", framealpha=0.8)
        else:
            ax_plddt.text(0.5, 0.5, "no scores", transform=ax_plddt.transAxes,
                          ha="center", va="center", color="gray")
        ax_plddt.set_xlabel("Residue", fontsize=8)
        ax_plddt.set_title(
            f"{orf.get('name','?')}  |  {orf['length_aa']} aa  "
            f"|  frame {orf['frame']}  |  pLDDT {plddt.mean():.1f}" if len(plddt) else name,
            fontsize=8, fontweight="bold")
        ax_plddt.spines[["top","right"]].set_visible(False)
        ax_plddt.tick_params(labelsize=7.5)

        # ── backbone trace ──
        if res.get("pdb") and res["pdb"].exists():
            coords, b = _parse_pdb_ca(res["pdb"])
            if coords.shape[0] > 3:
                xy = _pca_project(coords)
                sc = ax_struct.scatter(xy[:, 0], xy[:, 1], c=b, cmap="RdYlBu",
                                       vmin=50, vmax=100, s=6, zorder=3)
                for i in range(len(xy) - 1):
                    p_mid = (b[i] + b[i+1]) / 2
                    c_hex = next((h for lo, hi, h, _ in _PLDDT_TIERS if lo <= p_mid < hi),
                                 "#0053D6")
                    ax_struct.plot(xy[i:i+2, 0], xy[i:i+2, 1],
                                   color=c_hex, linewidth=0.9, zorder=2)
                plt.colorbar(sc, ax=ax_struct, shrink=0.75, label="pLDDT")
            else:
                ax_struct.text(0.5, 0.5, "PDB\nparsed\n(< 4 residues)",
                               transform=ax_struct.transAxes, ha="center", va="center",
                               fontsize=7, color="gray")
        else:
            ax_struct.text(0.5, 0.5, "no PDB", transform=ax_struct.transAxes,
                           ha="center", va="center", color="gray")
        ax_struct.set_title("Cα trace\n(PCA)", fontsize=7.5)
        ax_struct.set_xticks([]); ax_struct.set_yticks([])
        ax_struct.spines[["top","right","left","bottom"]].set_visible(False)

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col, 0].set_visible(False)
        axes[row, col, 1].set_visible(False)

    # pLDDT tier legend
    tier_patches = [mpatches.Patch(color=h, alpha=0.55, label=lbl)
                    for _, _, h, lbl in _PLDDT_TIERS]
    fig.legend(handles=tier_patches, loc="lower center", ncol=4,
               fontsize=7.5, framealpha=0.85,
               bbox_to_anchor=(0.5, -0.01))

    plt.suptitle("Per-Residue pLDDT (ColabFold) + Cα Backbone", fontsize=11,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    _pp(f"  Saved {out_path}")


def plot_fold_summary(fold_results: dict, family: str, out_path: Path):
    """Horizontal bar chart of mean pLDDT + PAE across all predicted models."""
    if not fold_results:
        return
    names, mean_plddts, pae_vals = [], [], []
    for name, res in fold_results.items():
        p = res["plddt"]
        names.append(res["orf"].get("name", name).replace(family + "_", ""))
        mean_plddts.append(float(p.mean()) if len(p) else 0.0)
        pae_vals.append(res["max_pae"] if res["max_pae"] is not None else float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(10, max(2.5, 0.5 + len(names) * 0.55)))

    # ── mean pLDDT bars ──
    ax = axes[0]
    bar_colors = []
    for mp in mean_plddts:
        bar_colors.append(next((h for lo, hi, h, _ in _PLDDT_TIERS if lo <= mp < hi), "#FF7D45"))
    bars = ax.barh(names, mean_plddts, color=bar_colors, edgecolor="white", height=0.6)
    ax.set_xlim(0, 105)
    ax.axvline(90, color="#0053D6", linewidth=1, linestyle="--", alpha=0.5, label=">90 very high")
    ax.axvline(70, color="#65CBF3", linewidth=1, linestyle="--", alpha=0.5, label=">70 confident")
    for bar, val in zip(bars, mean_plddts):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8)
    ax.set_xlabel("Mean pLDDT", fontsize=9)
    ax.set_title(f"{family} --- Model Confidence", fontweight="bold", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.8)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    # ── PAE bars ──
    ax2 = axes[1]
    valid_pae = [(n, v) for n, v in zip(names, pae_vals) if not (isinstance(v, float) and np.isnan(v))]
    if valid_pae:
        pae_names, pae_v = zip(*valid_pae)
        pae_colors = ["#E8604C" if v > 15 else "#F39C12" if v > 8 else "#2ECC71"
                      for v in pae_v]
        pae_bars = ax2.barh(pae_names, pae_v, color=pae_colors,
                             edgecolor="white", height=0.6)
        for bar, val in zip(pae_bars, pae_v):
            ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}", va="center", fontsize=8)
        ax2.set_xlabel("Max PAE (Å)", fontsize=9)
        ax2.set_title("Predicted Aligned Error", fontweight="bold", fontsize=10)
        ax2.spines[["top","right"]].set_visible(False)
        ax2.tick_params(labelsize=8)
        ax2.invert_xaxis()
    else:
        ax2.text(0.5, 0.5, "PAE not available", transform=ax2.transAxes,
                 ha="center", va="center", color="gray")
        ax2.set_title("Predicted Aligned Error", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    _pp(f"  Saved {out_path}")


# ── measured-values writer ─────────────────────────────────────────────────────

def write_fold_values(results: dict, reports_dir: Path):
    ts         = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_orfs     = results["n_orfs"]
    n_folded   = results["n_folded"]
    best_name  = results.get("best_name", "---")
    best_mean  = results.get("best_mean_plddt", 0.0)
    mean_all   = results.get("mean_plddt_all", 0.0)
    best_pae   = results.get("best_max_pae", float("nan"))
    family     = results.get("family", "")

    tex = [
        "% Auto-generated by run_fold_prediction.py",
        f"% {ts}",
        "",
        rf"\providecommand{{\foldFamily}}{{{_texesc(family)}}}",
        rf"\providecommand{{\foldNOrfs}}{{{n_orfs}}}",
        rf"\providecommand{{\foldNFolded}}{{{n_folded}}}",
        rf"\providecommand{{\foldBestName}}{{\texttt{{{_texesc(best_name)}}}}}",
        rf"\providecommand{{\foldBestMeanPlddt}}{{{best_mean:.1f}}}",
        rf"\providecommand{{\foldMeanPlddtAll}}{{{mean_all:.1f}}}",
        rf"\providecommand{{\foldBestMaxPae}}{{{best_pae:.1f}" if not np.isnan(best_pae) else r"\providecommand{\foldBestMaxPae}{---}",
        "",
    ]
    (reports_dir / "fold_measured_values.tex").write_text("\n".join(tex))

    txt = [
        "=" * 60,
        f"GAMECA Fold Prediction --- Measured Values",
        f"Generated: {ts}",
        "=" * 60,
        f"  Family:              {family}",
        f"  Unique ORFs found:   {n_orfs}",
        f"  Structures folded:   {n_folded}",
        "",
        "  ── Best model ────────────────────────────",
        f"  Name:                {best_name}",
        f"  Mean pLDDT:          {best_mean:.1f}",
        f"  Max PAE:             {best_pae:.1f}" if not np.isnan(best_pae) else "  Max PAE:             ---",
        "",
        "  ── All models ────────────────────────────",
        f"  Mean pLDDT (all):    {mean_all:.1f}",
    ]
    (reports_dir / "fold_report.txt").write_text("\n".join(txt))
    _pp("  Written fold_measured_values.tex and fold_report.txt")


# ── argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ORF prediction + ColabFold for TE family sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",         required=True,
                   help="CSV with Seq column (+ optional expression columns)")
    p.add_argument("--reports-dir",   default="/home/amodz/anmol/reports4",
                   help="Output directory for figures and tex values")
    p.add_argument("--family",        default="MT2_Mm")
    p.add_argument("--min-aa",        type=int, default=100,
                   help="Minimum ORF length in amino acids")
    p.add_argument("--top-n",         type=int, default=5,
                   help="Number of unique ORFs to fold")
    p.add_argument("--source-seqs",   type=int, default=100,
                   help="Number of top loci to scan for ORFs")
    p.add_argument("--consensus-fasta", default="",
                   help="Nucleotide FASTA of consensus sequences to fold (#11); "
                        "e.g. the clustering stage's all_cluster_consensuses.fa")
    p.add_argument("--per-cluster",   action="store_true",
                   help="Build a majority consensus per Cluster in --input and fold it (#11)")
    p.add_argument("--colabfold-cmd", default="",
                   help="Path to colabfold_batch (auto-detected if blank)")
    p.add_argument("--use-mafft",     action="store_true",
                   help="Align ORFs locally with MAFFT and pass the MSA to colabfold_batch "
                        "via --msa-mode custom, instead of the online MMseqs2 API.")
    p.add_argument("--singularity-image", default="",
                   help="Path to ColabFold .sif file; runs via 'singularity exec [--nv] <sif> colabfold_batch'. "
                        "Bypasses the host alphafold-module check (useful when host GCC is too old). "
                        "If the file does not exist it is built automatically from --singularity-source.")
    p.add_argument("--singularity-source", default=DEFAULT_SIF_SOURCE,
                   help="Source to build the .sif from when --singularity-image is missing: a "
                        "docker:///library:// URI or a path to a .def file. "
                        f"Default: {DEFAULT_SIF_SOURCE}")
    p.add_argument("--build-singularity", action="store_true",
                   help="Force (re)build of the --singularity-image from --singularity-source, "
                        "even if the .sif already exists.")
    p.add_argument("--no-gpu",        action="store_true",
                   help="Omit --nv from the singularity exec call (CPU-only mode)")
    p.add_argument("--num-recycles",  type=int, default=3)
    p.add_argument("--num-models",    type=int, default=1)
    p.add_argument("--force",         action="store_true",
                   help="Re-run ColabFold even if output already exists")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    reports    = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    cf_dir     = reports / "colabfold_output"
    fasta_path = reports / f"{args.family}_orfs.fasta"

    _pp("=" * 60)
    _pp(f"GAMECA Fold Prediction --- {args.family}")
    _pp(f"  Input:    {args.input}")
    _pp(f"  Reports:  {args.reports_dir}")
    _pp(f"  Min ORF:  {args.min_aa} aa")
    _pp(f"  Top-N:    {args.top_n}")
    _pp("=" * 60)

    # ── 1. Load CSV ─────────────────────────────────────────────────────────────
    _pp("Loading CSV...")
    df = pd.read_csv(args.input, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    if "Seq" not in df.columns:
        _pp("ERROR: CSV must have a 'Seq' column. Exiting.")
        sys.exit(1)
    _pp(f"  {len(df):,} rows, {len(df.columns)} columns")

    expr_cols = _auto_expr_cols(df)
    _pp(f"  Expression columns ({len(expr_cols)}): {expr_cols[:6]}")

    # ── 2. Select source loci ──────────────────────────────────────────────────
    _pp(f"Selecting top {args.source_seqs} loci for ORF scanning...")
    source_df = select_top_loci(df, args.source_seqs, expr_cols)
    _pp(f"  Using {len(source_df)} loci")

    # ── 3. Find ORFs ────────────────────────────────────────────────────────────
    # #11: consensus-sequence folding takes priority when requested.
    cons_orfs = []
    if args.consensus_fasta or args.per_cluster:
        _pp("Collecting consensus-sequence ORFs (#11)...")
        cons_orfs = collect_consensus_orfs(df, args.consensus_fasta, args.per_cluster,
                                           args.family, args.min_aa)
        _pp(f"  {len(cons_orfs)} consensus ORFs")

    _pp(f"Finding ORFs (min {args.min_aa} aa, 6 reading frames)...")
    locus_orfs = collect_unique_orfs(source_df, args.min_aa, args.top_n)
    _pp(f"  {len(locus_orfs)} unique locus ORFs found")

    if cons_orfs:
        # Fold all consensus ORFs first, then top up to --top-n with locus ORFs.
        seen = {o["protein"] for o in cons_orfs}
        extra = [o for o in locus_orfs if o["protein"] not in seen]
        orfs = (cons_orfs + extra)[: max(args.top_n, len(cons_orfs))]
    else:
        orfs = locus_orfs[: args.top_n]
    _pp(f"  {len(orfs)} ORFs selected for folding "
        f"({len(cons_orfs)} consensus + {len(orfs) - len(cons_orfs)} locus)")
    if not orfs:
        _pp("  No ORFs found. Try --min-aa with a smaller value.")
        sys.exit(0)

    write_fasta(orfs, args.family, fasta_path)

    # ── 4. ORF map figure ───────────────────────────────────────────────────────
    orf_map_path = reports / "fig_fold_orf_map.pdf"
    if args.force or not orf_map_path.exists():
        _pp("Generating ORF map figure...")
        plot_orf_map(source_df, orfs, args.family, orf_map_path)

    # ── 5. ColabFold ────────────────────────────────────────────────────────────
    fold_results: dict = {}
    sif = args.singularity_image
    use_gpu = not args.no_gpu

    # When a Singularity/Apptainer image is requested, colabfold_batch lives
    # *inside* the container, so the in-container command is always the bare
    # "colabfold_batch" name — never a host path. Build the image first if it
    # is missing (or --build-singularity was passed).
    if sif or args.build_singularity:
        if not sif:
            _pp("FATAL: --build-singularity requires --singularity-image <path.sif>")
            sys.exit(3)
        sif = ensure_singularity_image(sif, args.singularity_source,
                                       force=args.build_singularity)
        if not sif:
            sys.exit(3)
        cf_cmd = "colabfold_batch"       # resolved inside the container
    else:
        cf_cmd = find_colabfold(args.colabfold_cmd)
        if not cf_cmd:
            cf_cmd = _auto_install_colabfold()
    if not cf_cmd:
        _pp("FATAL: colabfold_batch not found and auto-install failed.")
        _pp("  Tried: pip install colabfold[alphafold]  and  localColabFold installer.")
        _pp("  Pass --colabfold-cmd /path/to/colabfold_batch, or install ColabFold manually.")
        _pp("  Or use --singularity-image /path/to/colabfold.sif to bypass host GCC issues.")
        sys.exit(3)
    else:
        _pp(f"ColabFold found: {cf_cmd}" + (f"  (via Singularity: {sif})" if sif else ""))
        # When using Singularity the alphafold module lives inside the container —
        # skip the host-side importability check entirely.
        if not sif:
            if not _alphafold_importable():
                _pp("  `alphafold` module not importable — installing colabfold[alphafold]...")
                _pip_install_colabfold_alphafold()
                if not _alphafold_importable():
                    _pp("FATAL: `alphafold` module could not be installed; colabfold_batch "
                        "cannot run. Aborting fold prediction (no fake structures emitted).")
                    _pp("  TIP: re-run with --singularity-image /path/to/colabfold.sif "
                        "to bypass host GCC/Python dependency issues.")
                    sys.exit(3)
                _pp("  alphafold module now importable.")
        already_done = (not args.force and cf_dir.exists()
                        and any(cf_dir.glob("*.pdb")))
        if already_done:
            _pp("  [cache] ColabFold output exists --- parsing without re-running")
        else:
            a3m_dir = None
            if args.use_mafft:
                a3m_dir = reports / "mafft_a3m"
                _pp("Running MAFFT alignment...")
                if not run_mafft(fasta_path, a3m_dir):
                    _pp("FATAL: MAFFT failed; cannot build custom MSA for ColabFold.")
                    sys.exit(3)
            _pp("Running ColabFold (this may take several minutes)...")
            ok = run_colabfold(fasta_path, cf_dir, cf_cmd,
                               args.num_recycles, args.num_models,
                               singularity_image=sif, use_gpu=use_gpu,
                               a3m_dir=a3m_dir)
            if not ok:
                _pp("WARNING: ColabFold exited with non-zero status; "
                    "attempting to parse partial results.")

        _pp("Parsing ColabFold results...")
        fold_results = parse_colabfold_results(cf_dir, orfs)
        _pp(f"  {len(fold_results)} models parsed")

    # ── 6. pLDDT + backbone figure ──────────────────────────────────────────────
    plddt_path = reports / "fig_fold_plddt.pdf"
    if fold_results and (args.force or not plddt_path.exists()):
        _pp("Generating pLDDT / backbone figure...")
        plot_plddt(fold_results, plddt_path)

    # ── 7. Summary figure ───────────────────────────────────────────────────────
    summary_path = reports / "fig_fold_summary.pdf"
    if fold_results and (args.force or not summary_path.exists()):
        _pp("Generating summary figure...")
        plot_fold_summary(fold_results, args.family, summary_path)

    # ── 8. Measured values ─────────────────────────────────────────────────────
    mean_plddts = [r["plddt"].mean() for r in fold_results.values() if len(r["plddt"])]
    best_res    = max(fold_results.values(), key=lambda r: r["plddt"].mean()
                      if len(r["plddt"]) else 0.0) if fold_results else {}
    results_dict = {
        "family":          args.family,
        "n_orfs":          len(orfs),
        "n_folded":        len(fold_results),
        "best_name":       best_res.get("orf", {}).get("name", "---") if best_res else "---",
        "best_mean_plddt": float(best_res["plddt"].mean()) if best_res and len(best_res["plddt"]) else 0.0,
        "mean_plddt_all":  float(np.mean(mean_plddts)) if mean_plddts else 0.0,
        "best_max_pae":    best_res.get("max_pae") or float("nan") if best_res else float("nan"),
    }
    write_fold_values(results_dict, reports)

    _pp("=" * 60)
    _pp("DONE")
    _pp(f"  Figures:      {reports}")
    _pp(f"  ORF FASTA:    {fasta_path}")
    if cf_dir.exists():
        _pp(f"  ColabFold:    {cf_dir}")
    _pp("=" * 60)

    print("\n" + "=" * 60)
    print("MEASURED VALUES (paste into chat):")
    print("=" * 60)
    print((reports / "fold_report.txt").read_text())


if __name__ == "__main__":
    main()
