#!/usr/bin/env python3
_UI_BUILD = "20260528-1"  # bump this when changing ui.py
"""
ui.py — GAMECA terminal launcher.

On launch, this program offers a fast path to connect to a SLURM or LSF
HPC cluster, then lets you run the pipeline interactively or submit it as
a batch job.  Local utilities remain available through --local.

Usage:
    python ui.py                                    # offers HPC connection first
    python ui.py --host cluster.edu --user myuser   # skip hostname/user prompts
    python ui.py --results <dir>                    # display previously-retrieved results
    python ui.py --help-workflow                    # print workflow overview
"""

import argparse
import getpass
import shlex
import sys
import os
import re
import datetime
from pathlib import Path


# ── ANSI helpers ──────────────────────────────────────────────────────────────

def _c(text, code):
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

def bold(t):      return _c(t, "1")
def green(t):     return _c(t, "32")
def yellow(t):    return _c(t, "33")
def cyan(t):      return _c(t, "36")
def red(t):       return _c(t, "31")
def dim(t):       return _c(t, "2")
def magenta(t):   return _c(t, "35")
def blue(t):      return _c(t, "34")
def bright(t):    return _c(t, "97")

def _log(message: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"  {dim(f'[{ts}]')} {message}", flush=True)

def _header(title: str, width: int = 66) -> str:
    pad = width - len(title) - 4
    left = pad // 2
    right = pad - left
    return (
        cyan("╔" + "═" * (width - 2) + "╗") + "\n"
        + cyan("║") + " " * left + bold(bright(title)) + " " * right + cyan("║") + "\n"
        + cyan("╚" + "═" * (width - 2) + "╝")
    )

def _box(lines: list, width: int = 66) -> str:
    result = cyan("┌" + "─" * (width - 2) + "┐") + "\n"
    for line in lines:
        visible = _strip_ansi(line)
        pad = width - 2 - len(visible)
        result += cyan("│") + " " + line + " " * max(0, pad - 1) + cyan("│") + "\n"
    result += cyan("└" + "─" * (width - 2) + "┘")
    return result

def _strip_ansi(s: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", s)

def _divider(label: str = "", width: int = 66, char: str = "─") -> str:
    if label:
        label_str = f" {label} "
        side = (width - len(label_str)) // 2
        return dim(char * side) + bold(label_str) + dim(char * (width - side - len(label_str)))
    return dim(char * width)


# ── Simple prompts (no local file picker — paths are on the remote cluster) ───

def _ask(prompt: str, default: str = "") -> str:
    val = input(f"  {cyan('›')} {prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return val or default


def _confirm(prompt: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    val = input(f"  {cyan('›')} {prompt} ({hint}): ").strip().lower()
    if not val:
        return default
    return val.startswith("y")


def _ask_species_assembly(default_species: str = "human", default_assembly: str = "hg38"):
    species = _ask("Species (human/mouse)", default_species).strip().lower() or default_species
    if species in {"mouse", "mus musculus"} and default_assembly.startswith("hg"):
        default_assembly = "mm10"
    elif species in {"human", "homo sapiens"} and default_assembly.startswith("mm"):
        default_assembly = "hg38"
    assembly = _ask("Assembly/build (human: hg38/hg19; mouse: mm10/mm39)", default_assembly)
    return species, assembly


def _ask_source(default: str = "rmsk") -> str:
    """Ask whether to fetch TE coordinates from RepeatMasker (rmsk) or Dfam."""
    raw = _ask("Coordinate source — rmsk (RepeatMasker) or dfam (Dfam)", default).strip().lower()
    if raw not in {"rmsk", "dfam"}:
        print(f"  Unknown source '{raw}' — defaulting to {default}")
        return default
    return raw


def _ask_notify_email(client) -> str:
    """Ask for a completion-notification email, remembering the answer across steps."""
    stored = client.params.get("NOTIFY_EMAIL", "")
    email = _ask("Email for completion notification (blank = none)", stored).strip()
    if email:
        client.params["NOTIFY_EMAIL"] = email
    return email


# ── ASCII art ─────────────────────────────────────────────────────────────────

GAMECA_ART = r"""
  ██████╗  █████╗ ███╗   ███╗███████╗ ██████╗  █████╗
 ██╔════╝ ██╔══██╗████╗ ████║██╔════╝██╔════╝ ██╔══██╗
 ██║  ███╗███████║██╔████╔██║█████╗  ██║      ███████║
 ██║   ██║██╔══██║██║╚██╔╝██║██╔══╝  ██║      ██╔══██║
 ╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗╚██████╗ ██║  ██║
  ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝
"""

TAGLINE = "Gene Alignment, Motif, Expression & Clustering Analysis"

HPC_NOTICE = (
    red("━" * 66) + "\n"
    + red("  ⚠  HPC MODE  ") + bold("This UI submits jobs to a remote cluster.") + "\n"
    + "     " + dim("For local mode (no HPC):") + "\n"
    + "     " + cyan("python query.py --local --family THE1D-int --assembly hg38") + "\n"
    + red("━" * 66)
)

PIPELINE_DIAGRAM = """
  Coords ──→ Get Sequences ──┬──→ MAFFT ──→ CIAlign ──→ AlphaFold   [Branch A]
                             ├──→ UMAP / HDBSCAN ──→ Cluster QC      [Branch B]
                             ├──→ JASPAR overlap ──→ Motif analysis   [Branch C]
                             └──→ Expression collation                [Branch D]
                                        │                   │
                                  [Join B+D]           [Join C+D]
                                        └────────┬──────────┘
                                           Generate gRNAs
                                        ┌────────┴──────────┐
                                   Off-target QC       Stability QC
"""


# ── Workflow docs ─────────────────────────────────────────────────────────────

WORKFLOW = """
All steps run on the remote HPC cluster.  Files referenced below are
paths on the cluster, not on your local machine.

STEP 1 — Prepare  (te_prep.py)
  Downloads RepeatMasker annotations from UCSC, extracts sequences
  for a TE family, and writes the input CSV.

  OUTPUT → {family}_sequences.csv
           columns: chr, start, stop, TE_name, strand, Seq

STEP C — Clustering  (te_clustering.py)
  k-mer encoding → PCA / UMAP / t-SNE → HDBSCAN clustering.

STEP A — Alignment  (te_alignment.py)
  MAFFT global + per-cluster alignment, CIAlign visualisation,
  majority-vote consensus generation.

STEP M — Motif  (te_motif.py)
  bedtools intersect TE loci against JASPAR TFBS predictions,
  then Fisher's exact test per cluster × motif.
  Optionally runs HOMER findMotifsGenome.pl (--homer flag) on each
  cluster for known-motif enrichment; results in homer_results/.

STEP G — Gene / GO  (te_go.py)
  Looks up GO terms for enriched TF motifs via mygene.info API.

STEP E — Expression  (te_expression.py)
  Per-cluster expression boxplots from numeric columns.

ORCHESTRATION (query.py — runs on cluster)
  Runs the full core pipeline:
    python query.py --family HERVK --assembly hg38 --genome /path/hg38.fa --out-dir ./results

HPC SCHEDULER SUPPORT
  Auto-detected on connect.  Supports:
    LSF  — bsub / bjobs / bkill / bpeek
    SLURM — sbatch / squeue / scancel / srun
"""


# ── Results summary (for locally downloaded results) ─────────────────────────

def _tick(p: Path) -> str:
    return green("✔") if p.exists() else dim("·")


def display_results(out_dir: Path):
    out_dir = Path(out_dir).expanduser()
    if not out_dir.exists():
        print(red(f"\n  Directory not found: {out_dir}"))
        return

    print()
    print(_header(f"Results  ·  {out_dir.name}"))
    print()

    sections = [
        ("Core outputs", [
            ("clustered_data.csv",           "Clustered sequences CSV"),
            ("clustering_coordinates.csv",    "UMAP / PCA / t-SNE coords"),
            ("clustering_visualization.html", "Clustering interactive plot"),
        ]),
        ("Alignment", [
            ("cluster_alignments",  "Per-cluster alignment FASTAs"),
            ("cialign_plots",       "CIAlign plots  (open index.html)"),
            ("cleaned_consensus",   "CIAlign-cleaned consensus FASTAs"),
        ]),
        ("Primers", [
            ("selected_primers_summary.csv",   "Global primer summary"),
            ("cluster_top_primers.csv",        "Per-cluster top primers"),
            ("primer_genome_hits_summary.csv", "Genome-wide hit counts"),
        ]),
        ("Motif  (te_motif.py)", [
            ("motif_analysis/all_overlaps.tsv",           "bedtools overlap TSV"),
            ("motif_analysis/overall_top_motifs.png",     "Top-20 motifs bar chart"),
            ("enrichment_results/enrichment_heatmap.png", "Enrichment heatmap"),
        ]),
        ("GO annotation  (te_go.py)", [
            ("go_annotations/gene_functions.csv", "Gene → GO terms"),
        ]),
        ("Expression  (te_expression.py)", [
            ("expression_plots/boxplot_all.png",      "Boxplot — all data"),
            ("expression_plots/boxplot_mid80.png",    "Boxplot — mid 80%"),
            ("expression_plots/expression_stats.csv", "Per-cluster stats"),
        ]),
    ]

    for section_label, items in sections:
        print(_divider(section_label))
        for name, label in items:
            p = out_dir / name
            if not p.exists() and "/" not in name:
                for alt_dir in ("06_primers", "03_clustering", "01_data"):
                    alt = out_dir / alt_dir / name
                    if alt.exists():
                        p = alt
                        break
                if not p.exists() and name == "clustered_data.csv":
                    clustered = sorted((out_dir / "01_data").glob("*_clustered.csv"))
                    if clustered:
                        p = clustered[0]
            basename = name.split("/")[-1]
            print(f"  {_tick(p)}  {label:<46} {dim(basename)}")
        print()

    print(_divider("Checkpoints"))
    cp_path = out_dir / "enrichment_checkpoints.json"
    if cp_path.exists():
        import json
        cp = json.loads(cp_path.read_text())
        for step, info in cp.items():
            ts = info.get("completed_at", "")
            print(f"  {green('✔')}  {bold(step):<20} {dim(ts)}")
    else:
        print(f"  {dim('·')}  No checkpoint file found")
    print()

    cluster_csv = out_dir / "cluster_alignments" / "cluster_consensus_summary.csv"
    if cluster_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(cluster_csv)
            disp_cols = [c for c in ["cluster", "size", "consensus_length"] if c in df.columns]
            print(_divider(f"Clusters  ({len(df)})"))
            for _, row in df[disp_cols].iterrows():
                parts = [f"{c}: {bold(str(row[c]))}" for c in disp_cols]
                print(f"  {cyan('·')}  {'   '.join(parts)}")
            print()
        except Exception:
            pass


# ── HPC connection ────────────────────────────────────────────────────────────

def _connect(hostname: str = None, username: str = None, port: int = 22, key_path: str = "", password: str = None):
    """Prompt for HPC credentials, connect, and return an HPCClient."""
    from hpc_client import HPCClient

    print()
    print(_header("HPC Connection"))
    print()
    print(f"  {dim('All pipeline steps run on the remote cluster.')}")
    print(f"  {dim('Enter SSH credentials for your SLURM or LSF cluster.')}")
    print()

    if not hostname:
        hostname = _ask("Cluster hostname (e.g. login.cluster.edu)")
    else:
        print(f"  {cyan('›')} Hostname: {hostname}")

    if not hostname:
        print(red("  Hostname is required."))
        sys.exit(1)

    port_str = _ask("SSH port", str(port))
    try:
        port = int(port_str)
    except ValueError:
        port = 22

    if not username:
        username = _ask("Username")
    else:
        print(f"  {cyan('›')} Username: {username}")

    if not username:
        print(red("  Username is required."))
        sys.exit(1)

    if password is not None:
        print(f"  {cyan('›')} Password: {'*' * len(password)}")
    elif key_path:
        print(f"  {cyan('›')} Key file: {key_path}")
        password = getpass.getpass(f"  {cyan('›')} Password (leave blank if key needs no passphrase): ")
    else:
        password = getpass.getpass(f"  {cyan('›')} Password: ")

    print()
    client = HPCClient()
    if not client.connect(hostname, username, password, port, key_path=key_path):
        print(red("\n  Connection failed.  Check credentials and try again."))
        sys.exit(1)

    sched = (client.scheduler or "unknown").upper()
    print()
    print(_box([
        green("✔") + f"  Connected to  {bold(hostname)}",
        f"   User:       {bold(username)}",
        f"   Scheduler:  {bold(sched)}",
        f"   Home:       {dim(client.remote_work_dir)}",
    ]))
    print()
    client._hostname = hostname
    return client


# ── HPC-aware step helpers ────────────────────────────────────────────────────

def _remote_run(client, label: str, cmd: str, timeout: int = 1800, stream: bool = True,
                force_direct: bool = False):
    """Run a command on the cluster.

    If the client has a scheduler detected and force_direct is False, submits as
    a batch job (non-blocking, visible in squeue/bjobs). Otherwise runs directly
    over SSH with streamed output.

    Automatically prepends venv setup in both paths.
    """
    has_scheduler = bool(getattr(client, "scheduler", None))
    if has_scheduler and not force_direct:
        job_id = client.submit_command_as_batch_job(label, cmd)
        return bool(job_id)

    venv_prefix = client.ensure_venv_cmd() if hasattr(client, "ensure_venv_cmd") else ""
    full_cmd = f"{venv_prefix}\n{cmd}" if venv_prefix else cmd
    print()
    _log(f"Starting remote step: {label} (timeout={timeout}s)")
    print(f"  {dim('$')} {dim(cmd)}")
    print(_divider())
    out, err, code = client.run_command(full_cmd, timeout=timeout, stream_output=stream)
    if not stream:
        if out:
            print(out)
        if err:
            print(red(err))
    print(_divider())
    if code != 0:
        print(f"  {red('✘')} {bold(label)} exited with code {code}")
        _log(f"Remote step failed: {label} (exit={code})")
    else:
        print(f"  {green('✔')} {bold(label)} complete")
        _log(f"Remote step complete: {label}")
    print()
    return code == 0


def _sync_remote_files(client, names):
    """Upload local pipeline files needed by a direct remote step."""
    base = Path(__file__).parent
    # Always include requirements.txt so the remote venv can install deps
    all_names = list(names)
    if "requirements.txt" not in all_names:
        all_names.append("requirements.txt")
    _log(f"Syncing {len(all_names)} file(s) to {client.remote_work_dir}")
    for name in all_names:
        local_path = base / name
        if not local_path.exists():
            print(yellow(f"  Skipping missing local file: {name}"))
            continue
        remote_path = f"{client.remote_work_dir}/{name}"
        _log(f"Uploading {name} -> {remote_path}")
        if not client._upload_text_file(local_path, remote_path, name):
            print(red(f"  Could not upload {name}; remote step was not started."))
            return False
    _log("Remote file sync complete")
    return True


def _ensure_input_csv(client, prompt: str = "Input clustered CSV (remote path)") -> str:
    """Ask whether the user already has an input CSV; if not, offer to run te_prep first.

    Returns the remote path string, or "" if the user cancels.
    """
    print()
    has_csv = _confirm("Do you already have an input CSV on the cluster?", default=True)
    if has_csv:
        path = _ask(prompt)
        if not path.strip():
            print(red("  No path entered — cancelling."))
            return ""
        return path.strip()

    # Offer to run te_prep now
    print()
    print(f"  {yellow('›')} No input CSV — run te_prep.py first to download coordinates and extract sequences.")
    if not getattr(client, "has_internet", False):
        print(f"  {yellow('⚠')}  This node has no internet — UCSC sequence fetch will fail.")
        print(f"       A local genome FASTA (--genome) is required, or run te_prep on a login node.")
    run_prep = _confirm("Run te_prep now?", default=True)
    if not run_prep:
        print("  Skipping — re-run this step once you have a CSV.")
        return ""

    # Collect te_prep parameters
    species, build = _ask_species_assembly()
    family  = _ask("TE family name (e.g. HERVK, THE1D-int)")
    source  = _ask_source()
    genome  = _ask("Local assembly FASTA path on cluster (blank = none)", "")
    rmsk_dir = _ask("Pre-downloaded rmsk directory on cluster (blank = auto-download)", "")
    out_dir = _ask("Output directory (remote)", client.remote_work_dir + "/te_data")
    email   = _ask_notify_email(client)

    if not _sync_remote_files(client, ["te_prep.py"]):
        return ""

    cmd = (
        f"cd {client.remote_work_dir} && "
        f"{client._python} -u te_prep.py --build {shlex.quote(build)}"
    )
    if family:
        cmd += f" --family {shlex.quote(family)}"
    cmd += f" --source {source}"
    if genome:
        cmd += f" --genome-fa {shlex.quote(genome)}"
    if rmsk_dir:
        cmd += f" --rmsk-dir {shlex.quote(rmsk_dir)}"
    cmd += f" --base-dir {shlex.quote(out_dir)}"
    if email:
        cmd += f" --notify-email {shlex.quote(email)}"
    # force_direct=True: run on the login node (has internet) not a compute node
    if not _remote_run(client, "te_prep", cmd, timeout=1800, force_direct=True):
        print(red("  te_prep failed — cannot continue."))
        return ""

    # te_prep writes <base-dir>/<family>/<family>_sequences.csv
    guessed = f"{out_dir}/{family}/{family}_sequences.csv" if family else f"{out_dir}/sequences.csv"
    print(f"\n  {dim('Expected output:')} {guessed}")
    csv_path = _ask("Confirm input CSV path (edit if different)", guessed)
    return csv_path.strip()


def _step_prep(client):
    import shlex as _sl
    print()
    print(_divider("te_prep.py  ·  Download rmsk/dfam + extract sequences"))
    if not getattr(client, "has_internet", False):
        print(f"  {yellow('⚠')}  This node has no internet — UCSC sequence fetch will fail.")
        print(f"       A local genome FASTA (--genome) is required.")
    species, build = _ask_species_assembly()
    family  = _ask("TE family name (e.g. HERVK)")
    source  = _ask_source()
    genome  = _ask("Local assembly FASTA path on cluster (blank = none)", "")
    rmsk_dir = _ask("Pre-downloaded rmsk directory on cluster (blank = auto-download)", "")
    out_dir = _ask("Remote output directory", client.remote_work_dir + "/te_data")
    email   = _ask_notify_email(client)
    if not _sync_remote_files(client, ["te_prep.py"]):
        return
    cmd = (
        f"cd {_sl.quote(client.remote_work_dir)} && "
        f"{client._python} -u te_prep.py --build {_sl.quote(build)}"
    )
    if family:
        cmd += f" --family {_sl.quote(family)}"
    cmd += f" --source {source}"
    if genome:
        cmd += f" --genome-fa {_sl.quote(genome)}"
    if rmsk_dir:
        cmd += f" --rmsk-dir {_sl.quote(rmsk_dir)}"
    cmd += f" --base-dir {_sl.quote(out_dir)}"
    if email:
        cmd += f" --notify-email {_sl.quote(email)}"
    # force_direct=True: run on the login node (has internet) not a compute node
    _remote_run(client, "te_prep", cmd, timeout=1800, force_direct=True)


def _batch_submit(client, label, cmd, scripts):
    """Upload scripts and submit cmd as a batch job. Returns job_id or None."""
    if not _sync_remote_files(client, scripts):
        return None
    job_id = client.submit_command_as_batch_job(label, cmd)
    if job_id:
        client.current_job_id = job_id
        safe = re.sub(r"[^a-z0-9_]", "_", label.lower())[:20]
        work = client.remote_work_dir
        print(f"\n  {green('✓')} Job submitted  (ID: {bold(str(job_id))})")
        print(f"  {dim('·')} Track in scheduler:  squeue -j {job_id}")
        print(f"  {dim('·')} Output log:          {work}/{safe}_*.out")
        print(f"  {dim('·')} Error log:           {work}/{safe}_*.err")
        print(f"  {dim('·')} Monitor with option  {bold('11')}  or tail with  {bold('12')}")
    else:
        print(f"\n  {red('✗')} Submission failed — see [SUBMIT] log above.")
    return job_id


def _step_clustering(client):
    import shlex as _sl
    print()
    print(_divider("te_clustering.py  ·  k-mer + UMAP + HDBSCAN  [batch]"))
    inp = _ensure_input_csv(client, "Input sequences CSV (remote path)")
    if not inp:
        return
    out     = _ask("Output clustered CSV (remote)", inp.replace(".csv", "_clustered.csv") if inp.endswith(".csv") else inp + "_clustered.csv")
    out_dir = _ask("Visualization directory (remote)", client.remote_work_dir + "/results")
    kmer    = _ask("k-mer size", "6")
    family  = _ask("Family name", "FAMILY")
    email   = _ask_notify_email(client)
    cmd = (
        f"cd {_sl.quote(client.remote_work_dir)} && "
        f"{client._python} -u te_clustering.py "
        f"--input {_sl.quote(inp)} --output {_sl.quote(out)} "
        f"--kmer {_sl.quote(kmer)} --out-dir {_sl.quote(out_dir)} --family {_sl.quote(family)}"
    )
    if email:
        cmd += f" --notify-email {_sl.quote(email)}"
    _batch_submit(client, "te_clustering", cmd, ["te_clustering.py"])


def _step_alignment(client):
    import shlex as _sl
    print()
    print(_divider("te_alignment.py  ·  MAFFT + CIAlign + consensus  [batch]"))
    inp = _ensure_input_csv(client)
    if not inp:
        return
    out_dir = _ask("Output directory (remote)", client.remote_work_dir + "/results")
    family  = _ask("Family name", "FAMILY")
    skip_ci = _confirm("Skip CIAlign?", default=False)
    email   = _ask_notify_email(client)
    cmd = (
        f"cd {_sl.quote(client.remote_work_dir)} && "
        f"{client._python} -u te_alignment.py "
        f"--input {_sl.quote(inp)} --out-dir {_sl.quote(out_dir)} --family {_sl.quote(family)}"
    )
    if skip_ci:
        cmd += " --no-cialign"
    if email:
        cmd += f" --notify-email {_sl.quote(email)}"
    _batch_submit(client, "te_alignment", cmd, ["te_alignment.py"])


def _step_motif(client):
    import shlex as _sl
    print()
    print(_divider("te_clustering → te_motif.py  ·  Cluster then JASPAR + Fisher + HOMER  [batch]"))
    inp = _ensure_input_csv(client)
    if not inp:
        return

    # ── Clustering params (always run first) ──────────────────────────────────
    default_clustered = inp.replace(".csv", "_clustered.csv") if inp.endswith(".csv") else inp + "_clustered.csv"
    clustered_out = _ask("Clustered CSV output path (remote)", default_clustered)
    family  = _ask("Family name (for clustering)", "FAMILY")
    kmer    = _ask("k-mer size", "6")

    # ── Motif params ─────────────────────────────────────────────────────────
    species, build = _ask_species_assembly()
    out_dir = _ask("Output directory (remote)", client.remote_work_dir + "/results")

    jaspar = ""
    if _confirm("Provide a JASPAR BED file (remote path)?", default=False):
        jaspar = _ask("JASPAR BED path (remote)")

    run_homer    = _confirm("Also run HOMER known-motif enrichment per cluster?", default=False)
    homer_genome = ""
    homer_size   = "200"
    homer_threads = "4"
    if run_homer:
        homer_genome  = _ask("HOMER genome name or FASTA (blank = same as build)", "")
        homer_size    = _ask("HOMER -size value", "200")
        homer_threads = _ask("HOMER threads per cluster", "4")
    email = _ask_notify_email(client)

    cluster_cmd = (
        f"{client._python} -u te_clustering.py "
        f"--input {_sl.quote(inp)} --output {_sl.quote(clustered_out)} "
        f"--kmer {_sl.quote(kmer)} --out-dir {_sl.quote(out_dir)} --family {_sl.quote(family)}"
    )
    motif_cmd = (
        f"{client._python} -u te_motif.py "
        f"--input {_sl.quote(clustered_out)} --build {_sl.quote(build)} --out-dir {_sl.quote(out_dir)}"
    )
    if jaspar:
        motif_cmd += f" --jaspar-bed {_sl.quote(jaspar)}"
    if run_homer:
        motif_cmd += " --homer"
        if homer_genome.strip():
            motif_cmd += f" --homer-genome {_sl.quote(homer_genome.strip())}"
        motif_cmd += f" --homer-size {_sl.quote(homer_size)} --homer-threads {_sl.quote(homer_threads)}"
    if email:
        motif_cmd += f" --notify-email {_sl.quote(email)}"

    cmd = (
        f"cd {_sl.quote(client.remote_work_dir)} && "
        f"echo '[GAMECA] Step 1/2: clustering ...' && "
        f"{cluster_cmd} && "
        f"echo '[GAMECA] Step 2/2: motif analysis ...' && "
        f"{motif_cmd}"
    )
    _batch_submit(client, "te_clust_motif", cmd, ["te_clustering.py", "te_motif.py"])


def _step_go(client):
    import shlex as _sl
    print()
    print(_divider("te_go.py  ·  GO annotation via mygene.info  [batch]"))

    family = _ask("TE family name (e.g. HERVK, THE1D-int)")
    if not family:
        print(red("  Family name is required."))
        return

    species, build = _ask_species_assembly()
    out_dir = _ask("Output directory (remote)", client.remote_work_dir + "/results")
    enrich_dir = f"{out_dir}/enrichment_results"

    # Check if enrichment results already exist on the cluster
    check_out, _, _ = client.run_command(
        f"ls {_sl.quote(enrich_dir)}/cluster_*_enrichment.csv 2>/dev/null | wc -l",
        timeout=15,
    )
    enrich_exists = check_out.strip() not in ("", "0")

    if not enrich_exists:
        print(f"\n  {yellow('›')} No enrichment results found in {enrich_dir}")
        print(f"  {yellow('›')} Will run: te_clustering → te_motif → te_go")

        default_inp = f"{client.remote_work_dir}/te_data/{family}/{family}_sequences.csv"
        inp = _ask("Input sequences CSV (remote path)", default_inp)
        clustered_out = inp.replace(".csv", "_clustered.csv")
        email = _ask_notify_email(client)

        cluster_cmd = (
            f"{client._python} -u te_clustering.py "
            f"--input {_sl.quote(inp)} --output {_sl.quote(clustered_out)} "
            f"--kmer 6 --out-dir {_sl.quote(out_dir)} --family {_sl.quote(family)}"
        )
        motif_cmd = (
            f"{client._python} -u te_motif.py "
            f"--input {_sl.quote(clustered_out)} --build {_sl.quote(build)} "
            f"--out-dir {_sl.quote(out_dir)}"
        )
        go_cmd = (
            f"{client._python} -u te_go.py "
            f"--enrichment-dir {_sl.quote(enrich_dir)} "
            f"--build {_sl.quote(build)} --out-dir {_sl.quote(out_dir)} "
            f"--clustered-csv {_sl.quote(clustered_out)}"
        )
        if email:
            go_cmd += f" --notify-email {_sl.quote(email)}"
        cmd = (
            f"cd {_sl.quote(client.remote_work_dir)} && "
            f"echo '[GAMECA] Step 1/3: clustering ...' && {cluster_cmd} && "
            f"echo '[GAMECA] Step 2/3: motif analysis ...' && {motif_cmd} && "
            f"echo '[GAMECA] Step 3/3: GO annotation ...' && {go_cmd}"
        )
        _batch_submit(client, "te_clust_motif_go", cmd,
                      ["te_clustering.py", "te_motif.py", "te_go.py"])
    else:
        print(f"\n  {green('✔')} Enrichment results found — running GO annotation only")

        # Auto-detect clustered CSV for strand plots
        default_clustered = (
            f"{client.remote_work_dir}/te_data/{family}/{family}_sequences_clustered.csv"
        )
        exists_out, _, _ = client.run_command(
            f"test -f {_sl.quote(default_clustered)} && echo EXISTS || echo MISSING",
            timeout=10,
        )
        clustered = ""
        if "EXISTS" in exists_out:
            if _confirm(f"Include strand plots using detected clustered CSV?", default=True):
                clustered = default_clustered
        elif _confirm("Include clustered CSV for strand plots?", default=False):
            clustered = _ask("Clustered CSV (remote path)")
        email = _ask_notify_email(client)

        cmd = (
            f"cd {_sl.quote(client.remote_work_dir)} && "
            f"{client._python} -u te_go.py "
            f"--enrichment-dir {_sl.quote(enrich_dir)} "
            f"--build {_sl.quote(build)} --out-dir {_sl.quote(out_dir)}"
        )
        if clustered:
            cmd += f" --clustered-csv {_sl.quote(clustered)}"
        if email:
            cmd += f" --notify-email {_sl.quote(email)}"
        _batch_submit(client, "te_go", cmd, ["te_go.py"])


def _step_expression(client):
    import shlex as _sl
    print()
    print(_divider("te_expression.py  ·  Per-cluster expression boxplots  [batch]"))
    inp = _ensure_input_csv(client)
    if not inp:
        return
    out_dir = _ask("Output directory (remote)", client.remote_work_dir + "/results")
    email   = _ask_notify_email(client)
    cmd = (
        f"cd {_sl.quote(client.remote_work_dir)} && "
        f"{client._python} -u te_expression.py "
        f"--input {_sl.quote(inp)} --out-dir {_sl.quote(out_dir)}"
    )
    if email:
        cmd += f" --notify-email {_sl.quote(email)}"
    _batch_submit(client, "te_expression", cmd, ["te_expression.py"])


def _step_enrichment(client):
    import shlex as _sl
    print()
    print(_divider("te_clustering → te_enrichment.py  ·  Cluster then M+G+E  [batch]"))
    inp = _ensure_input_csv(client)
    if not inp:
        return

    # ── Clustering params ─────────────────────────────────────────────────────
    default_clustered = inp.replace(".csv", "_clustered.csv") if inp.endswith(".csv") else inp + "_clustered.csv"
    clustered_out = _ask("Clustered CSV output path (remote)", default_clustered)
    family  = _ask("Family name (for clustering)", "FAMILY")
    kmer    = _ask("k-mer size", "6")

    # ── Enrichment params ─────────────────────────────────────────────────────
    species, build = _ask_species_assembly()
    out_dir = _ask("Output directory (remote)", client.remote_work_dir + "/results")
    jaspar  = ""
    if _confirm("Provide a JASPAR BED file (remote path)?", default=False):
        jaspar = _ask("JASPAR BED path (remote)")
    skip_go   = _confirm("Skip GO annotation?",    default=False)
    skip_expr = _confirm("Skip expression plots?", default=False)
    email     = _ask_notify_email(client)

    cluster_cmd = (
        f"{client._python} -u te_clustering.py "
        f"--input {_sl.quote(inp)} --output {_sl.quote(clustered_out)} "
        f"--kmer {_sl.quote(kmer)} --out-dir {_sl.quote(out_dir)} --family {_sl.quote(family)}"
    )
    enrich_cmd = (
        f"{client._python} -u te_enrichment.py "
        f"--input {_sl.quote(clustered_out)} --build {_sl.quote(build)} --out-dir {_sl.quote(out_dir)}"
    )
    if jaspar:    enrich_cmd += f" --jaspar-bed {_sl.quote(jaspar)}"
    if skip_go:   enrich_cmd += " --skip-go"
    if skip_expr: enrich_cmd += " --skip-expression"
    if email:     enrich_cmd += f" --notify-email {_sl.quote(email)}"

    cmd = (
        f"cd {_sl.quote(client.remote_work_dir)} && "
        f"echo '[GAMECA] Step 1/2: clustering ...' && "
        f"{cluster_cmd} && "
        f"echo '[GAMECA] Step 2/2: enrichment analysis ...' && "
        f"{enrich_cmd}"
    )
    _batch_submit(client, "te_clust_enrich", cmd, [
        "te_clustering.py", "te_enrichment.py", "te_motif.py", "te_go.py", "te_expression.py"
    ])


# ── Remote file viewer ────────────────────────────────────────────────────────

def _file_viewer(client):
    """Interactive directory browser for the remote HPC cluster."""
    cwd = client.remote_work_dir

    _TEXT_EXTS = {
        ".py", ".sh", ".txt", ".log", ".out", ".err", ".csv", ".tsv",
        ".json", ".yaml", ".yml", ".md", ".fa", ".fasta", ".bed", ".gtf",
        ".gff", ".r", ".R", ".conf", ".cfg", ".ini", ".env",
    }

    def _ls(path):
        out, err, code = client.run_command(
            f"ls -la --time-style=long-iso {path!r} 2>&1", timeout=15
        )
        if code != 0 or not out.strip():
            return None, err or "Could not list directory."
        entries = []
        for line in out.splitlines():
            if line.startswith("total") or not line.strip():
                continue
            parts = line.split(None, 8)
            if len(parts) < 9:
                continue
            perms, _, _, _, size, date, time_, *rest = parts
            name = rest[0] if rest else ""
            if name in (".", ".."):
                continue
            is_dir  = perms.startswith("d")
            is_link = perms.startswith("l")
            link_target = ""
            if is_link and " -> " in name:
                name, link_target = name.split(" -> ", 1)
            entries.append({
                "name": name,
                "perms": perms,
                "size": size,
                "date": date,
                "time": time_,
                "is_dir": is_dir,
                "is_link": is_link,
                "link_target": link_target,
            })
        entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
        return entries, None

    def _print_listing(path, entries):
        os.system("clear" if sys.stdout.isatty() else ":")
        print()
        print(_header("Remote File Viewer"))
        print()
        print(f"  {dim('Location:')}  {bold(path)}")
        print()
        print(_divider("Contents"))
        print(f"  {'#':<5} {'Name':<42} {'Size':>8}  {'Date':<10}  {'Time':<5}  {'Perms'}")
        print(f"  {dim('─' * 80)}")
        for i, e in enumerate(entries, 1):
            icon   = (blue("d") if e["is_dir"] else (cyan("l") if e["is_link"] else dim("f")))
            name_s = (bold(blue(e["name"])) if e["is_dir"]
                      else (cyan(e["name"]) if e["is_link"]
                            else e["name"]))
            link_s = f"  {dim('→')} {dim(e['link_target'])}" if e["link_target"] else ""
            size_s = dim(e["size"]) if e["is_dir"] else e["size"]
            print(f"  {dim(str(i)):<5} {icon} {name_s:<40}{link_s}  {size_s:>8}  {e['date']}  {e['time']}  {dim(e['perms'])}")
        print()
        print(_divider("Commands"))
        print(f"  {cyan('<number>')}  enter dir / view file    "
              f"  {cyan('..')}  up    "
              f"  {cyan('~')}  home    "
              f"  {cyan('cd <path>')}  jump")
        print(f"  {cyan('pwd')}  print path    "
              f"  {cyan('q')}  quit viewer")
        print()

    def _view_file(path, name):
        ext = Path(name).suffix.lower()
        size_out, _, _ = client.run_command(
            f"stat -c%s {path!r} 2>/dev/null || wc -c < {path!r}", timeout=10
        )
        try:
            byte_size = int(size_out.strip().split()[-1])
        except (ValueError, IndexError):
            byte_size = 0

        print()
        if ext not in _TEXT_EXTS and byte_size > 0:
            print(f"  {yellow('Binary or unknown file type')}  ({byte_size:,} bytes)")
            print(f"  {dim('Showing first 20 lines as text anyway (may be garbled).')}")
        else:
            print(f"  {dim('File:')}  {bold(name)}  {dim(f'({byte_size:,} bytes)')}")

        print(_divider())
        head_out, _, code = client.run_command(
            f"head -n 60 {path!r} 2>&1", timeout=15
        )
        if code != 0 or not head_out.strip():
            print(dim("  (empty or unreadable)"))
        else:
            for line in head_out.splitlines():
                print(f"  {line}")
        if byte_size > 4096:
            line_count_out, _, _ = client.run_command(
                f"wc -l < {path!r} 2>/dev/null", timeout=10
            )
            try:
                total_lines = int(line_count_out.strip())
                if total_lines > 60:
                    print()
                    print(dim(f"  … {total_lines - 60:,} more lines not shown …"))
            except (ValueError, IndexError):
                pass
        print(_divider())
        input(f"  {dim('Press Enter to return…')}")

    while True:
        entries, err = _ls(cwd)
        if entries is None:
            print(red(f"\n  {err}"))
            cwd = client.remote_work_dir
            input(f"  {dim('Press Enter to reset to home…')}")
            continue

        _print_listing(cwd, entries)
        raw = input(f"  {cyan('›')} ").strip()

        if not raw or raw.lower() == "q":
            break

        if raw == "..":
            parent = str(Path(cwd).parent)
            if parent != cwd:
                cwd = parent
            continue

        if raw in ("~", "home"):
            cwd = client.remote_work_dir
            continue

        if raw.lower() == "pwd":
            print(f"\n  {bold(cwd)}")
            input(f"  {dim('Press Enter…')}")
            continue

        if raw.lower().startswith("cd "):
            target = raw[3:].strip()
            if not target.startswith("/"):
                target = f"{cwd}/{target}"
            test_out, _, code = client.run_command(
                f"test -d {target!r} && echo OK || echo NOTDIR", timeout=10
            )
            if "OK" in test_out:
                cwd = target
            else:
                print(red(f"  Not a directory: {target}"))
                input(f"  {dim('Press Enter…')}")
            continue

        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(entries):
                entry = entries[idx]
                target = f"{cwd}/{entry['name']}"
                if entry["is_dir"]:
                    cwd = target
                else:
                    _view_file(target, entry["name"])
            else:
                print(red("  Number out of range."))
                input(f"  {dim('Press Enter…')}")
            continue

        print(red(f"  Unknown command: {raw!r}"))
        input(f"  {dim('Press Enter…')}")


# ── Interactive menu ──────────────────────────────────────────────────────────

MENU = [
    ("1",  "Workflow overview",                           "info"),
    (None, "Data prep",                                   "section"),
    ("0",  "Query RepeatMasker  (download rmsk + count sequences)", "action"),
    ("20", "Query Dfam  (download nrph hits + count sequences)",    "action"),
    ("2",  "te_prep.py       Download rmsk + sequences",  "action"),
    (None, "Core analysis  (direct remote execution)",    "section"),
    ("3",  "te_clustering.py k-mer · UMAP · HDBSCAN",    "action"),
    ("4",  "te_alignment.py  MAFFT · CIAlign · consensus","action"),
    (None, "Enrichment",                                  "section"),
    ("5",  "te_motif.py      JASPAR + Fisher + HOMER",      "action"),
    ("6",  "te_go.py         GO annotation (mygene.info)","action"),
    ("7",  "te_expression.py Expression boxplots",        "action"),
    ("8",  "te_enrichment.py All enrichment  (M+G+E)",   "action"),
    (None, "Batch / full pipeline  (scheduler)",          "section"),
    ("9",  "Configure + submit batch job  (bsub/sbatch)", "action"),
    ("16", "Submit motif+GO batch job  (from BED file)",  "action"),
    ("11", "Check batch job status",                      "action"),
    ("12", "Watch batch job  (live tail)",                "action"),
    (None, "Results",                                     "section"),
    ("13", "Retrieve results to local machine",           "action"),
    ("14", "View local results summary",                  "action"),
    (None, "Files",                                       "section"),
    ("18", "Browse remote filesystem",                    "action"),
    (None, "Session",                                     "section"),
    ("19", "Diagnose job submission  (test nohup + sbatch)","action"),
    ("17", "Test email  (set sender / receiver / password)","action"),
    ("21", "Test email notification  (submit batch job → auto-email on finish)","action"),
    ("15", "Disconnect and exit",                         "action"),
]


_preserve_output = False  # set True to skip the next screen clear


def _print_menu(client):
    global _preserve_output
    if _preserve_output:
        _preserve_output = False
        print("\n" + dim("─" * 66) + "\n")
    else:
        os.system("clear" if sys.stdout.isatty() else ":")
    print()
    print(cyan(GAMECA_ART.rstrip()))
    print(f"  {dim(TAGLINE)}")
    print()
    print(HPC_NOTICE)
    print()
    sched = (client.scheduler or "?").upper()
    host  = getattr(client, "_hostname", "cluster")
    net_indicator = green("internet ✓") if getattr(client, "has_internet", False) else red("no internet ✗")
    print(f"  {green('●')}  Connected  {bold(host)}  {dim('·')}  scheduler: {bold(sched)}  {dim('·')}  {net_indicator}")
    print()
    print(_divider("Menu"))
    for key, label, kind in MENU:
        if kind == "section":
            print(f"\n  {dim('┄' * 3)}  {bold(label)}")
        elif kind == "info":
            print(f"    {cyan(f'[{key}]')}  {dim(label)}")
        else:
            num = f"[{key}]"
            print(f"    {cyan(num):<14}  {label}")
    print()


def _pause(keep_output: bool = False):
    """Pause with 'Press Enter' and optionally suppress the next screen clear.
    Never raises — safe to call from finally blocks."""
    global _preserve_output
    if keep_output:
        _preserve_output = True
    print()
    try:
        input(f"  {dim('── Press Enter to return to menu ──')}")
    except (EOFError, OSError):
        pass




def interactive_menu(client):
    import traceback as _tb
    while True:
        _print_menu(client)
        choice = input(f"  {cyan('›')} {bold('Select')}: ").strip()

        try:
            if choice == '1':
                print()
                print(_header("GAMECA Pipeline  ·  Workflow Overview"))
                print(WORKFLOW)
                _pause()

            elif choice == '0':
                _step_rmsk_query(client)
                _pause()

            elif choice == '20':
                _step_dfam_query(client)
                _pause()

            elif choice == '2':
                _step_prep(client)
                _pause(keep_output=True)
            elif choice == '3':
                _step_clustering(client)
                _pause(keep_output=True)
            elif choice == '4':
                _step_alignment(client)
                _pause(keep_output=True)
            elif choice == '5':
                _step_motif(client)
                _pause(keep_output=True)
            elif choice == '6':
                _step_go(client)
                _pause(keep_output=True)
            elif choice == '7':
                _step_expression(client)
                _pause(keep_output=True)
            elif choice == '8':
                _step_enrichment(client)
                _pause(keep_output=True)

            elif choice == '9':
                if client.set_parameter_interactive():
                    client.submit_batch_job()
                _pause(keep_output=True)

            elif choice == '16':
                try:
                    client.submit_motif_batch_job()
                except Exception:
                    print()
                    print(red("  ── Motif submission error ───────────────────────────"))
                    _tb.print_exc()
                    print(red("  ────────────────────────────────────────────────────"))
                finally:
                    _pause(keep_output=True)

            elif choice == '11':
                client.check_job_status()
                _pause(keep_output=True)

            elif choice == '12':
                client.watch_job()
                _pause()

            elif choice == '13':
                cached_remote = client._state.get("last_remote_dir", "").strip()
                remote_override = _ask("Remote path (leave blank to auto-detect)", cached_remote) or cached_remote or None

                default_local = str(client.local_output_dir) if client.local_output_dir else "./hpc_results"
                local_dir = _ask("Local output directory", default_local)
                if local_dir:
                    client.retrieve_results(local_dir, remote_out_override=remote_override)
                _pause()

            elif choice == '14':
                out_dir = _ask("Local results directory", "./results")
                display_results(Path(out_dir))
                _pause()

            elif choice == '18':
                _file_viewer(client)

            elif choice == '19':
                client.diagnose_job_submission()
                _pause(keep_output=True)

            elif choice == '17':
                client.test_email_interactive()
                _pause()

            elif choice == '21':
                client.submit_email_test_job()
                _pause(keep_output=True)

            elif choice == '15':
                print(f"\n  {green('Disconnecting…')}")
                client.disconnect()
                print(f"  {green('Goodbye!')}  {dim('— GAMECA')}\n")
                sys.exit(0)

            else:
                print(red("  Invalid selection — please choose a number from the menu."))
                import time; time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n  {yellow('Interrupted.')}")
            _pause(keep_output=True)
        except Exception:
            print()
            print(red("  ── Unhandled error ─────────────────────────────────────"))
            _tb.print_exc()
            print(red("  ────────────────────────────────────────────────────────"))
            _pause(keep_output=True)


# ── Local mode ────────────────────────────────────────────────────────────────

def _step_local_run():
    """Configure and run the full local pipeline without HPC."""
    print()
    print(_divider("Local Pipeline  ·  auto-downloads rmsk, fetches seqs via UCSC"))

    family = _ask("TE family repName (e.g. THE1D-int, HERVK9)")
    if not family:
        print(red("  Family name is required."))
        return None, None

    species, assembly = _ask_species_assembly()
    source    = _ask_source()
    max_loci  = _ask("Max loci to analyse (blank = all)", "")
    genome    = _ask("Path to local assembly FASTA (blank = UCSC API for sequences)", "")
    expr_csv  = _ask("Expression assembly CSV/TSV/BED with chr/start/stop (blank = none)", "")
    expr_buf  = ""
    if expr_csv.strip():
        expr_buf = _ask("Expression coordinate buffer bp", "50")
    workers   = _ask("Parallel UCSC fetch workers", "3")
    out_root  = _ask("Output directory", "results")

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "query.py"),
        "--local",
        "--family",   family,
        "--assembly", assembly,
        "--source",   source,
        "--output",   out_root,
        "--fetch-workers", workers or "3",
    ]

    if genome.strip():
        cmd += ["--genome", genome.strip()]
    else:
        cmd.append("--skip-genome")   # no local FASTA → skip genome-wide primer search

    if max_loci.strip():
        cmd += ["--max-loci", max_loci.strip()]

    if expr_csv.strip():
        cmd += ["--expression-assembly", expr_csv.strip()]
        if expr_buf.strip():
            cmd += ["--expression-buffer", expr_buf.strip()]

    print()
    print(f"  {dim('$')} {' '.join(cmd)}")
    print(_divider())

    import subprocess
    ret = subprocess.run(cmd)

    out_dir = Path(out_root) / family.lower()
    if ret.returncode == 0 and out_dir.exists():
        print(f"\n  {green('✔')} Pipeline complete → {bold(str(out_dir))}")
        input(f"  {dim('Press Enter to continue…')}")
        return str(out_dir), family
    else:
        print(f"\n  {red('✘')} Pipeline failed (exit code {ret.returncode})")
        input(f"  {dim('Press Enter to continue…')}")
        return None, None


def _find_clustered_csv(results_dir, family=None):
    """Return a clustered CSV from a result root, family dir, or CSV path."""
    results_dir = Path(results_dir).expanduser()

    if results_dir.is_file():
        if results_dir.suffix.lower() == ".csv":
            return results_dir
        return None

    if not results_dir.exists():
        return None

    if family:
        fam_lo = family.lower()
        fam_key = re.sub(r"[^a-z0-9]", "", fam_lo)
        # Standard output layout from query.py
        candidate = results_dir / fam_lo / "01_data" / f"{fam_lo}_clustered.csv"
        if candidate.exists():
            return candidate
        candidate = results_dir / "01_data" / f"{fam_lo}_clustered.csv"
        if candidate.exists():
            return candidate
        # Also try with hyphens replaced by underscores, etc.
        for p in sorted(results_dir.glob("**/*_clustered.csv")):
            stem_key = re.sub(r"[^a-z0-9]", "", p.stem.lower())
            if fam_key in stem_key:
                return p

    # Fall back: any clustered CSV under results_dir, including files stored
    # directly in presentation/test folders rather than query.py's 01_data dir.
    candidates = sorted(results_dir.glob("**/*_clustered.csv"))
    return candidates[0] if candidates else None


def _launch_local_dashboard(results_dir=None, family=None, open_browser=True):
    """Launch an interactive Dash clustering dashboard.

    Features:
    - UMAP/PCA scatter with rich hover (coordinates, locus, cluster, GC, length)
    - Lasso/box selection → re-cluster the selected subset
    - Upload an indices file (CSV / TXT) to specify a subset
    - Type a new sequence → appended to the dataset before re-clustering
    - min_cluster_size = max(2, N // 5) applied automatically
    - Reset button restores the original dataset
    """
    # ── Dependency check ──────────────────────────────────────────────────────
    try:
        import dash
        from dash import Dash, dcc, html, Input, Output, State, ALL, callback_context, no_update
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print(red("\n  'dash' is not installed."))
        print(f"  Run:  {cyan('pip install dash plotly')}")
        input(f"\n  {dim('Press Enter to return…')}")
        return

    import pandas as pd
    import base64
    import io
    import threading
    import time as _t
    import tempfile
    import uuid
    import json
    from flask import send_from_directory

    # ── Locate data ───────────────────────────────────────────────────────────
    if not results_dir:
        results_dir = _ask("Local results directory", "./results")
    if not family:
        family = _ask("Family name (leave blank to auto-detect)", "")

    csv_path = _find_clustered_csv(results_dir, family or None)
    if csv_path is None:
        print(red(f"\n  No *_clustered.csv found under {results_dir}."))
        print(f"  Run the local pipeline first (option 1 in this menu).")
        input(f"\n  {dim('Press Enter to return…')}")
        return

    if not family:
        family = csv_path.stem.replace("_clustered", "").replace("_", " ").title()

    print(f"\n  {green('●')} Loading: {csv_path}")
    _log(f"Dashboard input CSV: {csv_path}")
    df_orig = pd.read_csv(csv_path)
    print(f"  {len(df_orig)} sequences, columns: {list(df_orig.columns)[:8]}…")
    _log(f"Loaded dashboard dataframe rows={len(df_orig):,} cols={len(df_orig.columns):,}")

    # ── Choose embedding ──────────────────────────────────────────────────────
    has_umap = {"umap_x", "umap_y"}.issubset(df_orig.columns)
    has_pca  = {"pca_x",  "pca_y" }.issubset(df_orig.columns)
    if not has_umap and not has_pca:
        print(red("  No UMAP/PCA embedding columns found.  Run clustering first."))
        input(f"  {dim('Press Enter to return…')}")
        return

    x_col, y_col = ("umap_x", "umap_y") if has_umap else ("pca_x", "pca_y")
    embed_name   = "UMAP" if has_umap else "PCA"
    _log(f"Using {embed_name} embedding columns: {x_col}, {y_col}")

    # ── Feature detection ─────────────────────────────────────────────────────
    chr_col = next((c for c in ("chr", "Chromosome", "chrom", "Chr") if c in df_orig.columns), None)
    strand_col = next((c for c in ("strand", "Strand") if c in df_orig.columns), None)
    excluded_numeric = {
        "start", "stop", "Start", "Stop", "End", "Cluster", "pca_x", "pca_y",
        "umap_x", "umap_y", "tsne_x", "tsne_y", "swScore", "milliDiv",
        "Unnamed: 0",
    }
    expr_cols = [
        c for c in df_orig.columns
        if c not in excluded_numeric and pd.api.types.is_numeric_dtype(df_orig[c])
    ]
    _log(
        "Detected feature columns: "
        f"chromosome={chr_col or 'none'}, strand={strand_col or 'none'}, "
        f"expression={len(expr_cols)}"
    )

    def _dropdown_options(values):
        return [{"label": str(v), "value": str(v)} for v in values]

    def _feature_match_mask(df, chromosomes=None, strands=None, expr_col=None, expr_band=None):
        mask = pd.Series([True] * len(df), index=df.index)
        active = []

        if chr_col and chromosomes:
            vals = {str(v) for v in chromosomes}
            mask &= df[chr_col].astype(str).isin(vals)
            active.append(f"{len(vals)} chromosome filter{'s' if len(vals) != 1 else ''}")

        if strand_col and strands:
            vals = {str(v) for v in strands}
            mask &= df[strand_col].astype(str).isin(vals)
            active.append(f"{len(vals)} strand filter{'s' if len(vals) != 1 else ''}")

        if expr_col and expr_col in df.columns and expr_band:
            vals = pd.to_numeric(df[expr_col], errors="coerce")
            if expr_band == "nonzero":
                mask &= vals.fillna(0) > 0
                active.append(f"{expr_col} > 0")
            elif expr_band in {"low", "mid", "high"}:
                q1, q2 = vals.quantile([0.33, 0.67])
                if expr_band == "low":
                    mask &= vals <= q1
                    active.append(f"low {expr_col}")
                elif expr_band == "mid":
                    mask &= (vals > q1) & (vals < q2)
                    active.append(f"mid {expr_col}")
                else:
                    mask &= vals >= q2
                    active.append(f"high {expr_col}")

        if not active:
            return pd.Series([False] * len(df), index=df.index), []
        return mask.fillna(False), active

    _state = {"df": df_orig.copy()}

    # ── Figure factory ────────────────────────────────────────────────────────
    _COLORS = (px.colors.qualitative.Set2
               + px.colors.qualitative.Plotly
               + px.colors.qualitative.Pastel)

    def _make_fig(df, feature_filters=None, feature_mode="highlight"):
        feature_filters = feature_filters or {}
        feature_mask, active_filters = _feature_match_mask(df, **feature_filters)
        matched_indices = set(df.index[feature_mask].tolist())
        clusters = sorted(df["Cluster"].unique()) if "Cluster" in df.columns else [0]
        n        = len(df)
        mcs      = max(2, n // 5)

        # Build per-row hover text
        hover = []
        for i, row in df.iterrows():
            parts = [f"<b>Row {int(i)}</b>"]
            for col in ("TE_name", "repName"):
                if col in row and pd.notna(row[col]):
                    parts.append(f"ID: {str(row[col])[:50]}")
                    break
            locus_parts = []
            for col in ("chr", "Chromosome"):
                if col in row and pd.notna(row[col]):
                    locus_parts.append(str(row[col]))
                    break
            for col in ("start", "Start"):
                if col in row and pd.notna(row[col]):
                    locus_parts.append(str(int(row[col])))
                    break
            for col in ("stop", "Stop"):
                if col in row and pd.notna(row[col]):
                    locus_parts.append(str(int(row[col])))
                    break
            if len(locus_parts) == 3:
                parts.append(f"Locus: {locus_parts[0]}:{locus_parts[1]}-{locus_parts[2]}")
            cl = int(row["Cluster"]) if "Cluster" in row else 0
            parts.append(f"Cluster: {'noise' if cl == -1 else cl}")
            parts.append(f"{embed_name}: ({row[x_col]:.4f}, {row[y_col]:.4f})")
            if "Seq" in row and isinstance(row["Seq"], str) and len(row["Seq"]) > 0:
                seq = str(row["Seq"])
                gc  = (seq.count("G") + seq.count("C")) / len(seq)
                parts.append(f"Length: {len(seq):,} bp  |  GC: {gc*100:.1f}%")
            hover.append("<br>".join(parts))

        fig  = go.Figure()
        cidx = {cl: i for i, cl in enumerate(clusters)}

        for cl in clusters:
            mask = (df["Cluster"] == cl) if "Cluster" in df.columns else pd.Series([True]*n)
            sub  = df[mask]
            selectedpoints = None
            if matched_indices and feature_mode == "select":
                selectedpoints = [
                    pos for pos, idx in enumerate(sub.index.tolist())
                    if idx in matched_indices
                ]
            fig.add_trace(go.Scatter(
                x=sub[x_col].tolist(),
                y=sub[y_col].tolist(),
                mode="markers",
                name="Noise" if cl == -1 else f"Cluster {cl}",
                marker=dict(
                    size=9,
                    color=("#cccccc" if cl == -1
                           else _COLORS[cidx[cl] % len(_COLORS)]),
                    opacity=0.82,
                    line=dict(width=0.6, color="white"),
                ),
                text=[hover[df.index.get_loc(i)] for i in sub.index],
                hovertemplate="%{text}<extra></extra>",
                customdata=sub.index.tolist(),
                selectedpoints=selectedpoints,
                selected=dict(marker=dict(size=13, opacity=1.0)),
                unselected=dict(marker=dict(opacity=0.25)),
            ))

        if matched_indices and feature_mode == "highlight":
            hi = df.loc[list(matched_indices)]
            fig.add_trace(go.Scatter(
                x=hi[x_col].tolist(),
                y=hi[y_col].tolist(),
                mode="markers",
                name="Feature match",
                marker=dict(
                    size=15,
                    color="rgba(255, 177, 38, 0.25)",
                    line=dict(width=2.4, color="#d35400"),
                    symbol="circle-open",
                ),
                text=[hover[df.index.get_loc(i)] for i in hi.index],
                hovertemplate="%{text}<extra></extra>",
                customdata=hi.index.tolist(),
            ))

        n_cl = len([c for c in clusters if c >= 0])
        filter_label = ""
        if active_filters:
            filter_label = f"  ·  {int(feature_mask.sum())} feature match(es)"
        fig.update_layout(
            title=dict(
                text=(f"{family}  ·  {embed_name}  ·  "
                      f"{n} sequences  ·  {n_cl} cluster(s)  ·  "
                      f"min_cluster_size = {mcs}{filter_label}"),
                font=dict(size=14, color="#333"),
            ),
            xaxis_title=f"{embed_name} 1",
            yaxis_title=f"{embed_name} 2",
            hovermode="closest",
            dragmode="lasso",
            height=640,
            plot_bgcolor="#f8f9fa",
            paper_bgcolor="white",
            margin=dict(l=40, r=16, t=64, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=12)),
        )
        return fig

    def _cluster_counts(df):
        if "Cluster" not in df.columns:
            return 0, 0
        n_clusters = len([c for c in df["Cluster"].unique() if c >= 0])
        n_noise = int((df["Cluster"] == -1).sum())
        return n_clusters, n_noise

    def _header_title():
        return f"GAMECA  ·  {family}  ·  Interactive Clustering"

    def _header_subtitle(df, source_label="original dataset"):
        n_clusters, n_noise = _cluster_counts(df)
        mcs = max(2, len(df) // 5)
        return (
            f"{embed_name} embedding  |  {len(df):,} sequences  |  "
            f"{n_clusters} cluster(s)  |  {n_noise} noise  |  "
            f"min_cluster_size = {mcs}  |  {source_label}"
        )

    def _bottom_info(df, source_label="original dataset", added_count=0):
        n_clusters, n_noise = _cluster_counts(df)
        mcs = max(2, len(df) // 5)
        added = f"  |  added sequences: {added_count:,}" if added_count else ""
        return (
            f"rows = {len(df):,}  |  clusters = {n_clusters}  |  noise = {n_noise:,}\n"
            f"min_cluster_size = {mcs}  |  source = {source_label}{added}"
        )

    def _tab_links(active_tab=None):
        tabs = _state.get("tabs", [])
        if not tabs:
            return html.Div("No recluster tabs yet.", style=_HINT)
        return html.Div([
            html.Div([
                html.Button(
                    t["label"],
                    id={"type": "cluster-tab", "tab_id": t["id"]},
                    n_clicks=0,
                    title="Show this recluster result",
                    style={
                        "border": "none",
                        "background": "transparent",
                        "color": "#1f6fb2",
                        "cursor": "pointer",
                        "fontSize": "11px",
                        "padding": "0 4px 0 0",
                    },
                ),
                html.Button(
                    "x",
                    id={"type": "close-tab", "tab_id": t["id"]},
                    n_clicks=0,
                    title="Close tab",
                    style={
                        "border": "none",
                        "background": "transparent",
                        "color": "#777",
                        "cursor": "pointer",
                        "fontSize": "11px",
                        "padding": "0 0 0 4px",
                    },
                ),
                html.Button(
                    "<",
                    id={"type": "move-tab-left", "tab_id": t["id"]},
                    n_clicks=0,
                    title="Move tab left",
                    style={
                        "border": "none",
                        "background": "transparent",
                        "color": "#777",
                        "cursor": "pointer",
                        "fontSize": "11px",
                        "padding": "0 0 0 6px",
                    },
                ),
                html.Button(
                    ">",
                    id={"type": "move-tab-right", "tab_id": t["id"]},
                    n_clicks=0,
                    title="Move tab right",
                    style={
                        "border": "none",
                        "background": "transparent",
                        "color": "#777",
                        "cursor": "pointer",
                        "fontSize": "11px",
                        "padding": "0 0 0 2px",
                    },
                ),
            ],
                style={
                    "display": "inline-flex",
                    "alignItems": "center",
                    "padding": "5px 8px",
                    "margin": "0 6px 6px 0",
                    "border": ("1px solid #4C9BE8"
                               if t["id"] == active_tab else "1px solid #b8d7f6"),
                    "borderRadius": "4px",
                    "background": ("#e6f2ff" if t["id"] == active_tab else "#f3f9ff"),
                },
            )
            for t in tabs
        ])

    def _tab_content(active_tab=None):
        tabs = _state.get("tabs", [])
        if not tabs:
            return ""
        active = next((t for t in tabs if t["id"] == active_tab), tabs[-1])
        return (
            f"Showing {active['label']}. Use the tab buttons above to move between "
            "recluster results, < and > to reorder tabs, or x to close a tab."
        )

    # ── App layout ────────────────────────────────────────────────────────────
    _BTN = {
        "width": "100%", "padding": "10px 0",
        "border": "none", "borderRadius": "5px",
        "fontSize": "13px", "cursor": "pointer",
        "marginBottom": "8px",
    }
    _LABEL = {"fontSize": "12px", "fontWeight": "bold",
               "display": "block", "marginBottom": "4px", "color": "#555"}
    _HINT  = {"fontSize": "11px", "color": "#999", "marginTop": "2px"}

    HOST = os.environ.get("GAMECA_HOST", "127.0.0.1")
    try:
        PORT = int(os.environ.get("GAMECA_PORT", "8765"))
    except ValueError:
        PORT = 8765
    tab_dir = Path(tempfile.gettempdir()) / f"gameca_recluster_tabs_{os.getpid()}"
    tab_dir.mkdir(parents=True, exist_ok=True)

    app = Dash(__name__, suppress_callback_exceptions=True)

    @app.server.route("/recluster-tabs/<path:filename>")
    def _serve_recluster_tab(filename):
        return send_from_directory(tab_dir, filename)

    _state["tabs"] = []
    _state["active_tab"] = None

    app.layout = html.Div([
        dcc.Store(id="open-tab-url"),
        html.Div(id="open-tab-sink", style={"display": "none"}),

        # ── Header ────────────────────────────────────────────────────────────
        html.Div([
            html.H3(id="header-title",
                    children=_header_title(),
                    style={"margin": "0 0 2px", "color": "#222", "fontWeight": "600"}),
            html.P(
                id="header-subtitle",
                children=_header_subtitle(_state["df"]),
                style={"margin": 0, "fontSize": "12px", "color": "#888"}),
            html.Div(id="tab-list", children=_tab_links(),
                     style={"marginTop": "8px", "minHeight": "26px"}),
            html.Div(id="tab-content", children=_tab_content(),
                     style={"fontSize": "11px", "color": "#777", "minHeight": "16px"}),
        ], style={"padding": "14px 20px 10px",
                   "borderBottom": "2px solid #e8e8e8",
                   "background": "white"}),

        # ── Body: scatter + control panel ─────────────────────────────────────
        html.Div([

            # Scatter (grows to fill space)
            html.Div([
                dcc.Graph(
                    id="scatter",
                    figure=_make_fig(_state["df"]),
                    config={
                        "scrollZoom": True,
                        "displayModeBar": True,
                        "modeBarButtonsToAdd": ["lasso2d", "select2d"],
                        "modeBarButtonsToRemove": ["autoScale2d"],
                        "toImageButtonOptions": {
                            "format": "png", "filename": f"{family}_clustering",
                        },
                    },
                    style={"height": "100%"},
                ),
            ], style={"flex": "1 1 auto", "minWidth": 0, "height": "100%"}),

            # Control panel (fixed width)
            html.Div([

                # ── Selection status ─────────────────────────────────────────
                html.Div(id="sel-status",
                         children="Lasso points on the plot or upload indices.",
                         style={"fontSize": "12px", "color": "#4C9BE8",
                                "padding": "6px 0 10px", "minHeight": "32px"}),

                html.Hr(style={"margin": "0 0 12px"}),

                # ── Core feature filters ─────────────────────────────────────
                html.Label("Core feature options:", style=_LABEL),
                dcc.Dropdown(
                    id="feature-chromosomes",
                    options=(_dropdown_options(sorted(df_orig[chr_col].dropna().astype(str).unique()))
                             if chr_col else []),
                    placeholder=("Chromosome" if chr_col else "No chromosome column"),
                    multi=True,
                    disabled=chr_col is None,
                    style={"fontSize": "12px", "marginBottom": "7px"},
                ),
                dcc.Dropdown(
                    id="feature-strands",
                    options=(_dropdown_options(sorted(df_orig[strand_col].dropna().astype(str).unique()))
                             if strand_col else []),
                    placeholder=("Strand" if strand_col else "No strand column"),
                    multi=True,
                    disabled=strand_col is None,
                    style={"fontSize": "12px", "marginBottom": "7px"},
                ),
                dcc.Dropdown(
                    id="feature-expression",
                    options=[{"label": c, "value": c} for c in expr_cols],
                    placeholder=("Expression column" if expr_cols else "No expression columns"),
                    disabled=not expr_cols,
                    style={"fontSize": "12px", "marginBottom": "7px"},
                ),
                dcc.Dropdown(
                    id="feature-expression-band",
                    options=[
                        {"label": "High expression", "value": "high"},
                        {"label": "Middle expression", "value": "mid"},
                        {"label": "Low expression", "value": "low"},
                        {"label": "Non-zero expression", "value": "nonzero"},
                    ],
                    placeholder="Expression range",
                    disabled=not expr_cols,
                    style={"fontSize": "12px", "marginBottom": "8px"},
                ),
                dcc.RadioItems(
                    id="feature-mode",
                    options=[
                        {"label": "Highlight on page", "value": "highlight"},
                        {"label": "Select matching points", "value": "select"},
                    ],
                    value="highlight",
                    labelStyle={"display": "block", "fontSize": "12px", "marginBottom": "4px"},
                    inputStyle={"marginRight": "6px"},
                ),
                html.Div(id="feature-status", style=_HINT),

                html.Hr(style={"margin": "12px 0"}),

                # ── Upload indices ───────────────────────────────────────────
                html.Label("Upload indices (CSV / TXT):", style=_LABEL),
                dcc.Upload(
                    id="upload-indices",
                    children=html.Div([
                        "Drag & Drop or ",
                        html.A("Browse", style={"color": "#4C9BE8", "cursor": "pointer"}),
                    ]),
                    style={
                        "border": "1.5px dashed #bbb", "borderRadius": "5px",
                        "padding": "10px 8px", "textAlign": "center",
                        "color": "#999", "fontSize": "12px", "cursor": "pointer",
                        "marginBottom": "4px",
                    },
                    multiple=False,
                ),
                html.Div(id="upload-status", style=_HINT),

                html.Hr(style={"margin": "12px 0"}),

                # ── Upload sequences CSV ─────────────────────────────────────
                html.Label("Upload sequences CSV:", style=_LABEL),
                dcc.Upload(
                    id="upload-sequences",
                    children=html.Div([
                        "Drag & Drop or ",
                        html.A("Browse", style={"color": "#4C9BE8", "cursor": "pointer"}),
                    ]),
                    style={
                        "border": "1.5px dashed #bbb", "borderRadius": "5px",
                        "padding": "10px 8px", "textAlign": "center",
                        "color": "#999", "fontSize": "12px", "cursor": "pointer",
                        "marginBottom": "4px",
                    },
                    multiple=False,
                ),
                html.P("CSV must include a Seq column. Extra metadata columns are kept.",
                       style=_HINT),

                html.Hr(style={"margin": "12px 0"}),

                # ── New sequence ─────────────────────────────────────────────
                html.Label("Inject a new sequence:", style=_LABEL),
                dcc.Textarea(
                    id="seq-input",
                    placeholder="Paste ACGT sequence here…",
                    style={
                        "width": "100%", "height": "72px",
                        "resize": "vertical", "fontFamily": "monospace",
                        "fontSize": "11px", "boxSizing": "border-box",
                        "border": "1px solid #ccc", "borderRadius": "4px",
                        "padding": "6px",
                    },
                ),
                html.P("Will be appended and embedded before re-clustering.",
                        style=_HINT),

                html.Hr(style={"margin": "12px 0"}),

                # ── Re-cluster ───────────────────────────────────────────────
                html.Button(
                    "⟳  Re-cluster",
                    id="recluster-btn", n_clicks=0,
                    style={**_BTN,
                           "background": "#4C9BE8", "color": "white",
                           "fontWeight": "bold", "fontSize": "14px"},
                ),
                html.Div(id="mcs-badge",
                         children=_bottom_info(_state["df"]),
                         style={"fontSize": "11px", "color": "#888",
                                "textAlign": "left", "marginBottom": "12px",
                                "whiteSpace": "pre-wrap", "lineHeight": "1.35"}),

                # ── Reset ────────────────────────────────────────────────────
                html.Button(
                    "↺  Reset to original",
                    id="reset-btn", n_clicks=0,
                    style={**_BTN,
                           "background": "white", "color": "#c0392b",
                           "border": "1px solid #c0392b"},
                ),

                html.Hr(style={"margin": "12px 0"}),

                # ── Status log ───────────────────────────────────────────────
                html.Div(id="run-status",
                         style={"fontSize": "12px", "color": "#555",
                                "lineHeight": "1.5", "whiteSpace": "pre-wrap"}),

            ], style={
                "width": "290px",
                "flexShrink": 0,
                "padding": "14px 16px",
                "borderLeft": "1px solid #e8e8e8",
                "overflowY": "auto",
                "background": "#fafafa",
                "height": "100%",
                "boxSizing": "border-box",
            }),

        ], style={
            "display": "flex",
            "flex": "1 1 auto",
            "minHeight": 0,
            "overflow": "hidden",
        }),

    ], style={"fontFamily": "Arial, sans-serif", "height": "100vh",
               "display": "flex", "flexDirection": "column",
               "overflow": "hidden", "background": "white"})

    # ── Callbacks ─────────────────────────────────────────────────────────────

    app.clientside_callback(
        """
        function(payload) {
            if (payload && payload.url) {
                window.open(payload.url, "_blank", "noopener,noreferrer");
            }
            return "";
        }
        """,
        Output("open-tab-sink", "children"),
        Input("open-tab-url", "data"),
    )

    @app.callback(
        Output("sel-status", "children"),
        Input("scatter", "selectedData"),
        Input("upload-indices", "contents"),
        State("upload-indices", "filename"),
        prevent_initial_call=False,
    )
    def _update_sel_status(sel, upload_contents, upload_filename):
        n_total = len(_state["df"])
        mcs_all = max(2, n_total // 5)

        if upload_contents:
            try:
                _, b64 = upload_contents.split(",", 1)
                text   = base64.b64decode(b64).decode("utf-8", errors="replace")
                idxs   = [int(x.strip()) for x in
                          text.replace(",", "\n").split()
                          if x.strip().lstrip("-").isdigit()]
                mcs_u  = max(2, len(idxs) // 5)
                return (f"📁 {len(idxs)} indices loaded from '{upload_filename}'  "
                        f"·  min_cluster_size = {mcs_u}")
            except Exception as e:
                return f"⚠ Parse error: {e}"

        if sel and sel.get("points"):
            n_sel = len(sel["points"])
            mcs_s = max(2, n_sel // 5)
            return (f"✓ {n_sel} point{'s' if n_sel != 1 else ''} selected  "
                    f"·  min_cluster_size = {mcs_s}")

        return (f"No selection  ·  {n_total} sequences  "
                f"·  min_cluster_size = {mcs_all}")

    @app.callback(
        Output("upload-status", "children"),
        Input("upload-indices", "filename"),
    )
    def _upload_label(fname):
        return f"Loaded: {fname}" if fname else ""

    @app.callback(
        Output("feature-status", "children"),
        Input("feature-chromosomes", "value"),
        Input("feature-strands", "value"),
        Input("feature-expression", "value"),
        Input("feature-expression-band", "value"),
        Input("feature-mode", "value"),
    )
    def _feature_status(chromosomes, strands, expr_col, expr_band, feature_mode):
        mask, active = _feature_match_mask(
            _state["df"],
            chromosomes=chromosomes,
            strands=strands,
            expr_col=expr_col,
            expr_band=expr_band,
        )
        if not active:
            return "No feature filter active."
        verb = "highlighted" if feature_mode == "highlight" else "selected"
        _log(
            "Feature filter updated: "
            f"mode={feature_mode}, matches={int(mask.sum())}, filters={'; '.join(active)}"
        )
        return f"{int(mask.sum())} sequence(s) {verb} by {', '.join(active)}."

    def _parse_uploaded_sequence_csv(contents):
        _, b64 = contents.split(",", 1)
        text = base64.b64decode(b64).decode("utf-8", errors="replace")
        uploaded = pd.read_csv(io.StringIO(text))
        original_rows = len(uploaded)
        seq_col = next((c for c in uploaded.columns if c.lower() == "seq"), None)
        if seq_col is None:
            raise ValueError("uploaded CSV must include a Seq column")
        if seq_col != "Seq":
            uploaded = uploaded.rename(columns={seq_col: "Seq"})
        uploaded["Seq"] = uploaded["Seq"].astype(str).str.upper().str.replace(r"[^ACGTN]", "", regex=True)
        uploaded = uploaded[uploaded["Seq"].str.len() >= 18].copy()
        if uploaded.empty:
            raise ValueError("no valid Seq values of at least 18 bp found")
        _log(
            "Parsed uploaded sequence CSV: "
            f"input_rows={original_rows}, valid_rows={len(uploaded)}, columns={list(uploaded.columns)}"
        )
        return uploaded

    @app.callback(
        Output("scatter",    "figure"),
        Output("run-status", "children"),
        Output("mcs-badge",  "children"),
        Output("open-tab-url", "data"),
        Output("header-title", "children"),
        Output("header-subtitle", "children"),
        Output("tab-list", "children"),
        Output("tab-content", "children"),
        Input("recluster-btn", "n_clicks"),
        Input("reset-btn",     "n_clicks"),
        Input("feature-chromosomes", "value"),
        Input("feature-strands", "value"),
        Input("feature-expression", "value"),
        Input("feature-expression-band", "value"),
        Input("feature-mode", "value"),
        State("scatter",        "selectedData"),
        State("seq-input",      "value"),
        State("upload-indices", "contents"),
        State("upload-sequences", "contents"),
        prevent_initial_call=True,
    )
    def _recluster(recluster_clicks, reset_clicks,
                   chromosomes, strands, expr_col, expr_band, feature_mode,
                   sel_data, new_seq, upload_contents, sequence_upload_contents):
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
        _log(f"Dashboard callback triggered by {triggered}")
        feature_filters = {
            "chromosomes": chromosomes,
            "strands": strands,
            "expr_col": expr_col,
            "expr_band": expr_band,
        }

        if triggered in {
            "feature-chromosomes", "feature-strands", "feature-expression",
            "feature-expression-band", "feature-mode",
        }:
            return (
                _make_fig(_state["df"], feature_filters, feature_mode),
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        # ── Reset ─────────────────────────────────────────────────────────────
        if triggered == "reset-btn":
            _state["df"] = df_orig.copy()
            _state["tabs"] = []
            _state["active_tab"] = None
            n   = len(df_orig)
            mcs = max(2, n // 5)
            _log(f"Dashboard reset: rows={n}, min_cluster_size={mcs}")
            return (_make_fig(_state["df"], feature_filters, feature_mode),
                    f"Reset to original {n} sequences.",
                    _bottom_info(_state["df"]),
                    no_update,
                    _header_title(),
                    _header_subtitle(_state["df"]),
                    _tab_links(),
                    _tab_content())

        # ── Build working copy ─────────────────────────────────────────────────
        df_work = _state["df"].copy()

        # Append new sequence if provided
        added_count = 0
        if new_seq and new_seq.strip():
            seq_clean = "".join(
                c for c in new_seq.upper() if c in "ACGTN"
            )
            if len(seq_clean) >= 18:
                new_row = {col: (df_work[col].dtype.type(0)
                                 if pd.api.types.is_numeric_dtype(df_work[col])
                                 else "")
                           for col in df_work.columns}
                new_row["Seq"]     = seq_clean
                new_row["Cluster"] = -1
                for col in ("TE_name", "repName"):
                    if col in new_row:
                        new_row[col] = "INJECTED"
                for c in (x_col, y_col):
                    new_row[c] = 0.0
                df_work = pd.concat(
                    [df_work, pd.DataFrame([new_row])], ignore_index=True
                )
                added_count += 1
                _log(f"Added pasted sequence length={len(seq_clean)}")
            else:
                return (no_update,
                        f"Sequence too short ({len(seq_clean)} bp < 18).",
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update)

        if sequence_upload_contents:
            try:
                upload_df = _parse_uploaded_sequence_csv(sequence_upload_contents)
                for col in upload_df.columns:
                    if col not in df_work.columns:
                        df_work[col] = ""
                for col in df_work.columns:
                    if col not in upload_df.columns:
                        upload_df[col] = (0 if pd.api.types.is_numeric_dtype(df_work[col]) else "")
                upload_df = upload_df[df_work.columns.tolist()]
                if "Cluster" in upload_df.columns:
                    upload_df["Cluster"] = -1
                for c in (x_col, y_col):
                    if c in upload_df.columns:
                        upload_df[c] = 0.0
                df_work = pd.concat([df_work, upload_df], ignore_index=True)
                added_count += len(upload_df)
                _log(f"Added {len(upload_df)} uploaded sequence(s); working rows={len(df_work)}")
            except Exception as e:
                _log(f"Sequence CSV upload failed: {e}")
                return (
                    no_update,
                    f"Sequence CSV error: {e}",
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )

        # ── Determine subset ──────────────────────────────────────────────────
        indices = None

        if upload_contents:
            try:
                _, b64 = upload_contents.split(",", 1)
                text   = base64.b64decode(b64).decode("utf-8", errors="replace")
                indices = [int(x.strip()) for x in
                           text.replace(",", "\n").split()
                           if x.strip().lstrip("-").isdigit()]
                indices = [i for i in indices if 0 <= i < len(df_work)]
                _log(f"Loaded {len(indices)} valid index selection(s) from upload")
            except Exception as e:
                _log(f"Index upload failed: {e}")
                return no_update, f"⚠ Index file error: {e}", no_update, no_update, no_update, no_update, no_update, no_update

        elif sel_data and sel_data.get("points"):
            indices = [int(pt["customdata"])
                       for pt in sel_data["points"]
                       if pt.get("customdata") is not None]
            _log(f"Using lasso/box selection with {len(indices)} point(s)")

        if indices is not None and len(indices) > 0:
            df_sub     = df_work.iloc[indices].reset_index(drop=True)
            sub_label  = f"{len(df_sub)}-sequence subset"
        else:
            df_sub     = df_work
            sub_label  = f"all {len(df_work)} sequences"
            if added_count:
                sub_label += f" (including {added_count} added)"

        n_sub = len(df_sub)
        if n_sub < 4:
            return (no_update,
                    f"⚠ Too few sequences ({n_sub}) — need at least 4 to cluster.",
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update)

        mcs = max(2, n_sub // 5)
        status_lines = [f"Re-clustering {sub_label}  (min_cluster_size = {mcs})..."]
        _log(f"Reclustering started: rows={n_sub}, min_cluster_size={mcs}, added={added_count}")

        try:
            from te_clustering import clustering_analysis

            df_re, _ = clustering_analysis(
                df_sub,
                kmer=6,
                min_cluster_size=mcs,
                out_dir=None,
                family_name=family,
                debug=False,
                pca_dims=40,
                n_epochs=120,
                random_state=None,
                compute_tsne=False,
            )
            _state["df"] = df_re

            n_cl    = len([c for c in df_re["Cluster"].unique() if c >= 0])
            n_noise = int((df_re["Cluster"] == -1).sum())
            status_lines.append(
                f"✓ Done  ·  {n_sub} sequences  ·  "
                f"{n_cl} cluster(s)  ·  {n_noise} noise"
            )
            _log(f"Reclustering complete: clusters={n_cl}, noise={n_noise}")

        except Exception as e:
            import traceback as _tb
            status_lines.append(f"✘ Clustering error:\n{_tb.format_exc()[-400:]}")
            _log(f"Reclustering failed: {e}")
            return (
                no_update,
                "\n".join(status_lines),
                f"min_cluster_size = {mcs}",
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        fig = _make_fig(_state["df"], feature_filters, feature_mode)
        tab_name = f"recluster_{uuid.uuid4().hex}.html"
        tab_path = tab_dir / tab_name
        tab_label = f"Recluster {len(_state['tabs']) + 1}"
        tab_id = uuid.uuid4().hex
        try:
            fig.write_html(
                tab_path,
                include_plotlyjs=True,
                full_html=True,
                config={
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "modeBarButtonsToAdd": ["lasso2d", "select2d"],
                },
            )
            tab_host = "127.0.0.1" if HOST in {"0.0.0.0", "::"} else HOST
            tab_url = f"http://{tab_host}:{PORT}/recluster-tabs/{tab_name}"
            _state["tabs"].append({
                "id": tab_id,
                "label": f"{tab_label} ({n_sub:,} rows)",
                "url": tab_url,
                "figure": fig.to_dict(),
                "status": "\n".join(status_lines),
            })
            _state["active_tab"] = tab_id
            status_lines.append(f"Created interactive tab: {tab_label}")
            _log(f"Wrote recluster tab HTML: {tab_path}")
            tab_payload = no_update
        except Exception as e:
            status_lines.append(f"Could not create new tab HTML: {e}")
            _log(f"Failed writing recluster tab HTML: {e}")
            tab_payload = no_update

        return (fig,
                "\n".join(status_lines),
                _bottom_info(_state["df"], sub_label, added_count),
                tab_payload,
                _header_title(),
                _header_subtitle(_state["df"], sub_label),
                _tab_links(_state["active_tab"]),
                _tab_content(_state["active_tab"]))

    @app.callback(
        Output("scatter", "figure", allow_duplicate=True),
        Output("run-status", "children", allow_duplicate=True),
        Output("tab-list", "children", allow_duplicate=True),
        Output("tab-content", "children", allow_duplicate=True),
        Input({"type": "cluster-tab", "tab_id": ALL}, "n_clicks"),
        Input({"type": "close-tab", "tab_id": ALL}, "n_clicks"),
        Input({"type": "move-tab-left", "tab_id": ALL}, "n_clicks"),
        Input({"type": "move-tab-right", "tab_id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _manage_recluster_tabs(tab_clicks, close_clicks, move_left_clicks, move_right_clicks):
        prop_id = callback_context.triggered[0]["prop_id"].split(".")[0]
        try:
            payload = json.loads(prop_id)
        except Exception:
            return no_update, no_update, no_update, no_update

        tab_id = payload.get("tab_id")
        action = payload.get("type")
        tabs = _state.get("tabs", [])

        if action == "close-tab":
            closed_active = tab_id == _state.get("active_tab")
            tabs = [t for t in tabs if t["id"] != tab_id]
            _state["tabs"] = tabs
            if closed_active:
                _state["active_tab"] = tabs[-1]["id"] if tabs else None
            active = next((t for t in tabs if t["id"] == _state.get("active_tab")), None)
            fig = go.Figure(active["figure"]) if active else _make_fig(_state["df"])
            status = active["status"] if active else "Closed recluster tab."
            return fig, status, _tab_links(_state["active_tab"]), _tab_content(_state["active_tab"])

        if action in {"move-tab-left", "move-tab-right"}:
            index = next((i for i, t in enumerate(tabs) if t["id"] == tab_id), None)
            if index is None:
                return no_update, no_update, _tab_links(_state.get("active_tab")), _tab_content(_state.get("active_tab"))
            delta = -1 if action == "move-tab-left" else 1
            new_index = index + delta
            if 0 <= new_index < len(tabs):
                tabs[index], tabs[new_index] = tabs[new_index], tabs[index]
                _state["tabs"] = tabs
            active = next((t for t in tabs if t["id"] == _state.get("active_tab")), None)
            fig = go.Figure(active["figure"]) if active else _make_fig(_state["df"])
            status = active["status"] if active else no_update
            return fig, status, _tab_links(_state.get("active_tab")), _tab_content(_state.get("active_tab"))

        active = next((t for t in tabs if t["id"] == tab_id), None)
        if not active:
            return no_update, no_update, _tab_links(_state.get("active_tab")), _tab_content(_state.get("active_tab"))
        _state["active_tab"] = tab_id
        return (
            go.Figure(active["figure"]),
            active["status"],
            _tab_links(tab_id),
            _tab_content(tab_id),
        )

    # ── Launch ────────────────────────────────────────────────────────────────
    display_host = "127.0.0.1" if HOST in {"0.0.0.0", "::"} else HOST
    url  = f"http://{display_host}:{PORT}"
    print(f"\n  {green('●')}  Dashboard → {bold(url)}")
    print(f"  {dim('Press Ctrl+C to stop.')}")
    if open_browser:
        def _open_dashboard():
            try:
                import webbrowser
                webbrowser.open(url)
            except Exception as exc:
                _log(f"Browser auto-open failed: {exc}")

        threading.Timer(1.2, _open_dashboard).start()

    try:
        app.run(debug=False, host=HOST, port=PORT, use_reloader=False)
    except KeyboardInterrupt:
        print(f"\n  {yellow('Stopped.')}")


# ── Local mode menu ───────────────────────────────────────────────────────────

def local_mode_menu():
    """Terminal menu for local (no HPC) operations."""
    _last_results = [None]   # mutable so inner funcs can update it
    _last_family  = [None]

    while True:
        os.system("clear" if sys.stdout.isatty() else ":")
        print()
        print(cyan(GAMECA_ART.rstrip()))
        print(f"  {dim(TAGLINE)}")
        print()
        print(
            green("━" * 66) + "\n"
            + green("  ✓  LOCAL MODE  ") + bold("No HPC connection required.") + "\n"
            + "     " + dim("rmsk data is auto-downloaded (~150 MB, once).  "
                            "Sequences fetched via UCSC.") + "\n"
            + green("━" * 66)
        )
        print()

        last = f"  {dim('Last run:')} {dim(_last_results[0] or '(none)')}"
        print(last)
        print()
        print(_divider("Menu"))
        print(f"    {cyan('[1]')}  Run local pipeline  (auto-downloads rmsk, UCSC API for seqs)")
        print(f"    {cyan('[2]')}  Launch interactive clustering dashboard")
        print(f"    {cyan('[3]')}  View local results summary")
        print(f"    {cyan('[4]')}  Exit")
        print()

        choice = input(f"  {cyan('›')} {bold('Select')}: ").strip()

        if choice == "1":
            out, fam = _step_local_run()
            if out:
                _last_results[0] = out
                _last_family[0]  = fam
                if _confirm("Launch interactive dashboard now?", default=True):
                    _launch_local_dashboard(out, fam)

        elif choice == "2":
            out = _last_results[0] or _ask("Results directory", "./results")
            fam = _last_family[0]  or _ask("Family name (blank = auto-detect)", "")
            _launch_local_dashboard(out, fam or None)

        elif choice == "3":
            out = _last_results[0] or _ask("Results directory", "./results")
            display_results(Path(out))
            input(f"\n  {dim('Press Enter to return…')}")

        elif choice == "4":
            break

        else:
            print(red("  Invalid selection."))
            _t_sleep(0.8)


def _print_banner():
    os.system("clear" if sys.stdout.isatty() else ":")
    print()
    print(cyan(GAMECA_ART.rstrip()))
    print(f"  {dim(TAGLINE)}")
    print(f"  {dim('ui build: ' + _UI_BUILD)}")
    print()


def _startup_mode_prompt():
    """Offer HPC connection immediately on launch. Enter selects HPC."""
    _print_banner()
    print(_box([
        bold("Launch Mode"),
        green("1") + "  Connect to HPC  " + dim("(recommended, fastest full pipeline path)"),
        "2  Local mode",
        "3  Exit",
    ]))
    print()
    choice = input(f"  {cyan('›')} {bold('Select launch mode')} [1]: ").strip().lower()
    if choice in ("", "1", "h", "hpc", "connect"):
        return "hpc"
    if choice in ("2", "l", "local"):
        return "local"
    return "exit"


def _step_rmsk_query(client):
    """Download RepeatMasker table for a build and count sequences matching a family."""
    print()
    print(_divider("RepeatMasker Query  ·  download rmsk + count sequences"))
    family   = (client.params.get("FAMILY_NAME", "") or "") if hasattr(client, "params") else ""
    assembly = (client.params.get("ASSEMBLY", "hg38") or "hg38") if hasattr(client, "params") else "hg38"

    family = _ask("TE family repName (e.g. HERVK, THE1D-int, L1HS)", family)
    if not family:
        print(red("  Family name is required."))
        return

    _, assembly = _ask_species_assembly(default_assembly=assembly)
    rmsk_dir  = f"{client.remote_work_dir}/te_rmsk"
    rmsk_file = f"{rmsk_dir}/rmsk_{assembly}.txt.gz"

    if not _sync_remote_files(client, ["te_prep.py"]):
        return

    # Download rmsk only if not already cached on the cluster
    check_out, _, _ = client.run_command(
        f"test -f {rmsk_file} && echo EXISTS || echo MISSING", timeout=15
    )
    if "EXISTS" not in check_out:
        _log(f"rmsk_{assembly}.txt.gz not found — downloading from UCSC…")
        dl_cmd = (
            f"mkdir -p {rmsk_dir} && "
            f"{client._python} -u {client.remote_work_dir}/te_prep.py "
            f"--download {assembly} --rmsk-dir {rmsk_dir}"
        )
        if not _remote_run(client, f"Download rmsk ({assembly})", dl_cmd, timeout=900, force_direct=True):
            return
    else:
        _log(f"rmsk_{assembly}.txt.gz already present — skipping download")

    # Count rows where repName (column 11 in awk) matches the family exactly
    _log(f"Counting sequences for repName='{family}' in {assembly}…")
    count_out, _, _ = client.run_command(
        f"zcat {rmsk_file} | awk -F'\\t' '$11==\"{family}\"' | wc -l",
        timeout=120,
    )
    count_str = count_out.strip().split()[-1] if count_out.strip() else "?"
    try:
        count = int(count_str)
        count_line = green(f"   Sequences matching:  {count:,}")
    except ValueError:
        count = None
        count_line = yellow(f"   Could not parse count: {count_str!r}")

    print()
    print(_box([
        bold("RepeatMasker Query Result"),
        f"   Family:    {bold(family)}",
        f"   Assembly:  {bold(assembly)}",
        f"   rmsk file: {dim(rmsk_file)}",
        count_line,
    ]))
    print()

    # Persist into client params so an immediately following batch job inherits them
    if hasattr(client, "params"):
        client.params["FAMILY_NAME"] = family
        client.params["ASSEMBLY"]    = assembly
        _log("Saved family + assembly to session parameters.")


def _step_dfam_query(client):
    """Count Dfam sequences for a family via the REST API (no bulk download needed)."""
    print()
    print(_divider("Dfam Query  ·  Dfam REST API locus count"))
    family   = (client.params.get("FAMILY_NAME", "") or "") if hasattr(client, "params") else ""
    assembly = (client.params.get("ASSEMBLY", "hg38") or "hg38") if hasattr(client, "params") else "hg38"

    family = _ask("TE family name (e.g. HERVK, THE1D-int, L1HS)", family)
    if not family:
        print(red("  Family name is required."))
        return

    _, assembly = _ask_species_assembly(default_assembly=assembly)

    if not _sync_remote_files(client, ["te_prep.py"]):
        return

    # Dfam uses the REST API per-family — no bulk file download needed
    _log(f"Counting sequences for family='{family}' in {assembly} via Dfam REST API…")
    count_out, count_err, count_rc = client.run_command(
        f"{client._python} {client.remote_work_dir}/te_prep.py "
        f"--family {family} --build {assembly} --count-only --source dfam",
        timeout=120,
    )
    if count_rc != 0:
        print(red(f"\n  Dfam count failed (exit {count_rc})."))
        # Skip the always-present conda noise; show the real error lines
        all_text = (count_out + "\n" + count_err)
        noise = {"conda-libmamba", "libicui18n", "conda entry point"}
        relevant = [l for l in all_text.splitlines()
                    if l.strip() and not any(n in l for n in noise)]
        for ln in relevant[-8:]:
            print(red(f"  {ln.strip()}"))
        return
    # Parse "Locus count for 'FAMILY' (BUILD, dfam): N,NNN  [Xs]"
    import re as _re
    m = _re.search(r"Locus count for '[^']+' \([^,]+, [^)]+\):\s*([\d,]+)", count_out)
    count_str = m.group(1) if m else "?"
    try:
        count = int(count_str.replace(",", ""))
        count_line = green(f"   Sequences matching:  {count:,}")
    except ValueError:
        count = None
        count_line = yellow(f"   Could not parse count: {count_str!r}")

    print()
    print(_box([
        bold("Dfam Query Result"),
        f"   Family:    {bold(family)}",
        f"   Assembly:  {bold(assembly)}",
        f"   Source:    Dfam REST API (dfam.org)",
        count_line,
    ]))
    print()

    if hasattr(client, "params"):
        client.params["FAMILY_NAME"] = family
        client.params["ASSEMBLY"]    = assembly
        _log("Saved family + assembly to session parameters.")


def _post_connect_launch_action(client):
    """Prompt for the first action after connecting to HPC."""
    sched = (client.scheduler or "?").upper()
    print(_box([
        bold("HPC Run Mode"),
        f"1  Query RepeatMasker  {dim('(download rmsk + count sequences)')}",
        f"2  Query Dfam  {dim('(download nrph hits + count sequences)')}",
        f"3  Configure + submit batch job   {dim(f'({sched}, background)')}",
        "4  Open full HPC menu",
        "5  Download results",
        "6  Disconnect",
    ]))
    print()

    while True:
        choice = input(f"  {cyan('›')} {bold('Select run mode')} [4]: ").strip().lower()
        if choice in ("", "4", "m", "menu"):
            return True

        if choice in ("1", "r", "rmsk", "query"):
            _step_rmsk_query(client)
            input(f"  {dim('Press Enter to open the full HPC menu…')}")
            return True

        if choice in ("2", "d", "dfam"):
            _step_dfam_query(client)
            input(f"  {dim('Press Enter to open the full HPC menu…')}")
            return True

        if choice in ("3", "b", "batch"):
            if client.set_parameter_interactive():
                client.submit_batch_job()
            input(f"  {dim('Press Enter to open the full HPC menu…')}")
            return True

        if choice in ("5", "dl", "download"):
            cached_remote = client._state.get("last_remote_dir", "").strip()
            if cached_remote:
                remote_prompt = f"  Remote path [{cached_remote}]: "
            else:
                remote_prompt = "  Remote path (leave blank to auto-detect): "
            remote_override = input(remote_prompt).strip() or cached_remote or None

            cached_local = str(client.local_output_dir) if client.local_output_dir else "./hpc_results"
            local_dir = input(f"  Local destination directory [{cached_local}]: ").strip() or cached_local
            client.retrieve_results(local_dir, remote_out_override=remote_override)
            input(f"  {dim('Press Enter to open the full HPC menu…')}")
            return True

        if choice in ("6", "d", "disconnect", "q", "quit", "exit"):
            return False

        print(red("  Invalid selection."))


def _t_sleep(s):
    import time
    time.sleep(s)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GAMECA pipeline launcher  (HPC or local)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Local mode (no HPC required):\n"
            "  python ui.py --local\n"
            "  python query.py --local --family THE1D-int --assembly hg38\n"
        ),
    )
    parser.add_argument("-H", "--host",     help="HPC cluster hostname or IP")
    parser.add_argument("-p", "--port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("-u", "--user",     help="SSH username")
    parser.add_argument("-i", "--key", metavar="KEY_FILE",
                        help="SSH private key file (e.g. ~/.ssh/id_rsa)")
    parser.add_argument("-P", "--password", metavar="PASSWORD",
                        help="SSH password (WARNING: visible in shell history)")
    parser.add_argument("--scheduler", choices=["lsf", "slurm"],
                        help="Force scheduler type. Auto-detected from PATH if omitted.")
    parser.add_argument("--local",          action="store_true",
                        help="Run in local mode (no HPC connection)")
    parser.add_argument("--results", metavar="DIR",
                        help="Display local results summary for DIR and exit")
    parser.add_argument("--dashboard", metavar="DIR",
                        help="Launch the clustering dashboard for a local/downloaded results directory")
    parser.add_argument("--family",
                        help="Family name to use with --dashboard when auto-detect is ambiguous")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not automatically open a browser for --dashboard")
    parser.add_argument("--help-workflow",  action="store_true",
                        help="Print workflow overview and exit")
    args = parser.parse_args()

    if args.help_workflow:
        print(cyan(GAMECA_ART))
        print(_header("GAMECA Pipeline  ·  Workflow Overview"))
        print(WORKFLOW)
        return

    if args.results:
        display_results(Path(args.results))
        return

    if args.dashboard:
        _launch_local_dashboard(args.dashboard, args.family, open_browser=not args.no_browser)
        return

    # ── Local mode: no SSH needed ─────────────────────────────────────────────
    if args.local:
        _print_banner()
        local_mode_menu()
        return

    launch_mode = "hpc" if (args.host or args.user) else _startup_mode_prompt()
    if launch_mode == "local":
        local_mode_menu()
        return
    if launch_mode == "exit":
        return

    # ── HPC mode ──────────────────────────────────────────────────────────────
    _print_banner()
    print(HPC_NOTICE)
    print()

    client = _connect(hostname=args.host, username=args.user, port=args.port,
                      key_path=args.key or "", password=args.password)

    # Apply --scheduler override, or prompt if auto-detection failed
    if args.scheduler:
        client.scheduler = args.scheduler
        print(f"  {green('›')} Scheduler override: {bold(args.scheduler.upper())}")
    elif not client.scheduler:
        print(f"\n  {yellow('⚠')}  Could not auto-detect a scheduler (bsub/sbatch not found on PATH).")
        while True:
            choice = _ask("Choose scheduler [lsf/slurm]").strip().lower()
            if choice in ("lsf", "slurm"):
                client.scheduler = choice
                break
            print(red("  Please enter 'lsf' or 'slurm'."))

    try:
        if _post_connect_launch_action(client):
            interactive_menu(client)
    except KeyboardInterrupt:
        print(f"\n\n  {yellow('Interrupted.')}  Disconnecting…")
        client.disconnect()
        sys.exit(0)
    finally:
        if client.connected:
            client.disconnect()


if __name__ == "__main__":
    main()
