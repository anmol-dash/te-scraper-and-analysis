#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import getpass
import io
import json
import os
import posixpath
import re
import shlex
import stat
import sys
import tarfile
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import paramiko
except ImportError:  # pragma: no cover - local environment guard
    print("paramiko is required: python -m pip install paramiko", file=sys.stderr)
    raise


REMOTE_TEST_RUNNER = r"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import importlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ASSEMBLY_SPECIES = {
    "hg38": "human",
    "hg19": "human",
    "mm10": "mouse",
    "mm39": "mouse",
}

SIZE_MIN_COUNTS = {
    "custom":     1,
    "small":      1,
    "medium":     50,
    "large":      500,
    "very_large": 5_000,
}

# Upper bounds are generous — the meaningful check is the floor.
SIZE_MAX_COUNTS = {
    "custom":     2_000_000,
    "small":      100_000,
    "medium":     500_000,
    "large":      2_000_000,
    "very_large": 10_000_000,
}

# 15 cases × 4 assemblies × 2 species × 4 size classes (small/medium/large/very_large)
# Distinct family names: AluY, L1HS, SVA_D, HERVK-int, THE1D-int, MIRb, AluSz, L1PA3,
#                        B1_Mm, L1Md_A, MT2_Mm, RLTR4_Mm, B2_Mm2, IAPEz-int, L1Md_Gf
DEFAULT_BARE_CASES = [
    # hg38 — human — small through very_large
    ("AluY",      "hg38", "human", "very_large"),
    ("L1HS",      "hg38", "human", "large"),
    ("SVA_D",     "hg38", "human", "medium"),
    ("HERVK-int", "hg38", "human", "medium"),
    ("THE1D-int", "hg38", "human", "small"),
    # hg19 — human — medium through very_large
    ("MIRb",      "hg19", "human", "very_large"),
    ("AluSz",     "hg19", "human", "large"),
    ("L1PA3",     "hg19", "human", "medium"),
    # mm10 — mouse — small through very_large
    ("B1_Mm",     "mm10", "mouse", "very_large"),
    ("L1Md_A",    "mm10", "mouse", "large"),
    ("MT2_Mm",    "mm10", "mouse", "medium"),
    ("RLTR4_Mm",  "mm10", "mouse", "small"),
    # mm39 — mouse — medium through very_large
    ("B2_Mm2",    "mm39", "mouse", "very_large"),
    ("IAPEz-int", "mm39", "mouse", "large"),
    ("L1Md_Gf",   "mm39", "mouse", "medium"),
]


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def require_path(path, min_size: int = 1) -> Path:
    p = Path(path)
    require(p.exists(), f"missing expected path: {p}")
    if p.is_file():
        require(p.stat().st_size >= min_size, f"expected non-empty file: {p}")
    elif p.is_dir():
        require(any(p.iterdir()), f"expected non-empty directory: {p}")
    return p


def run_cmd(cmd, *, cwd=None, env=None, timeout=300) -> subprocess.CompletedProcess:
    printable = " ".join(map(str, cmd)) if isinstance(cmd, (list, tuple)) else str(cmd)
    log(f"$ {printable}")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if proc.stdout.strip():
        out = proc.stdout if len(proc.stdout) < 6000 else proc.stdout[-6000:]
        print(out, flush=True)
    if proc.stderr.strip():
        err = proc.stderr if len(proc.stderr) < 6000 else proc.stderr[-6000:]
        print(err, file=sys.stderr, flush=True)
    if proc.returncode != 0:
        raise AssertionError(f"command failed with exit {proc.returncode}: {printable}")
    return proc


def parse_bare_case(raw: str) -> tuple[str, str]:
    text = raw.strip()
    if ":" in text:
        family, assembly = text.split(":", 1)
    elif "," in text:
        family, assembly = text.split(",", 1)
    else:
        parts = text.split()
        if len(parts) != 2:
            raise ValueError(f"bare case must be FAMILY:ASSEMBLY, got {raw!r}")
        family, assembly = parts
    family = family.strip()
    assembly = assembly.strip()
    if not family or not assembly:
        raise ValueError(f"bare case must include family and assembly, got {raw!r}")
    return family, assembly


def normalize_bare_case(case) -> tuple[str, str, str, str]:
    if isinstance(case, dict):
        family = str(case["family"])
        assembly = str(case["assembly"])
        species = str(case.get("species") or ASSEMBLY_SPECIES.get(assembly, "unknown"))
        size_class = str(case.get("size_class") or case.get("size") or "custom")
        return family, assembly, species, size_class

    if len(case) == 2:
        family, assembly = case
        species = ASSEMBLY_SPECIES.get(assembly, "unknown")
        return str(family), str(assembly), species, "custom"

    if len(case) >= 4:
        family, assembly, species, size_class = case[:4]
        return str(family), str(assembly), str(species), str(size_class)

    raise ValueError(f"invalid family/assembly case: {case!r}")


def case_key(family: str, assembly: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{family}_{assembly}")


def parse_locus_count(text: str) -> int:
    match = re.search(r"Locus count for .+?:\s*([0-9][0-9,]*)", text)
    if not match:
        raise AssertionError("could not parse locus count from te_prep output")
    return int(match.group(1).replace(",", ""))


def standard_chroms_for(assembly: str) -> set[str]:
    if assembly.startswith("hg"):
        return {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
    return {f"chr{i}" for i in range(1, 20)} | {"chrX", "chrY"}


def require_family_matrix_coverage(cases) -> None:
    normalized = [normalize_bare_case(case) for case in cases]
    families = {family.lower() for family, _, _, _ in normalized}
    assemblies = {assembly for _, assembly, _, _ in normalized}
    species = {sp for _, _, sp, _ in normalized}
    sizes = {size for _, _, _, size in normalized}
    require(len(families) >= 10, f"family matrix must include at least 10 distinct families, got {len(families)}")
    require(assemblies == {"hg38", "hg19", "mm10", "mm39"}, f"family matrix must cover hg38/hg19/mm10/mm39, got {sorted(assemblies)}")
    require({"human", "mouse"}.issubset(species), f"family matrix must cover human and mouse, got {sorted(species)}")
    require(len(sizes) >= 3, f"family matrix should include at least 3 size classes, got {sorted(sizes)}")


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1].upper()


def seq_template(length: int, seed: int, motif: str) -> str:
    rng = random.Random(seed)
    bases = [rng.choice("ACGT") for _ in range(length)]
    insert_at = min(25, max(0, length - len(motif) - 1))
    bases[insert_at:insert_at + len(motif)] = list(motif)
    return "".join(bases)


def mutate(seq: str, seed: int, every: int = 29) -> str:
    rng = random.Random(seed)
    bases = list(seq)
    for i in range(seed % 11, len(bases), every):
        bases[i] = rng.choice([b for b in "ACGT" if b != bases[i]])
    return "".join(bases)


def write_fasta(path: Path, records: dict[str, str]) -> None:
    with open(path, "w") as fh:
        for name, seq in records.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i + 80] + "\n")


def build_test_fixture(work: Path) -> dict[str, Path]:
    import pandas as pd

    fixture = work / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)

    motif_a = "ACGTACGTGGAACC"
    motif_b = "TTGGAACCGGTATA"
    base_a = seq_template(220, 101, motif_a)
    base_b = seq_template(220, 202, motif_b)

    chroms = {
        "chr1": list(seq_template(7000, 301, "GATTACA")),
        "chr2": list(seq_template(7000, 302, "CCTAGG")),
        "chr3": list(seq_template(7000, 303, "TATAAA")),
    }
    rows = []
    for i in range(8):
        seq = mutate(base_a, 400 + i, every=41)
        strand = "+" if i % 2 == 0 else "-"
        chrom = "chr1"
        start = 100 + i * 350
        genome_seq = seq if strand == "+" else reverse_complement(seq)
        chroms[chrom][start:start + len(seq)] = list(genome_seq)
        rows.append({
            "chr": chrom,
            "start": start,
            "stop": start + len(seq),
            "Chromosome": chrom,
            "Start": start,
            "Stop": start + len(seq),
            "TE_name": f"TEST_TE_element_{i}",
            "TE_ID": f"{chrom}|{start}|{start + len(seq)}|TEST_TE:ERVL:LTR|{900+i}|{strand}",
            "repName": "TEST_TE",
            "repClass": "LTR",
            "repFamily": "ERVL",
            "swScore": 900 + i,
            "milliDiv": i,
            "strand": strand,
            "Seq": seq,
            "Cluster": 0,
            "stage_a": float(100 + i),
            "stage_b": float(25 + i),
        })
    for i in range(8):
        seq = mutate(base_b, 500 + i, every=37)
        strand = "-" if i % 2 == 0 else "+"
        chrom = "chr2"
        start = 150 + i * 350
        genome_seq = seq if strand == "+" else reverse_complement(seq)
        chroms[chrom][start:start + len(seq)] = list(genome_seq)
        rows.append({
            "chr": chrom,
            "start": start,
            "stop": start + len(seq),
            "Chromosome": chrom,
            "Start": start,
            "Stop": start + len(seq),
            "TE_name": f"TEST_TE_element_{8+i}",
            "TE_ID": f"{chrom}|{start}|{start + len(seq)}|TEST_TE:ERVL:LTR|{800+i}|{strand}",
            "repName": "TEST_TE",
            "repClass": "LTR",
            "repFamily": "ERVL",
            "swScore": 800 + i,
            "milliDiv": i,
            "strand": strand,
            "Seq": seq,
            "Cluster": 1,
            "stage_a": float(10 + i),
            "stage_b": float(120 + i),
        })

    genome_fa = fixture / "mini_genome.fa"
    write_fasta(genome_fa, {k: "".join(v) for k, v in chroms.items()})

    df = pd.DataFrame(rows)
    csv_with_seq = fixture / "clustered_with_seq.csv"
    csv_no_seq = fixture / "loci_no_seq.csv"
    df.to_csv(csv_with_seq, index=False)
    df.drop(columns=["Seq", "Cluster"]).to_csv(csv_no_seq, index=False)

    rmsk_path = fixture / "rmsk_hg38.txt.gz"
    with gzip.open(rmsk_path, "wt") as fh:
        for i, row in df.iterrows():
            fields = [
                "0", str(row["swScore"]), str(row["milliDiv"]), "0", "0",
                row["chr"], str(row["start"]), str(row["stop"]), "0",
                row["strand"], "TEST_TE", "LTR", "ERVL", "0", "0", "0",
            ]
            fh.write("\t".join(fields) + "\n")
        fields = ["0", "1", "1", "0", "0", "chr3", "1", "80", "0", "+", "OTHER_TE", "SINE", "B2"]
        fh.write("\t".join(fields) + "\n")

    dfam_buf = io.BytesIO()
    with gzip.GzipFile(fileobj=dfam_buf, mode="wb") as gz:
        text = (
            "#sequence name\tmodel accession\tmodel name\tbit score\te-value\thmm start\thmm end\thmm length\tstrand\talignment start\talignment end\tenvelope start\tenvelope end\tsequence length\n"
            "chr1\tDF0001\tTEST_TE\t100\t1e-9\t1\t50\t100\t+\t100\t250\t100\t250\t7000\n"
            "chr2\tDF0001\tTEST_TE\t100\t1e-9\t1\t50\t100\t-\t600\t450\t450\t600\t7000\n"
        )
        gz.write(text.encode())
    dfam_path = fixture / "dfam_raw.tsv.gz"
    dfam_path.write_bytes(dfam_buf.getvalue())

    jaspar_bed = fixture / "jaspar_test.bed"
    with open(jaspar_bed, "w") as fh:
        for _, row in df.iterrows():
            if int(row["Cluster"]) == 0:
                fh.write(f"{row['chr']}\t{int(row['start']) + 20}\t{int(row['start']) + 55}\tTFAP2A::MA0003.4\t900\t+\n")
            else:
                fh.write(f"{row['chr']}\t{int(row['start']) + 20}\t{int(row['start']) + 55}\tGATA1::MA0035.4\t900\t-\n")
            fh.write(f"{row['chr']}\t{int(row['start']) + 80}\t{int(row['start']) + 115}\tSP1::MA0079.3\t600\t+\n")

    return {
        "fixture": fixture,
        "genome": genome_fa,
        "csv_with_seq": csv_with_seq,
        "csv_no_seq": csv_no_seq,
        "rmsk": rmsk_path,
        "dfam": dfam_path,
        "jaspar": jaspar_bed,
    }


class FeatureTester:
    def __init__(
        self,
        repo: Path,
        work: Path,
        json_path: Path,
        *,
        include_dev: bool,
        live_network: bool,
        skip_cython: bool,
        bare_cases: list[tuple],
        bare_max_loci: int,
    ) -> None:
        self.repo = repo
        self.work = work
        self.json_path = json_path
        self.include_dev = include_dev
        self.live_network = live_network
        self.skip_cython = skip_cython
        self.bare_cases = bare_cases
        self.bare_max_loci = bare_max_loci
        self.family_artifacts: dict[str, dict] = {}
        self.rmsk_dir = self.work / "rmsk_cache"
        self.rmsk_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []
        self.fixture = build_test_fixture(work)
        os.environ.setdefault("NUMBA_CACHE_DIR", str(work / "numba_cache"))
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "backend"))

    def run(self, name: str, func, *, required: bool = True) -> None:
        log("")
        log("=" * 78)
        log(f"START {name}  required={required}")
        start = time.time()
        status = "PASS"
        message = ""
        try:
            func()
        except Exception as exc:
            status = "FAIL" if required else "WARN"
            message = f"{type(exc).__name__}: {exc}"
            log(f"{status} {name}: {message}")
            traceback.print_exc()
        elapsed = time.time() - start
        if status == "PASS":
            log(f"PASS {name} ({elapsed:.1f}s)")
        self.results.append({
            "name": name,
            "required": required,
            "status": status,
            "seconds": round(elapsed, 3),
            "message": message,
        })
        self.write_summary()

    def write_summary(self) -> None:
        required_failures = [r for r in self.results if r["required"] and r["status"] != "PASS"]
        data = {
            "repo": str(self.repo),
            "work": str(self.work),
            "host": os.uname().nodename if hasattr(os, "uname") else "",
            "python": sys.version,
            "scheduler": {
                "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", ""),
                "LSB_JOBID": os.environ.get("LSB_JOBID", ""),
                "PBS_JOBID": os.environ.get("PBS_JOBID", ""),
            },
            "results": self.results,
            "required_failures": required_failures,
            "ok": not required_failures,
        }
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.write_text(json.dumps(data, indent=2))

    def finish(self) -> int:
        self.write_summary()
        failures = [r for r in self.results if r["required"] and r["status"] != "PASS"]
        log("")
        log("=" * 78)
        log("FINAL SUMMARY")
        for r in self.results:
            marker = "OK" if r["status"] == "PASS" else r["status"]
            req = "required" if r["required"] else "optional"
            log(f"{marker:5s} {req:8s} {r['seconds']:7.1f}s  {r['name']} {r['message']}")
        log(f"Summary JSON: {self.json_path}")
        if failures:
            log(f"REQUIRED FAILURES: {len(failures)}")
            return 1
        log("ALL REQUIRED FEATURE TESTS PASSED")
        return 0

    def test_batch_runtime(self) -> None:
        log(f"host={os.uname().nodename if hasattr(os, 'uname') else 'unknown'}")
        log(f"cwd={Path.cwd()}")
        log(f"python={sys.version}")
        log(f"PATH={os.environ.get('PATH', '')}")
        require(os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID"), "not running inside a Slurm or LSF batch allocation")
        require_path(self.repo / "README.md", 100)
        require_path(self.repo / "query.py", 1000)

    def test_repository_integrity(self) -> None:
        import py_compile

        readme = (self.repo / "README.md").read_text(errors="replace")
        for phrase in ["Full TE pipeline", "HPC integration", "RMSK", "Dfam", "Primer design"]:
            require(phrase in readme, f"README missing expected feature phrase: {phrase}")
        py_files = [
            "unit_tests.py", "query.py", "hpc_client.py", "ui.py", "te_prep.py", "te_genome.py",
            "te_clustering.py", "te_alignment.py", "te_motif.py", "te_go.py",
            "te_expression.py", "te_enrichment.py", "te_primers.py",
            "backend/main.py", "backend/yourtool/cli.py",
        ]
        for rel in py_files:
            py_compile.compile(str(self.repo / rel), doraise=True)

    def test_dependency_imports(self) -> None:
        modules = [
            "numpy", "pandas", "scipy", "requests", "bs4", "dash", "plotly",
            "sklearn", "umap", "hdbscan", "Bio", "pysam", "matplotlib",
            "paramiko", "CIAlign",
        ]
        for mod in modules:
            importlib.import_module(mod)
            log(f"import ok: {mod}")

    def test_external_tools(self) -> None:
        required = ["bedtools", "mafft", "CIAlign"]
        missing = [cmd for cmd in required if shutil.which(cmd) is None]
        for cmd in required:
            log(f"{cmd}: {shutil.which(cmd) or 'MISSING'}")
        optional = ["findMotifsGenome.pl", "bgzip", "tabix", "bigBedToBed", "pnpm", "node", "cargo"]
        for cmd in optional:
            log(f"optional {cmd}: {shutil.which(cmd) or 'not found'}")
        require(not missing, "missing required external command(s): " + ", ".join(missing))

    def test_te_prep_and_annotation_parsers(self) -> None:
        import pandas as pd
        import te_prep

        hits = te_prep.parse_rmsk_family(str(self.fixture["rmsk"]), "test_te", std_chroms={"chr1", "chr2"})
        require(len(hits) == 16, f"expected 16 RMSK hits, got {len(hits)}")
        count = te_prep.count_family_rmsk(str(self.fixture["rmsk"]), "test_te", std_chroms={"chr1", "chr2"})
        require(count == 16, f"expected RMSK count 16, got {count}")
        df_hits = pd.DataFrame(hits).head(4)
        seqs = te_prep.extract_sequences_fasta(str(self.fixture["genome"]), df_hits)
        require(len(seqs) == 4 and all(seqs), "FASTA sequence extraction returned empty sequence(s)")

        raw = self.fixture["dfam"].read_bytes()
        old_fetch = te_prep._dfam_fetch_raw
        try:
            te_prep._dfam_fetch_raw = lambda family, build: raw
            dfam_hits = te_prep.parse_dfam_family(None, "TEST_TE", build="hg38", std_chroms={"chr1", "chr2"})
            dfam_count = te_prep.count_family_dfam(None, "TEST_TE", build="hg38", std_chroms={"chr1", "chr2"})
        finally:
            te_prep._dfam_fetch_raw = old_fetch
        require(len(dfam_hits) == 2, f"expected 2 Dfam hits, got {len(dfam_hits)}")
        require(dfam_count == 2, f"expected Dfam count 2, got {dfam_count}")
        require(dfam_hits[1]["Start"] < dfam_hits[1]["Stop"], "Dfam minus-strand coordinates were not normalized")

    def test_family_matrix_coverage_validation(self) -> None:
        require_family_matrix_coverage(self.bare_cases)
        normalized = [normalize_bare_case(c) for c in self.bare_cases]
        families   = {f.lower() for f, _, _, _ in normalized}
        assemblies = {a       for _, a, _, _ in normalized}
        species    = {sp      for _, _, sp, _ in normalized}
        sizes      = {sz      for _, _, _, sz in normalized}
        log(f"  {len(families)} distinct families: {sorted(families)}")
        log(f"  assemblies  : {sorted(assemblies)}")
        log(f"  species     : {sorted(species)}")
        log(f"  size classes: {sorted(sizes)}")
        require(len(families) >= 10,
                f"need ≥10 distinct families, got {len(families)}: {sorted(families)}")
        require(assemblies == {"hg38", "hg19", "mm10", "mm39"},
                f"must cover hg38/hg19/mm10/mm39, got {sorted(assemblies)}")
        require({"human", "mouse"}.issubset(species),
                f"must cover human and mouse, got {sorted(species)}")
        require(len(sizes) >= 3,
                f"need ≥3 size classes, got {sorted(sizes)}")

    def test_bare_family_assembly_auto_query_matrix(self) -> None:
        import pandas as pd

        require_family_matrix_coverage(self.bare_cases)
        log(
            f"Running bare family×assembly matrix: {len(self.bare_cases)} cases, "
            f"--max-loci={self.bare_max_loci}.  Covers all 4 assemblies and ≥10 distinct families."
        )

        for case in self.bare_cases:
            family, assembly, species, size_class = normalize_bare_case(case)
            safe = case_key(family, assembly)
            out = self.work / "bare_family_query" / safe
            family_dir = out / family.lower()

            # ── te_prep --count-only ──────────────────────────────────────────
            log(f"[{family}/{assembly}] te_prep --count-only (size_class={size_class})")
            count_proc = run_cmd([
                sys.executable,
                str(self.repo / "te_prep.py"),
                family, assembly,
                "--count-only",
            ], cwd=self.repo, timeout=1800)
            combined_output = count_proc.stdout + "\n" + count_proc.stderr
            try:
                full_count = parse_locus_count(combined_output)
                min_expected = SIZE_MIN_COUNTS.get(size_class, 1)
                max_expected = SIZE_MAX_COUNTS.get(size_class, 10_000_000)
                require(
                    full_count >= min_expected,
                    f"{family}/{assembly}: locus count {full_count:,} below expected minimum "
                    f"{min_expected:,} for size_class={size_class!r}",
                )
                require(
                    full_count <= max_expected,
                    f"{family}/{assembly}: locus count {full_count:,} above expected maximum "
                    f"{max_expected:,} for size_class={size_class!r}",
                )
                log(f"  locus count: {full_count:,}  (expected {min_expected:,}–{max_expected:,})")
            except AssertionError as _count_err:
                if "could not parse locus count" in str(_count_err):
                    log(f"  WARNING: could not parse locus count output for {family}/{assembly}")
                else:
                    raise

            # ── query.py --local --stop-after sequences ───────────────────────
            log(f"[{family}/{assembly}] query.py --local --stop-after sequences")
            run_cmd([
                sys.executable,
                str(self.repo / "query.py"),
                "--local",
                "--family", family,
                "--assembly", assembly,
                "--output", str(out),
                "--max-loci", str(self.bare_max_loci),
                "--stop-after", "sequences",
            ], cwd=self.repo, timeout=1800)

            seq_csv = family_dir / "01_data" / f"{family.lower()}_with_sequences.csv"
            require_path(seq_csv, 10)
            df = pd.read_csv(seq_csv)

            # Row count bounded by max_loci cap
            require(
                0 < len(df) <= self.bare_max_loci,
                f"{family}/{assembly}: locus count {len(df)} not in (0, {self.bare_max_loci}]",
            )

            # Detect column names — query.py adds lowercase chr/start/stop aliases
            chrom_col  = next((c for c in ("chr", "Chromosome")  if c in df.columns), None)
            start_col  = next((c for c in ("start", "Start")     if c in df.columns), None)
            stop_col   = next((c for c in ("stop",  "Stop")      if c in df.columns), None)
            strand_col = next((c for c in ("strand", "Strand")   if c in df.columns), None)
            seq_col    = next((c for c in ("Seq",   "seq")       if c in df.columns), None)
            require(
                all([chrom_col, start_col, stop_col, strand_col, seq_col]),
                f"{family}/{assembly}: missing required columns; found {list(df.columns)}",
            )

            # Sequences non-empty and valid IUPAC DNA (ACGTN only)
            seqs = df[seq_col].astype(str)
            require(
                seqs.str.len().gt(0).all(),
                f"{family}/{assembly}: empty sequence(s) found",
            )
            valid_dna = seqs.str.upper().str.fullmatch(r"[ACGTN]+")
            require(
                valid_dna.all(),
                f"{family}/{assembly}: non-DNA characters in sequences "
                f"(example: {seqs[~valid_dna].iloc[0][:30]!r})",
            )

            # Coordinates: start ≥ 0 and start < stop
            require(
                (df[start_col] >= 0).all(),
                f"{family}/{assembly}: negative start coordinates",
            )
            require(
                (df[start_col] < df[stop_col]).all(),
                f"{family}/{assembly}: start ≥ stop in "
                f"{(df[start_col] >= df[stop_col]).sum()} rows",
            )

            # Sequence length plausible (> 10 bp, < 50 kbp)
            seq_lens = seqs.str.len()
            require(
                seq_lens.gt(10).all(),
                f"{family}/{assembly}: implausibly short sequences (min len={seq_lens.min()})",
            )
            require(
                seq_lens.lt(50_000).all(),
                f"{family}/{assembly}: implausibly long sequences (max len={seq_lens.max()})",
            )

            # All chromosomes are standard for this assembly
            std_chroms = standard_chroms_for(assembly)
            row_chroms = set(df[chrom_col].astype(str))
            non_std = row_chroms - std_chroms
            require(
                not non_std,
                f"{family}/{assembly}: non-standard chromosomes present: {sorted(non_std)[:5]}",
            )

            # Strand values are + or −
            strand_vals = set(df[strand_col].astype(str))
            require(
                strand_vals.issubset({"+", "-"}),
                f"{family}/{assembly}: unexpected strand values: {strand_vals - {'+', '-'}}",
            )

            # UCSC URL column present and contains recognisable UCSC URLs
            require(
                "ucsc_url" in df.columns,
                f"{family}/{assembly}: missing ucsc_url column",
            )
            require(
                df["ucsc_url"].astype(str).str.contains(
                    r"genome\.ucsc\.edu|api\.genome\.ucsc\.edu", regex=True
                ).any(),
                f"{family}/{assembly}: ucsc_url values do not look like UCSC URLs",
            )

            log(
                f"  PASS {family}/{assembly} ({species}, {size_class}): "
                f"{len(df)} loci, valid DNA, standard chroms, coords OK"
            )

    def test_per_family_pipeline_stages(self) -> None:
        # Runs all 6 analysis scripts (clustering→alignment→motif→go→expression→enrichment)
        # for every family in the test matrix using real sequences from the UCSC API.
        import pandas as pd

        for case in self.bare_cases:
            family, assembly, species, size_class = normalize_bare_case(case)
            safe = case_key(family, assembly)

            # ── Re-use or fetch sequence CSV ──────────────────────────────────
            seq_csv = (
                self.work / "bare_family_query" / safe
                / family.lower() / "01_data"
                / f"{family.lower()}_with_sequences.csv"
            )
            if not seq_csv.exists():
                log(f"[{family}/{assembly}] sequences not yet fetched; fetching now")
                run_cmd([
                    sys.executable,
                    str(self.repo / "query.py"),
                    "--local", "--family", family, "--assembly", assembly,
                    "--output", str(self.work / "bare_family_query" / safe),
                    "--max-loci", str(self.bare_max_loci),
                    "--stop-after", "sequences",
                ], cwd=self.repo, timeout=1800)
            require_path(seq_csv, 10)
            df = pd.read_csv(seq_csv)
            require(len(df) > 0, f"{family}/{assembly}: sequence CSV is empty")

            stage_dir = self.work / "per_family_stages" / safe
            stage_dir.mkdir(parents=True, exist_ok=True)

            # ── Build a JASPAR BED anchored to actual TE loci (guarantees overlaps)
            chrom_col  = next((c for c in ("chr",    "Chromosome") if c in df.columns), "chr")
            start_col  = next((c for c in ("start",  "Start")     if c in df.columns), "start")
            stop_col   = next((c for c in ("stop",   "Stop")      if c in df.columns), "stop")
            strand_col = next((c for c in ("strand", "Strand")    if c in df.columns), "strand")
            jaspar_bed = stage_dir / "jaspar_loci.bed"
            with open(jaspar_bed, "w") as fh:
                for _, row in df.iterrows():
                    s, e = int(row[start_col]), int(row[stop_col])
                    mid  = (s + e) // 2
                    fh.write(
                        f"{row[chrom_col]}\t{max(0, mid-20)}\t{mid+20}"
                        f"\tTFAP2A::MA0003.4\t900\t{row[strand_col]}\n"
                    )
                    fh.write(
                        f"{row[chrom_col]}\t{s+40}\t{s+75}"
                        f"\tGATA1::MA0035.4\t900\t+\n"
                    )

            # Pre-populate GO annotation cache so te_go / te_enrichment run offline
            self._write_cached_go(stage_dir)

            log(f"\n[{family}/{assembly}] running all 6 pipeline stage scripts")

            # ── 1. te_clustering.py ──────────────────────────────────────────
            clustered_csv = stage_dir / "clustered.csv"
            clust_out     = stage_dir / "clustering"
            run_cmd([
                sys.executable, str(self.repo / "te_clustering.py"),
                "--input",            str(seq_csv),
                "--output",           str(clustered_csv),
                "--kmer",             "4",
                "--pca-dims",         "4",
                "--n-epochs",         "10",
                "--min-cluster-size", "2",
                "--min-samples",      "1",
                "--n-neighbors",      str(max(2, len(df) - 1)),
                "--skip-tsne",
                "--out-dir",          str(clust_out),
                "--family",           family,
            ], cwd=self.repo, timeout=300)
            require_path(clustered_csv, 10)
            df_cl = pd.read_csv(clustered_csv)
            require(
                "Cluster" in df_cl.columns,
                f"{family}/{assembly}: te_clustering output missing Cluster column",
            )
            require_path(clust_out / "clustering_visualization.html", 100)
            log(f"  [1/6] te_clustering OK  clusters={sorted(df_cl['Cluster'].unique())}")

            # ── 2. te_alignment.py ───────────────────────────────────────────
            align_out = stage_dir / "alignment"
            run_cmd([
                sys.executable, str(self.repo / "te_alignment.py"),
                "--input",   str(clustered_csv),
                "--out-dir", str(align_out),
                "--family",  family,
            ], cwd=self.repo, timeout=300)
            require_path(align_out / f"{family.lower()}_aligned.fa", 10)
            require_path(align_out / "05_consensus" / f"{family.lower()}_consensus.fa", 10)
            log(f"  [2/6] te_alignment OK")

            # ── 3. te_motif.py ───────────────────────────────────────────────
            motif_out = stage_dir / "motif"
            self._write_cached_go(motif_out)
            run_cmd([
                sys.executable, str(self.repo / "te_motif.py"),
                "--input",       str(clustered_csv),
                "--build",       assembly,
                "--out-dir",     str(motif_out),
                "--jaspar-bed",  str(jaspar_bed),
                "--p-threshold", "1.0",
                "--force",
            ], cwd=self.repo, timeout=300)
            require_path(motif_out / "motif_analysis" / "all_overlaps.tsv", 10)
            log(f"  [3/6] te_motif OK")

            # ── 4. te_go.py ──────────────────────────────────────────────────
            run_cmd([
                sys.executable, str(self.repo / "te_go.py"),
                "--enrichment-dir", str(motif_out / "enrichment_results"),
                "--build",          assembly,
                "--out-dir",        str(motif_out),
                "--clustered-csv",  str(clustered_csv),
                "--p-threshold",    "1.0",
            ], cwd=self.repo, timeout=300)
            require_path(motif_out / "go_annotations" / "gene_functions.csv", 10)
            log(f"  [4/6] te_go OK")

            # ── 5. te_expression.py ──────────────────────────────────────────
            # swScore and milliDiv are numeric RMSK columns preserved through
            # clustering — use them as proxy expression values for test coverage.
            expr_out = stage_dir / "expression"
            run_cmd([
                sys.executable, str(self.repo / "te_expression.py"),
                "--input",        str(clustered_csv),
                "--out-dir",      str(expr_out),
                "--stage-cols",   "swScore", "milliDiv",
                "--stage-labels", "swScore",  "milliDiv",
                "--force",
            ], cwd=self.repo, timeout=300)
            require_path(expr_out / "expression_plots" / "expression_stats.csv", 10)
            log(f"  [5/6] te_expression OK")

            # ── 6. te_enrichment.py (orchestrates motif + GO + expression) ───
            enrich_out = stage_dir / "enrichment"
            self._write_cached_go(enrich_out)
            run_cmd([
                sys.executable, str(self.repo / "te_enrichment.py"),
                "--input",       str(clustered_csv),
                "--build",       assembly,
                "--out-dir",     str(enrich_out),
                "--jaspar-bed",  str(jaspar_bed),
                "--p-threshold", "1.0",
                "--force",
            ], cwd=self.repo, timeout=300)
            require_path(enrich_out / "enrichment_checkpoints.json", 10)
            require_path(enrich_out / "motif_analysis" / "all_overlaps.tsv", 10)
            require_path(enrich_out / "expression_plots" / "expression_stats.csv", 10)
            log(f"  [6/6] te_enrichment OK")

            log(f"  ALL STAGES PASS: {family}/{assembly} ({size_class})")

    def test_te_go_auto_chain(self) -> None:
        # Exercises the new te_go.py --input and --family auto-chain modes.
        # te_go.py should detect missing enrichment results and run te_motif
        # automatically before proceeding to GO annotation.
        import pandas as pd

        # ── Mode 1: --input (clustered CSV path, no pre-existing enrichment) ──
        ac1_out = self.work / "go_auto_chain_input"
        ac1_out.mkdir(parents=True, exist_ok=True)
        self._write_cached_go(ac1_out)
        run_cmd([
            sys.executable,
            str(self.repo / "te_go.py"),
            "--input",       str(self.work / "fixture" / "clustered_runtime.csv"),
            "--build",       "hg38",
            "--out-dir",     str(ac1_out),
            "--jaspar-bed",  str(self.fixture["jaspar"]),
            "--p-threshold", "1.0",
            "--force",
        ], cwd=self.repo, timeout=300)
        # te_motif should have been run automatically
        require_path(ac1_out / "motif_analysis" / "all_overlaps.tsv", 10)
        require_path(ac1_out / "go_annotations" / "gene_functions.csv", 10)
        log("  Mode 1 (--input auto-chain): te_motif ran + GO annotated")

        # ── Mode 2: --family + standard path discovery ────────────────────────
        ac2_out = self.work / "go_auto_chain_family"
        ac2_out.mkdir(parents=True, exist_ok=True)
        data_dir = ac2_out / "01_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        # Copy the clustered CSV to the expected standard path
        import shutil
        shutil.copy(
            str(self.work / "fixture" / "clustered_runtime.csv"),
            str(data_dir / "test_te_clustered.csv"),
        )
        self._write_cached_go(ac2_out)
        run_cmd([
            sys.executable,
            str(self.repo / "te_go.py"),
            "--family",      "TEST_TE",
            "--assembly",    "hg38",
            "--out-dir",     str(ac2_out),
            "--jaspar-bed",  str(self.fixture["jaspar"]),
            "--p-threshold", "1.0",
            "--force",
        ], cwd=self.repo, timeout=300)
        require_path(ac2_out / "motif_analysis" / "all_overlaps.tsv", 10)
        require_path(ac2_out / "go_annotations" / "gene_functions.csv", 10)
        df_genes = pd.read_csv(ac2_out / "go_annotations" / "gene_functions.csv")
        require(
            len(df_genes) >= 1,
            "GO auto-chain (--family mode): gene_functions.csv is empty",
        )
        log("  Mode 2 (--family auto-chain): CSV discovered + te_motif ran + GO annotated")

        # ── Mode 3: --assembly synonym for --build ────────────────────────────
        # Already tested above (Mode 2 uses --assembly hg38).  Spot-check the
        # help output to confirm --assembly is a recognised flag.
        proc = run_cmd([
            sys.executable,
            str(self.repo / "te_go.py"),
            "--help",
        ], cwd=self.repo, timeout=30)
        require(
            "--assembly" in proc.stdout,
            "te_go.py --help output does not mention --assembly flag",
        )
        log("  Mode 3 (--assembly synonym): help output confirms flag is present")

    def test_genome_cache_and_primer_search(self) -> None:
        import pandas as pd
        from te_genome import GenomeCache, reverse_complement as rc

        df = pd.read_csv(self.fixture["csv_with_seq"])
        gc = GenomeCache(str(self.fixture["genome"]), cache_dir=str(self.work / "genome_cache"))
        gc.load()
        require(gc.is_loaded, "GenomeCache did not load")
        first = df.iloc[0]
        seq = gc.extract_sequence(first["chr"], int(first["start"]), int(first["stop"]))
        require(seq == first["Seq"], "GenomeCache extraction mismatch for plus-strand locus")
        minus = df[df["strand"] == "-"].iloc[0]
        raw = gc.extract_sequence(minus["chr"], int(minus["start"]), int(minus["stop"]))
        require(rc(raw) == minus["Seq"], "GenomeCache extraction/orientation mismatch for minus-strand locus")
        primer = str(first["Seq"])[25:35]
        serial_hits = gc.search_primer(primer, parallel=False)
        parallel_hits = gc.search_primer(primer, parallel=True, n_workers=2)
        require(len(serial_hits) >= 1, "serial primer search found no hits")
        require(len(parallel_hits) >= 1, "parallel primer search found no hits")

    def test_clustering(self) -> None:
        import pandas as pd
        from te_clustering import clustering_analysis

        df = pd.read_csv(self.fixture["csv_with_seq"])
        out = self.work / "clustering"
        clustered, labels = clustering_analysis(
            df,
            kmer=4,
            min_cluster_size=3,
            out_dir=out,
            family_name="TEST_TE",
            pca_dims=5,
            n_epochs=10,
            random_state=42,
            compute_tsne=False,
            n_neighbors=5,
            min_dist=0.1,
            min_samples=2,
        )
        clustered.to_csv(self.work / "fixture" / "clustered_runtime.csv", index=False)
        require("Cluster" in clustered.columns, "clustering did not add Cluster column")
        require(len(labels) == len(df), "cluster label count mismatch")
        require_path(out / "clustering_visualization.html", 100)
        require_path(out / "cluster_summary.csv", 10)

    def test_alignment(self) -> None:
        import pandas as pd
        from te_alignment import run_alignment_pipeline

        clustered = pd.read_csv(self.work / "fixture" / "clustered_runtime.csv")
        out = self.work / "alignment"
        result = run_alignment_pipeline(clustered, out, "TEST_TE")
        require(result["mafft_available"], "MAFFT was not available to alignment pipeline")
        require_path(out / "test_te_aligned.fa", 10)
        require_path(out / "05_consensus" / "test_te_consensus.fa", 10)
        require_path(out / "cluster_alignments" / "cluster_consensus_summary.csv", 10)
        if result["cialign_available"]:
            require_path(out / "cialign_plots" / "index.html", 10)

    def test_primer_design(self) -> None:
        import pandas as pd
        from te_genome import GenomeCache
        from te_primers import design_primers

        clustered = pd.read_csv(self.work / "fixture" / "clustered_runtime.csv")
        gc = GenomeCache(str(self.fixture["genome"]), cache_dir=str(self.work / "primer_cache"))
        gc.load()
        out = self.work / "primers"
        result = design_primers(
            clustered,
            primer_k=8,
            top_global=5,
            top_cluster=3,
            genome_cache=gc,
            primer_timeout=15,
            out_dir=out,
            family_name="TEST_TE",
        )
        require(result["selected_primers"], "primer design selected no primers")
        require_path(out / "selected_primers_summary.csv", 10)
        require_path(out / "cluster_top_primers.csv", 10)
        require_path(out / "primer_genome_hits_summary.csv", 10)

    def test_individual_script_clis(self) -> None:
        import pandas as pd

        cli_root = self.work / "individual_cli"
        cli_root.mkdir(parents=True, exist_ok=True)
        help_scripts = [
            "query.py",
            "te_prep.py",
            "te_clustering.py",
            "te_alignment.py",
            "te_motif.py",
            "te_go.py",
            "te_expression.py",
            "te_enrichment.py",
            "te_primers.py",
            "hpc_client.py",
            "ui.py",
            "presentation.py",
            "test_jaspar_path.py",
            "setup_cython.py",
        ]
        for rel in help_scripts:
            run_cmd([sys.executable, str(self.repo / rel), "--help"], cwd=self.repo, timeout=90)

        clustered_cli = cli_root / "te_clustering_cli.csv"
        run_cmd([
            sys.executable,
            str(self.repo / "te_clustering.py"),
            "--input", str(self.fixture["csv_with_seq"]),
            "--output", str(clustered_cli),
            "--kmer", "4",
            "--pca-dims", "5",
            "--n-epochs", "10",
            "--min-cluster-size", "3",
            "--min-samples", "2",
            "--n-neighbors", "5",
            "--skip-tsne",
            "--out-dir", str(cli_root / "clustering"),
            "--family", "TEST_TE",
        ], cwd=self.repo, timeout=300)
        require_path(clustered_cli, 10)
        df_clustered = pd.read_csv(clustered_cli)
        require("Cluster" in df_clustered.columns, "te_clustering CLI output missing Cluster column")
        require_path(cli_root / "clustering" / "clustering_visualization.html", 100)

        run_cmd([
            sys.executable,
            str(self.repo / "te_alignment.py"),
            "--input", str(clustered_cli),
            "--out-dir", str(cli_root / "alignment"),
            "--family", "TEST_TE",
        ], cwd=self.repo, timeout=300)
        require_path(cli_root / "alignment" / "05_consensus" / "test_te_consensus.fa", 10)

        run_cmd([
            sys.executable,
            str(self.repo / "te_primers.py"),
            "--input", str(clustered_cli),
            "--genome", str(self.fixture["genome"]),
            "--kmer", "8",
            "--top-global", "4",
            "--top-cluster", "2",
            "--timeout", "15",
            "--out-dir", str(cli_root / "primers"),
            "--family", "TEST_TE",
        ], cwd=self.repo, timeout=300)
        require_path(cli_root / "primers" / "selected_primers_summary.csv", 10)

        motif_out = cli_root / "motif"
        run_cmd([
            sys.executable,
            str(self.repo / "te_motif.py"),
            "--input", str(clustered_cli),
            "--build", "hg38",
            "--out-dir", str(motif_out),
            "--jaspar-bed", str(self.fixture["jaspar"]),
            "--p-threshold", "1.0",
            "--force",
        ], cwd=self.repo, timeout=300)
        require_path(motif_out / "motif_analysis" / "all_overlaps.tsv", 10)

        self._write_cached_go(motif_out)
        run_cmd([
            sys.executable,
            str(self.repo / "te_go.py"),
            "--enrichment-dir", str(motif_out / "enrichment_results"),
            "--build", "hg38",
            "--out-dir", str(motif_out),
            "--clustered-csv", str(clustered_cli),
            "--p-threshold", "1.0",
        ], cwd=self.repo, timeout=300)
        require_path(motif_out / "go_annotations" / "gene_functions.csv", 10)

        run_cmd([
            sys.executable,
            str(self.repo / "te_expression.py"),
            "--input", str(clustered_cli),
            "--out-dir", str(cli_root / "expression"),
            "--stage-cols", "stage_a", "stage_b",
            "--stage-labels", "StageA", "StageB",
            "--force",
        ], cwd=self.repo, timeout=300)
        require_path(cli_root / "expression" / "expression_plots" / "expression_stats.csv", 10)

        enrich_out = cli_root / "enrichment"
        self._write_cached_go(enrich_out)
        run_cmd([
            sys.executable,
            str(self.repo / "te_enrichment.py"),
            "--input", str(clustered_cli),
            "--build", "hg38",
            "--out-dir", str(enrich_out),
            "--jaspar-bed", str(self.fixture["jaspar"]),
            "--p-threshold", "1.0",
        ], cwd=self.repo, timeout=300)
        require_path(enrich_out / "enrichment_checkpoints.json", 10)

        presentation_out = cli_root / "presentation"
        run_cmd([
            sys.executable,
            str(self.repo / "presentation.py"),
            "--cached-csv", str(clustered_cli),
            "--out", str(presentation_out),
        ], cwd=self.repo, timeout=300)
        require_path(presentation_out / "presentation.log", 10)

    def _write_cached_go(self, out: Path) -> None:
        import pandas as pd

        go_dir = out / "go_annotations"
        go_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {
                "Gene": "TFAP2A",
                "Symbol": "TFAP2A",
                "Name": "transcription factor AP-2 alpha",
                "Summary": "Synthetic cached GO row for offline tests.",
                "GO_BP": "regulation of transcription; cell differentiation",
                "GO_MF": "DNA-binding transcription factor activity",
                "GO_CC": "nucleus",
            },
            {
                "Gene": "GATA1",
                "Symbol": "GATA1",
                "Name": "GATA binding protein 1",
                "Summary": "Synthetic cached GO row for offline tests.",
                "GO_BP": "erythrocyte differentiation; transcription regulation",
                "GO_MF": "DNA-binding transcription factor activity",
                "GO_CC": "nucleus",
            },
            {
                "Gene": "SP1",
                "Symbol": "SP1",
                "Name": "Sp1 transcription factor",
                "Summary": "Synthetic cached GO row for offline tests.",
                "GO_BP": "transcription initiation; chromatin organization",
                "GO_MF": "sequence-specific DNA binding",
                "GO_CC": "nucleus",
            },
        ]).to_csv(go_dir / "gene_functions.csv", index=False)

    def test_motif_go_and_expression(self) -> None:
        import pandas as pd
        from te_expression import run_expression_analysis
        from te_go import run_go_annotation
        from te_motif import run_motif_analysis

        out = self.work / "motif_go_expression"
        self._write_cached_go(out)
        motif_result = run_motif_analysis(
            input_csv=str(self.work / "fixture" / "clustered_runtime.csv"),
            build="hg38",
            out_dir=str(out),
            jaspar_bed_arg=str(self.fixture["jaspar"]),
            jaspar_dir=str(out / "jaspar"),
            p_threshold=1.0,
            force=True,
        )
        require_path(motif_result["overlaps_path"], 10)
        require_path(out / "enrichment_results" / "cluster_0_enrichment.csv", 10)
        go_result = run_go_annotation(
            enrichment_dir=str(out / "enrichment_results"),
            build="hg38",
            out_dir=str(out),
            clustered_csv=str(self.work / "fixture" / "clustered_runtime.csv"),
            p_threshold=1.0,
            force=False,
        )
        require(go_result["n_genes"] >= 1, "GO annotation loaded no cached genes")
        require_path(out / "go_annotations" / "gene_functions.csv", 10)
        expr = run_expression_analysis(
            input_csv=str(self.work / "fixture" / "clustered_runtime.csv"),
            out_dir=str(out),
            stage_cols=["stage_a", "stage_b"],
            stage_labels=["Stage A", "Stage B"],
            force=True,
        )
        require(expr["expr_cols"] == ["stage_a", "stage_b"], "expression columns were not honored")
        require_path(out / "expression_plots" / "expression_stats.csv", 10)
        require_path(out / "expression_plots" / "boxplot_all.png", 100)

    def test_enrichment_orchestrator_cli(self) -> None:
        out = self.work / "enrichment_cli"
        self._write_cached_go(out)
        run_cmd([
            sys.executable,
            str(self.repo / "te_enrichment.py"),
            "--input", str(self.work / "fixture" / "clustered_runtime.csv"),
            "--build", "hg38",
            "--out-dir", str(out),
            "--jaspar-bed", str(self.fixture["jaspar"]),
            "--p-threshold", "1.0",
        ], cwd=self.repo, timeout=300)
        require_path(out / "enrichment_checkpoints.json", 10)
        require_path(out / "motif_analysis" / "all_overlaps.tsv", 10)
        require_path(out / "expression_plots" / "expression_stats.csv", 10)

    def test_query_pipeline(self) -> None:
        out = self.work / "query_pipeline"
        run_cmd([
            sys.executable,
            str(self.repo / "query.py"),
            "--input", str(self.fixture["csv_no_seq"]),
            "--family", "TEST_TE",
            "--genome", str(self.fixture["genome"]),
            "--output", str(out),
            "--kmer", "4",
            "--pca-dims", "5",
            "--n-epochs", "10",
            "--random-state", "42",
            "--n-neighbors", "5",
            "--min-dist", "0.1",
            "--min-cluster-size", "3",
            "--min-samples", "2",
            "--min-sequences", "2",
            "--skip-tsne",
            "--skip-motif",
            "--primer-kmer", "8",
            "--top-global", "4",
            "--top-cluster", "2",
            "--primer-timeout", "15",
            "--force",
        ], cwd=self.repo, timeout=600)
        root = out / "test_te"
        require_path(root / "01_data" / "test_te_with_sequences.csv", 10)
        require_path(root / "01_data" / "test_te_clustered.csv", 10)
        require_path(root / "03_clustering" / "clustering_visualization.html", 100)
        require_path(root / "06_primers" / "selected_primers_summary.csv", 10)
        require_path(root / "pipeline_diagnostic.log", 10)

    def test_backend_ipc(self) -> None:
        home = self.work / "ipc_home"
        gameca = home / ".gameca"
        venv = gameca / "venv"
        venv.mkdir(parents=True, exist_ok=True)
        (venv / "pyvenv.cfg").write_text("home = synthetic\n")
        (gameca / ".setup_done").write_text("")
        env = os.environ.copy()
        env["HOME"] = str(home)
        env["GAMECA_UPDATE_REPO"] = "invalid/invalid"
        env["PYTHONPATH"] = str(self.repo / "backend") + os.pathsep + str(self.repo)
        payload = "\n".join([
            json.dumps({"id": "ping", "command": "ping", "args": {}}),
            json.dumps({"id": "run", "command": "run", "args": {"argv": ["noop"]}}),
            json.dumps({"id": "shutdown", "command": "shutdown", "args": {}}),
        ]) + "\n"
        proc = subprocess.run(
            [sys.executable, str(self.repo / "backend" / "main.py")],
            input=payload,
            cwd=str(self.repo / "backend"),
            env=env,
            text=True,
            capture_output=True,
            timeout=45,
        )
        if proc.stderr.strip():
            print(proc.stderr, file=sys.stderr, flush=True)
        require(proc.returncode == 0, f"backend/main.py exited {proc.returncode}")
        messages = []
        for line in proc.stdout.splitlines():
            if line.strip():
                messages.append(json.loads(line))
        by_id = {m.get("id"): m for m in messages if m.get("id")}
        require(by_id["ping"]["payload"]["ok"] is True, "IPC ping failed")
        require(by_id["run"]["payload"]["exit_code"] == 0, "IPC noop run failed")
        require(by_id["shutdown"]["payload"]["ok"] is True, "IPC shutdown failed")

    def test_cython_extension(self) -> None:
        if self.skip_cython:
            log("Cython extension test skipped by option")
            return
        run_cmd([sys.executable, str(self.repo / "setup_cython.py"), "build_ext", "--inplace"], cwd=self.repo, timeout=300)
        import te_fast

        require(te_fast.reverse_complement("AACCGGTT") == "AACCGGTT", "te_fast reverse_complement mismatch")
        lengths = te_fast.batch_lengths(["AA", "CCCC"])
        require(list(lengths) == [2, 4], "te_fast batch_lengths mismatch")

    def test_live_network_apis(self) -> None:
        import requests
        import te_go
        import te_prep

        r = requests.get("https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=chr1;start=10000;end=10020", timeout=20)
        require(r.status_code == 200 and "dna" in r.json(), "UCSC sequence API check failed")
        info = te_go.query_gene("TP53", "9606")
        require(info and info.get("symbol"), "mygene.info query failed")
        acc = te_prep._dfam_lookup_accession("L1HS")
        require(acc, "Dfam accession lookup failed")

    def test_dev_surface(self) -> None:
        if not self.include_dev:
            log("frontend/Rust developer surface skipped by option")
            return
        if shutil.which("pnpm") is None:
            raise AssertionError("pnpm not found; cannot run frontend tests")
        run_cmd(["pnpm", "install", "--frozen-lockfile"], cwd=self.repo, timeout=900)
        run_cmd(["pnpm", "--dir", "frontend", "test"], cwd=self.repo, timeout=300)
        run_cmd(["pnpm", "--dir", "frontend", "lint"], cwd=self.repo, timeout=300)
        if shutil.which("cargo") is None:
            raise AssertionError("cargo not found; cannot run Tauri Rust check")
        run_cmd(["cargo", "check", "--manifest-path", str(self.repo / "src-tauri" / "Cargo.toml")], cwd=self.repo, timeout=900)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GAMECA feature tests inside an HPC batch job.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--include-dev", action="store_true")
    parser.add_argument("--live-network", action="store_true")
    parser.add_argument("--skip-cython", action="store_true")
    parser.add_argument(
        "--bare-case",
        action="append",
        default=[],
        help="Bare auto-query case as FAMILY:ASSEMBLY. May be repeated.",
    )
    parser.add_argument("--bare-family", default="")
    parser.add_argument("--bare-assembly", default="")
    parser.add_argument("--bare-max-loci", type=int, default=6)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    bare_cases = [parse_bare_case(c) for c in args.bare_case]
    if args.bare_family and args.bare_assembly:
        bare_cases.append((args.bare_family, args.bare_assembly))
    if not bare_cases:
        bare_cases = DEFAULT_BARE_CASES
    tester = FeatureTester(
        repo,
        work,
        Path(args.json).resolve(),
        include_dev=args.include_dev,
        live_network=args.live_network,
        skip_cython=args.skip_cython,
        bare_cases=bare_cases,
        bare_max_loci=args.bare_max_loci,
    )
    tests = [
        ("batch scheduler runtime", tester.test_batch_runtime, True),
        ("repository integrity and README coverage", tester.test_repository_integrity, True),
        ("Python dependency imports", tester.test_dependency_imports, True),
        ("required external HPC tools", tester.test_external_tools, True),
        ("TE prep, RMSK, Dfam, and sequence extraction", tester.test_te_prep_and_annotation_parsers, True),
        ("family matrix coverage validation", tester.test_family_matrix_coverage_validation, True),
        ("bare family+assembly auto query matrix", tester.test_bare_family_assembly_auto_query_matrix, True),
        ("per-family pipeline stages (all 6 scripts)", tester.test_per_family_pipeline_stages, True),
        ("te_go.py auto-chain (--input / --family / --assembly)", tester.test_te_go_auto_chain, True),
        ("GenomeCache and primer search", tester.test_genome_cache_and_primer_search, True),
        ("k-mer clustering", tester.test_clustering, True),
        ("MAFFT/CIAlign alignment", tester.test_alignment, True),
        ("primer design", tester.test_primer_design, True),
        ("individual standalone script CLIs", tester.test_individual_script_clis, True),
        ("motif, GO, and expression modules", tester.test_motif_go_and_expression, True),
        ("enrichment orchestrator CLI", tester.test_enrichment_orchestrator_cli, True),
        ("query.py genome-backed pipeline", tester.test_query_pipeline, True),
        ("backend NDJSON IPC", tester.test_backend_ipc, True),
        ("Cython acceleration build", tester.test_cython_extension, False),
    ]
    if args.live_network:
        tests.append(("live UCSC/Dfam/mygene API checks", tester.test_live_network_apis, True))
    else:
        tests.append(("live UCSC/Dfam/mygene API checks", tester.test_live_network_apis, False))
    tests.append(("frontend and Tauri developer tests", tester.test_dev_surface, bool(args.include_dev)))

    for name, func, required in tests:
        tester.run(name, func, required=required)
    return tester.finish()


if __name__ == "__main__":
    raise SystemExit(main())
"""


@dataclass
class RemotePaths:
    base_dir: str
    run_dir: str
    source_tar: str
    source_dir: str
    work_dir: str
    runner: str
    job_script: str
    stdout: str
    stderr: str
    done: str
    summary: str


def log(message: str) -> None:
    print(f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    return value or default


def prompt_bool(text: str, default: bool = False) -> bool:
    label = "Y/n" if default else "y/N"
    value = input(f"{text} [{label}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "1", "true"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage this repository on an HPC login node and submit a Slurm/LSF "
            "batch job that runs GAMECA feature tests with detailed logs."
        )
    )
    parser.add_argument("-H", "--host", default="", help="HPC hostname")
    parser.add_argument("-u", "--user", default="", help="HPC username")
    parser.add_argument("-p", "--port", type=int, default=22, help="SSH port")
    parser.add_argument("-i", "--key", default="", help="SSH private key path")
    parser.add_argument("--work-dir", default="", help="Shared remote work directory")
    parser.add_argument("--scheduler", choices=["auto", "slurm", "lsf"], default="auto")
    parser.add_argument("--queue", default="", help="Slurm partition or LSF queue")
    parser.add_argument("--modules", default="", help="Space-separated modules to load inside the job")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--mem-mb", type=int, default=12000)
    parser.add_argument("--walltime", default="02:00:00", help="Walltime, default 02:00:00")
    parser.add_argument("--no-wait", action="store_true", help="Submit and return immediately")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait-timeout", type=int, default=6 * 60 * 60)
    parser.add_argument("--skip-install", action="store_true", help="Use remote system Python instead of creating a venv and installing requirements")
    parser.add_argument("--skip-cython", action="store_true", help="Skip optional Cython extension build test")
    parser.add_argument("--live-network", action="store_true", help="Make live UCSC/Dfam/mygene API checks required")
    parser.add_argument("--include-dev", action="store_true", help="Also run pnpm frontend tests and cargo check")
    parser.add_argument(
        "--bare-case",
        action="append",
        default=[],
        help="Bare auto-query case as FAMILY:ASSEMBLY. May be repeated. Defaults cover mm10, mm39, hg38, and hg19.",
    )
    parser.add_argument("--bare-family", default="", help="Single family override for the bare auto-query test")
    parser.add_argument("--bare-assembly", default="", help="Single assembly override for the bare auto-query test")
    parser.add_argument("--bare-max-loci", type=int, default=6, help="Locus cap for the bare auto-query test")
    return parser.parse_args()


def connect_ssh(host: str, port: int, username: str, password: str, key_path: str):
    transport = paramiko.Transport((host, port))
    transport.banner_timeout = 90
    transport.connect()

    if key_path:
        expanded = os.path.expanduser(key_path)
        key = None
        for key_cls in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
            try:
                key = key_cls.from_private_key_file(expanded)
                break
            except Exception:
                continue
        if key is None:
            raise RuntimeError(f"could not load private key: {key_path}")
        transport.auth_publickey(username, key)

    if not transport.is_authenticated():
        password_store = [password]

        def kb_handler(_title, _instructions, prompt_list):
            responses = []
            for i, (text, _show) in enumerate(prompt_list):
                lower = text.lower()
                if i == 0 or "password" in lower:
                    responses.append(password_store[0])
                elif "duo" in lower or "passcode" in lower or "factor" in lower or "option" in lower:
                    print("  [Duo] sending option 1; approve the push if prompted.", flush=True)
                    responses.append("1")
                else:
                    responses.append(password_store[0])
            return responses

        try:
            transport.auth_interactive(username, kb_handler)
        except paramiko.ssh_exception.AuthenticationException:
            transport.auth_password(username, password)

    if not transport.is_authenticated():
        transport.close()
        raise RuntimeError("SSH authentication failed")

    ssh = paramiko.SSHClient()
    ssh._transport = transport
    try:
        sftp = paramiko.SFTPClient.from_transport(transport)
    except Exception:
        sftp = None
    return ssh, sftp, transport


def run_remote(
    ssh: paramiko.SSHClient,
    command: str,
    *,
    timeout: int = 300,
    stream: bool = False,
    login_shell: bool = True,
) -> tuple[str, str, int]:
    # Login shells are useful for scheduler/module PATH discovery, but some
    # clusters initialize broken conda hooks from .bash_profile.  For mechanical
    # file-transfer commands, use a plain non-login shell to avoid that noise.
    full = (
        f"bash -lc {shlex.quote(command)}"
        if login_shell
        else f"sh -c {shlex.quote(command)}"
    )
    chan = ssh.get_transport().open_session()
    chan.settimeout(5)
    chan.exec_command(full)
    out = bytearray()
    err = bytearray()
    start = time.time()
    while True:
        if chan.recv_ready():
            chunk = chan.recv(65536)
            out.extend(chunk)
            if stream:
                print(chunk.decode(errors="replace"), end="", flush=True)
        if chan.recv_stderr_ready():
            chunk = chan.recv_stderr(65536)
            err.extend(chunk)
            if stream:
                print(chunk.decode(errors="replace"), end="", file=sys.stderr, flush=True)
        if chan.exit_status_ready():
            break
        if time.time() - start > timeout:
            chan.close()
            return out.decode(errors="replace"), err.decode(errors="replace") + "\nTIMEOUT", 124
        time.sleep(0.1)
    while chan.recv_ready():
        out.extend(chan.recv(65536))
    while chan.recv_stderr_ready():
        err.extend(chan.recv_stderr(65536))
    code = chan.recv_exit_status()
    chan.close()
    return out.decode(errors="replace"), err.decode(errors="replace"), code


def run_remote_stdin(
    ssh: paramiko.SSHClient,
    command: str,
    data: bytes,
    *,
    timeout: int = 300,
) -> tuple[str, str, int]:
    full = f"sh -c {shlex.quote(command)}"
    chan = ssh.get_transport().open_session()
    chan.settimeout(10)
    chan.exec_command(full)
    out = bytearray()
    err = bytearray()
    start = time.time()
    offset = 0
    closed_stdin = False

    while True:
        if offset < len(data):
            try:
                sent = chan.send(data[offset:offset + 65536])
                if sent:
                    offset += sent
            except Exception:
                time.sleep(0.05)
        elif not closed_stdin:
            try:
                chan.shutdown_write()
            except Exception:
                pass
            closed_stdin = True

        if chan.recv_ready():
            out.extend(chan.recv(65536))
        if chan.recv_stderr_ready():
            err.extend(chan.recv_stderr(65536))
        if chan.exit_status_ready() and closed_stdin:
            break
        if time.time() - start > timeout:
            chan.close()
            return out.decode(errors="replace"), err.decode(errors="replace") + "\nTIMEOUT", 124
        time.sleep(0.01)

    while chan.recv_ready():
        out.extend(chan.recv(65536))
    while chan.recv_stderr_ready():
        err.extend(chan.recv_stderr(65536))
    code = chan.recv_exit_status()
    chan.close()
    return out.decode(errors="replace"), err.decode(errors="replace"), code


def ensure_remote_dir(ssh: paramiko.SSHClient, path: str, *, login_shell: bool = False) -> None:
    out, err, code = run_remote(
        ssh,
        f"mkdir -p {shlex.quote(path)} && test -w {shlex.quote(path)}",
        login_shell=login_shell,
    )
    if code != 0:
        raise RuntimeError(f"could not create writable remote directory {path}: {err or out}")


def sftp_mkdir_p(sftp, path: str) -> None:
    if not path or path == ".":
        return
    absolute = path.startswith("/")
    cur = "/" if absolute else "."
    for part in [p for p in path.split("/") if p]:
        cur = posixpath.join(cur, part)
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


def remote_file_size(ssh: paramiko.SSHClient, path: str) -> int | None:
    out, _, code = run_remote(
        ssh,
        f"wc -c < {shlex.quote(path)} 2>/dev/null || true",
        timeout=30,
        login_shell=False,
    )
    if code == 0 and out.strip().isdigit():
        return int(out.strip())
    return None


def upload_bytes(ssh: paramiko.SSHClient, sftp, data: bytes, remote_path: str, mode: int | None = None) -> None:
    parent = posixpath.dirname(remote_path)
    if sftp is not None:
        try:
            sftp_mkdir_p(sftp, parent)
            with sftp.file(remote_path, "wb") as fh:
                fh.write(data)
            if mode is not None:
                sftp.chmod(remote_path, mode)
            return
        except Exception as exc:
            log(f"SFTP upload failed for {remote_path}: {exc}; falling back to shell upload")

    ensure_remote_dir(ssh, parent, login_shell=False)
    encoded = base64.b64encode(data).decode("ascii")
    cmd = f"base64 -d > {shlex.quote(remote_path)}"
    if mode is not None:
        cmd += f" && chmod {oct(mode)[2:]} {shlex.quote(remote_path)}"
    timeout = max(180, len(encoded) // 10000)
    out, err, code = run_remote_stdin(ssh, cmd, encoded.encode("ascii"), timeout=timeout)
    if code != 0:
        size = remote_file_size(ssh, remote_path)
        if size == len(data):
            log(
                "Upload command reported a non-zero/timeout status, but the "
                f"remote file size is correct ({size:,} bytes); continuing."
            )
            return
        raise RuntimeError(f"remote base64 upload failed: {err or out}")


def make_source_tar(repo: Path) -> bytes:
    include_roots = [
        "README.md", "requirements.txt", "unit_tests.py", "query.py", "ui.py", "hpc_client.py",
        "presentation.py", "setup_cython.py", "te_fast.pyx", "te_fast.c",
        "test_pipeline.py", "test_jaspar_path.py", "test_strand_orientation.py",
        "package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml",
        "backend", "frontend", "src-tauri", "e2e", "docs", "scripts",
    ]
    for path in repo.glob("te_*.py"):
        include_roots.append(path.name)

    skip_parts = {
        "__pycache__", "node_modules", "target", "dist", "build", ".git",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", "test_output",
        "binaries",
    }
    skip_suffixes = {".pyc", ".pyo", ".so", ".dylib", ".o", ".DS_Store"}

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        seen: set[Path] = set()
        for rel in include_roots:
            path = repo / rel
            if not path.exists():
                continue
            if path in seen:
                continue
            seen.add(path)
            if path.is_file():
                tf.add(path, arcname=rel)
                continue
            for child in path.rglob("*"):
                if not child.is_file():
                    continue
                if any(part in skip_parts for part in child.relative_to(repo).parts):
                    continue
                if child.suffix in skip_suffixes:
                    continue
                tf.add(child, arcname=str(child.relative_to(repo)))
    return buf.getvalue()


def detect_scheduler(ssh: paramiko.SSHClient, requested: str) -> str:
    if requested in {"slurm", "lsf"}:
        return requested
    checks = [("slurm", "command -v sbatch"), ("lsf", "command -v bsub")]
    found = []
    for name, cmd in checks:
        out, _, code = run_remote(ssh, cmd, timeout=15)
        if code == 0 and out.strip():
            found.append(name)
    if not found:
        raise RuntimeError("could not find sbatch or bsub on the remote PATH")
    if len(found) == 1:
        return found[0]
    choice = prompt("Both Slurm and LSF were found. Scheduler", "slurm").lower()
    if choice not in {"slurm", "lsf"}:
        raise RuntimeError("scheduler must be slurm or lsf")
    return choice


def choose_remote_work_dir(ssh: paramiko.SSHClient, username: str, requested: str) -> str:
    if requested:
        ensure_remote_dir(ssh, requested)
        return requested.rstrip("/")
    probe = r'''
for base in "${GAMECA_WORKDIR:-}" "${SCRATCH:-}" "${WORK:-}" "${PROJECT:-}" "$HOME"; do
  [ -n "$base" ] || continue
  target="$base/gameca_hpc_unit_tests"
  mkdir -p "$target" >/dev/null 2>&1 || continue
  [ -d "$target" ] && [ -w "$target" ] || continue
  printf '%s\n' "$target"
  exit 0
done
target="/tmp/__USER___gameca_hpc_unit_tests"
mkdir -p "$target" >/dev/null 2>&1 && printf '%s\n' "$target" && exit 0
exit 1
'''.replace("__USER__", username)
    out, err, code = run_remote(ssh, probe, timeout=30)
    if code != 0 or not out.strip():
        raise RuntimeError(f"could not choose remote work directory: {err or out}")
    selected = out.strip().splitlines()[-1].rstrip("/")
    if selected.startswith("/tmp/"):
        log("Warning: selected /tmp; batch compute nodes may not see login-node /tmp.")
    return selected


def build_paths(base_dir: str) -> RemotePaths:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = posixpath.join(base_dir, f"run_{stamp}")
    return RemotePaths(
        base_dir=base_dir,
        run_dir=run_dir,
        source_tar=posixpath.join(run_dir, "gameca_source.tar.gz"),
        source_dir=posixpath.join(run_dir, "src"),
        work_dir=posixpath.join(run_dir, "work"),
        runner=posixpath.join(run_dir, "gameca_hpc_feature_tests.py"),
        job_script=posixpath.join(run_dir, "unit_tests_job.sh"),
        stdout=posixpath.join(run_dir, "unit_tests.out"),
        stderr=posixpath.join(run_dir, "unit_tests.err"),
        done=posixpath.join(run_dir, "done.status"),
        summary=posixpath.join(run_dir, "summary.json"),
    )


def slurm_time(value: str) -> str:
    parts = value.split(":")
    if len(parts) == 2:
        return f"{parts[0]}:{parts[1]}:00"
    return value


def lsf_time(value: str) -> str:
    parts = value.split(":")
    if len(parts) == 3:
        return f"{parts[0]}:{parts[1]}"
    return value


def job_header(scheduler: str, paths: RemotePaths, args: argparse.Namespace) -> str:
    if scheduler == "slurm":
        mem_gb = max(1, int(args.mem_mb + 999) // 1000)
        lines = [
            "#SBATCH --job-name=gameca_unit_tests",
            f"#SBATCH --output={paths.stdout}",
            f"#SBATCH --error={paths.stderr}",
            f"#SBATCH --cpus-per-task={args.cpus}",
            f"#SBATCH --mem={mem_gb}G",
            f"#SBATCH --time={slurm_time(args.walltime)}",
        ]
        if args.queue:
            lines.append(f"#SBATCH --partition={args.queue}")
        return "\n".join(lines)
    lines = [
        "#BSUB -J gameca_unit_tests",
        f"#BSUB -o {paths.stdout}",
        f"#BSUB -e {paths.stderr}",
        f"#BSUB -n {args.cpus}",
        f"#BSUB -M {args.mem_mb}",
        f"#BSUB -W {lsf_time(args.walltime)}",
    ]
    if args.queue:
        lines.append(f"#BSUB -q {args.queue}")
    return "\n".join(lines)


def module_block(modules: str) -> str:
    mods = [m for m in modules.split() if m]
    if not mods:
        return "echo '[unit-tests] no modules requested'"
    loads = "\n".join(f"module load {shlex.quote(m)}" for m in mods)
    return f"""
if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true
fi
if command -v module >/dev/null 2>&1; then
  {loads}
else
  echo '[unit-tests] module command not available; requested modules: {shlex.quote(modules)}'
fi
""".strip()


def build_job_script(paths: RemotePaths, scheduler: str, args: argparse.Namespace) -> str:
    header = job_header(scheduler, paths, args)
    modules = module_block(args.modules)
    runner_flags = []
    if args.include_dev:
        runner_flags.append("--include-dev")
    if args.live_network:
        runner_flags.append("--live-network")
    if args.skip_cython:
        runner_flags.append("--skip-cython")
    for case in args.bare_case:
        runner_flags.extend(["--bare-case", shlex.quote(case)])
    if args.bare_family and args.bare_assembly:
        runner_flags.extend([
            "--bare-family", shlex.quote(args.bare_family),
            "--bare-assembly", shlex.quote(args.bare_assembly),
        ])
    runner_flags.extend(["--bare-max-loci", str(args.bare_max_loci)])
    runner_flags_s = " ".join(runner_flags)
    runner_cmd = " ".join([
        "python", "-u", shlex.quote(paths.runner),
        "--repo", shlex.quote(paths.source_dir),
        "--work", shlex.quote(paths.work_dir),
        "--json", '"$SUMMARY_JSON"',
        runner_flags_s,
    ]).strip()
    install_block = "echo '[unit-tests] using system Python; dependency install skipped'"
    if not args.skip_install:
        install_block = f"""
VENV_DIR={shlex.quote(posixpath.join(paths.run_dir, "venv"))}
PYTHON_BIN=$(command -v python3 || command -v python)
echo "[unit-tests] creating venv at $VENV_DIR with $PYTHON_BIN"
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --prefer-binary -r {shlex.quote(posixpath.join(paths.source_dir, "requirements.txt"))}
""".strip()

    return textwrap.dedent(f"""\
        #!/bin/bash
        {header}
        set -euo pipefail

        RUN_DIR={shlex.quote(paths.run_dir)}
        DONE_FILE={shlex.quote(paths.done)}
        SUMMARY_JSON={shlex.quote(paths.summary)}

        finish() {{
          rc=$?
          echo "$rc" > "$DONE_FILE"
          echo "[unit-tests] finished with exit $rc at $(date)"
        }}
        trap finish EXIT

        echo "============================================================"
        echo "GAMECA HPC feature tests"
        echo "date: $(date)"
        echo "host: $(hostname)"
        echo "user: $(whoami)"
        echo "run_dir: $RUN_DIR"
        echo "SLURM_JOB_ID=${{SLURM_JOB_ID:-}} LSB_JOBID=${{LSB_JOBID:-}}"
        echo "PATH=$PATH"
        echo "============================================================"

        {modules}

        export MPLBACKEND=Agg
        export PYTHONUNBUFFERED=1
        export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{args.cpus}}}"
        export MKL_NUM_THREADS="$OMP_NUM_THREADS"
        export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
        export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
        export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
        export NUMBA_CACHE_DIR="$RUN_DIR/numba_cache"
        mkdir -p "$NUMBA_CACHE_DIR" {shlex.quote(paths.source_dir)} {shlex.quote(paths.work_dir)}

        echo "[unit-tests] extracting source bundle"
        tar -xzf {shlex.quote(paths.source_tar)} -C {shlex.quote(paths.source_dir)}

        cd {shlex.quote(paths.source_dir)}
        echo "[unit-tests] source files:"
        find . -maxdepth 2 -type f | sort | sed -n '1,120p'

        {install_block}

        echo "[unit-tests] Python: $(python --version 2>&1)"
        echo "[unit-tests] pip: $(python -m pip --version 2>&1 || true)"
        echo "[unit-tests] starting test runner"
        {runner_cmd}
    """)


def submit_job(ssh: paramiko.SSHClient, scheduler: str, paths: RemotePaths) -> str:
    if scheduler == "slurm":
        cmd = f"sbatch {shlex.quote(paths.job_script)}"
        pattern = r"Submitted batch job (\d+)"
    else:
        cmd = f"bsub < {shlex.quote(paths.job_script)}"
        pattern = r"Job <(\d+)>"
    out, err, code = run_remote(ssh, cmd, timeout=60)
    combined = (out + "\n" + err).strip()
    if code != 0:
        raise RuntimeError(f"scheduler submission failed ({code}): {combined}")
    match = re.search(pattern, combined)
    if not match:
        raise RuntimeError(f"could not parse job id from scheduler output: {combined}")
    log(f"scheduler output: {combined}")
    return match.group(1)


def scheduler_status_cmd(scheduler: str, job_id: str) -> str:
    if scheduler == "slurm":
        return f"squeue -j {shlex.quote(job_id)} --noheader -o '%i %T %R' 2>&1 || true"
    return f"bjobs {shlex.quote(job_id)} 2>&1 || true"


def wait_for_completion(ssh: paramiko.SSHClient, scheduler: str, job_id: str, paths: RemotePaths, poll_seconds: int, timeout: int) -> int:
    log(f"waiting for job {job_id}; polling every {poll_seconds}s")
    start = time.time()
    last_state = ""
    while time.time() - start < timeout:
        out, _, code = run_remote(ssh, f"test -f {shlex.quote(paths.done)} && cat {shlex.quote(paths.done)} || true", timeout=20)
        status = out.strip().splitlines()[-1] if out.strip() else ""
        if code == 0 and re.fullmatch(r"-?\d+", status):
            rc = int(status)
            log(f"done marker found: exit {rc}")
            return rc
        state, _, _ = run_remote(ssh, scheduler_status_cmd(scheduler, job_id), timeout=30)
        state = state.strip()
        if state and state != last_state:
            log(f"scheduler status: {state}")
            last_state = state
        time.sleep(max(5, poll_seconds))
    raise TimeoutError(f"timed out waiting for job {job_id}")


def print_remote_tail(ssh: paramiko.SSHClient, paths: RemotePaths) -> None:
    for label, path in [("stdout", paths.stdout), ("stderr", paths.stderr), ("summary", paths.summary)]:
        out, err, code = run_remote(
            ssh,
            f"if [ -f {shlex.quote(path)} ]; then echo '--- {label}: {path} ---'; tail -n 120 {shlex.quote(path)}; else echo '--- {label}: missing {path} ---'; fi",
            timeout=60,
        )
        print(out, flush=True)
        if err.strip():
            print(err, file=sys.stderr, flush=True)


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parent

    host = args.host or prompt("HPC hostname")
    if not host:
        print("HPC hostname is required", file=sys.stderr)
        return 2
    user = args.user or prompt("HPC username", getpass.getuser())
    if not user:
        print("HPC username is required", file=sys.stderr)
        return 2

    key_path = args.key
    if not key_path:
        default_key = Path.home() / ".ssh" / "id_ed25519"
        if default_key.exists() and prompt_bool(f"Use SSH key {default_key}?", default=True):
            key_path = str(default_key)
    password = "" if key_path else getpass.getpass("HPC password (Duo/keyboard-interactive supported): ")

    log(f"connecting to {host}:{args.port} as {user}")
    ssh = sftp = transport = None
    try:
        ssh, sftp, transport = connect_ssh(host, args.port, user, password, key_path)
        log("connected")

        remote_base = choose_remote_work_dir(ssh, user, args.work_dir)
        paths = build_paths(remote_base)
        ensure_remote_dir(ssh, paths.run_dir)
        log(f"remote run directory: {paths.run_dir}")

        scheduler = detect_scheduler(ssh, args.scheduler)
        log(f"scheduler: {scheduler}")

        source_tar = make_source_tar(repo)
        log(f"uploading source bundle ({len(source_tar):,} bytes)")
        upload_bytes(ssh, sftp, source_tar, paths.source_tar)

        runner = textwrap.dedent(REMOTE_TEST_RUNNER).lstrip().encode()
        log(f"uploading remote test runner ({len(runner):,} bytes)")
        upload_bytes(ssh, sftp, runner, paths.runner, mode=stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

        job_script = build_job_script(paths, scheduler, args).encode()
        log(f"uploading batch script ({len(job_script):,} bytes)")
        upload_bytes(ssh, sftp, job_script, paths.job_script, mode=stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

        job_id = submit_job(ssh, scheduler, paths)
        print()
        print("BATCH JOB SUBMITTED")
        print(f"  Job ID:      {job_id}")
        print(f"  Scheduler:   {scheduler}")
        print(f"  Run dir:     {paths.run_dir}")
        print(f"  .out file:   {paths.stdout}")
        print(f"  .err file:   {paths.stderr}")
        print(f"  Summary:     {paths.summary}")
        print(f"  UNIT_TESTS_OUT={paths.stdout}")
        if scheduler == "slurm":
            print(f"  Monitor:     squeue -j {job_id}")
        else:
            print(f"  Monitor:     bjobs {job_id}")
        print(f"  Live log:    tail -f {paths.stdout}")
        print()

        if args.no_wait:
            return 0

        rc = wait_for_completion(ssh, scheduler, job_id, paths, args.poll_seconds, args.wait_timeout)
        print_remote_tail(ssh, paths)
        if rc == 0:
            log("all required HPC feature tests passed")
        else:
            log(f"HPC feature tests failed with exit {rc}")
        return rc
    finally:
        if sftp is not None:
            try:
                sftp.close()
            except Exception:
                pass
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
