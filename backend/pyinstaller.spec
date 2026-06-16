# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-file spec for the pytool IPC bridge."""

import os

# Auto list: extend if you add submodules under backend/yourtool/.
# Kept explicit so a third-party `yourtool` on sys.path cannot replace this package in the bundle.
_hidden_auto = [
    "yourtool",
    "yourtool.cli",
    "yourtool.__main__",
]

hiddenimports: list[str] = [
    # "some.optional.module",
]

hiddenimports = sorted(set(hiddenimports) | set(_hidden_auto))

_repo_root = os.path.abspath(os.path.join(SPECPATH, os.pardir))
_pipeline_datas = [
    (os.path.join(_repo_root, name), ".")
    for name in [
        "hpc_client.py",
        "query.py",
        "requirements.txt",
        "te_prep.py",
        "te_genome.py",
        "te_clustering.py",
        "te_primers.py",
        "te_alignment.py",
        "te_motif.py",
        "te_go.py",
        "te_expression.py",
        "te_enrichment.py",
        "te_notify.py",
        "te_fast.pyx",
        "setup_cython.py",
        "ui.py",
        "fetch_jaspar.py",
        "run_fold_prediction.py",
        # ── standout analysis modules ──
        "run_phylo_analysis.py",
        "run_grna_analysis.py",
        "run_grna_offtarget.py",
        "run_transduction.py",
        "run_antisense_promoter.py",
        "run_ctcf_tad.py",
        "run_epigenetic_overlay.py",
        "run_ortholog_insertion.py",
        "run_multiassembly_liftover.py",
        "run_motif_gain.py",
        "te_motif_gain.py",
        "run_stage11_all.py",
        "run_divergence.py",
        "run_ltr_struct.py",
        "run_subfamily.py",
        "run_benchmark.py",
        "run_line_sine_ltr_analysis.py",
        # ── shared libs + infra ──
        "te_moods_scan.py",
        "te_divergence.py",
        "te_ltr_struct.py",
        "te_subfamily.py",
        "te_overlay.py",
        "te_scheduler.py",
        "te_provenance.py",
        "gameca_pipeline.py",
        "gameca.yaml",
    ]
    if os.path.exists(os.path.join(_repo_root, name))
]

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=[SPECPATH, _repo_root],
    binaries=[],
    datas=_pipeline_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_tmpdir=None,
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="pytool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,
)
