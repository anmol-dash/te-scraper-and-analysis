#!/usr/bin/env python3
"""
submit_phylo_analysis.py — submit run_phylo_analysis.py to HPC (LSF/Slurm/local).

Usage:
    python submit_phylo_analysis.py --host HOST --user USER \\
        --input loci.csv --reports-dir /path/reports --family LTR5_Hs \\
        [--subst-rate 2.2e-9] [--clock-divisor 1] [--intact-orf-aa 100] \\
        [--scheduler auto] [--mem-mb 16000] [--cpus 4]

Output (stdout):  OUT <path>   /   ERR <path>
"""
import te_scheduler as sched


def main():
    p = sched.base_parser("Submit TE phylogenetics / divergence-age analysis")
    p.add_argument("--subst-rate", type=float, default=2.2e-9)
    p.add_argument("--clock-divisor", type=float, default=1.0)
    p.add_argument("--intact-orf-aa", type=int, default=100)
    p.add_argument("--max-tree-tips", type=int, default=60)
    args = p.parse_args()

    analysis_args = (
        "--input {csv} --reports-dir {reports} --family {family} "
        f"--subst-rate {args.subst_rate} --clock-divisor {args.clock_divisor} "
        f"--intact-orf-aa {args.intact_orf_aa} --max-tree-tips {args.max_tree_tips}"
    )
    sched.submit(
        args,
        run_script="run_phylo_analysis.py",
        upload_files=["run_phylo_analysis.py", "te_scheduler.py", "requirements.txt"],
        analysis_args=analysis_args,
        job_prefix="gameca_phylo",
    )


if __name__ == "__main__":
    main()
