#!/usr/bin/env python3
"""submit_transduction.py — submit run_transduction.py to HPC (LSF/Slurm/local)."""
import te_scheduler as sched


def main():
    p = sched.base_parser("Submit 3' transduction detection")
    p.add_argument("--tail-bp", type=int, default=150)
    p.add_argument("--kmer", type=int, default=12)
    p.add_argument("--min-shared", type=int, default=3)
    p.add_argument("--flank-col", default="")
    args = p.parse_args()

    flank = f"--flank-col {args.flank_col}" if args.flank_col else ""
    analysis_args = (
        "--input {csv} --reports-dir {reports} --family {family} "
        f"--tail-bp {args.tail_bp} --kmer {args.kmer} --min-shared {args.min_shared} {flank}"
    ).strip()
    sched.submit(args, run_script="run_transduction.py",
                 upload_files=["run_transduction.py", "te_scheduler.py", "requirements.txt"],
                 analysis_args=analysis_args, job_prefix="gameca_trans")


if __name__ == "__main__":
    main()
