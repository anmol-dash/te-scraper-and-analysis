#!/usr/bin/env python3
"""submit_ctcf_tad.py — submit run_ctcf_tad.py to HPC (LSF/Slurm/local)."""
import te_scheduler as sched


def main():
    p = sched.base_parser("Submit CTCF / TAD-boundary analysis")
    p.add_argument("--ctcf", default="")
    p.add_argument("--tads", default="")
    p.add_argument("--motif-mismatch", type=int, default=3)
    p.add_argument("--boundary-window", type=int, default=50000)
    args = p.parse_args()
    extra = (f"--ctcf {args.ctcf} " if args.ctcf else "") + \
            (f"--tads {args.tads} " if args.tads else "")
    analysis_args = (
        "--input {csv} --reports-dir {reports} --family {family} "
        f"--motif-mismatch {args.motif_mismatch} "
        f"--boundary-window {args.boundary_window} {extra}"
    ).strip()
    sched.submit(args, run_script="run_ctcf_tad.py",
                 upload_files=["run_ctcf_tad.py", "te_overlay.py", "te_scheduler.py",
                               "requirements.txt"],
                 analysis_args=analysis_args, job_prefix="gameca_ctcf")


if __name__ == "__main__":
    main()
