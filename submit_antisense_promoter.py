#!/usr/bin/env python3
"""submit_antisense_promoter.py — submit run_antisense_promoter.py to HPC."""
import te_scheduler as sched


def main():
    p = sched.base_parser("Submit antisense / bidirectional promoter analysis")
    p.add_argument("--promoter-bp", type=int, default=200)
    args = p.parse_args()
    analysis_args = (
        "--input {csv} --reports-dir {reports} --family {family} "
        f"--promoter-bp {args.promoter_bp}"
    )
    sched.submit(args, run_script="run_antisense_promoter.py",
                 upload_files=["run_antisense_promoter.py", "te_scheduler.py", "requirements.txt"],
                 analysis_args=analysis_args, job_prefix="gameca_anti")


if __name__ == "__main__":
    main()
