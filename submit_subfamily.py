#!/usr/bin/env python3
"""submit_subfamily.py — submit run_subfamily.py to HPC (LSF/Slurm/local)."""
import te_scheduler as sched


def main():
    p = sched.base_parser("Submit GAMECA subfamily resolution analysis")
    p.add_argument("--assembly",  default="hg38")
    p.add_argument("--cpg-omega", type=float, default=10.0)
    args = p.parse_args()

    analysis_args = (
        "--input {csv} --reports-dir {reports} --family {family} "
        f"--assembly {args.assembly} --cpg-omega {args.cpg_omega}"
    )
    sched.submit(
        args,
        run_script="run_subfamily.py",
        upload_files=["run_subfamily.py", "te_subfamily.py",
                      "te_scheduler.py", "requirements.txt"],
        analysis_args=analysis_args,
        job_prefix="gameca_subfamily",
    )


if __name__ == "__main__":
    main()
