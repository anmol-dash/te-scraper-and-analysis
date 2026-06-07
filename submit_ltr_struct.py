#!/usr/bin/env python3
"""submit_ltr_struct.py — submit run_ltr_struct.py to HPC (LSF/Slurm/local)."""
import te_scheduler as sched


def main():
    p = sched.base_parser("Submit GAMECA LTR structural annotation")
    p.add_argument("--min-ltr-identity", type=float, default=0.65)
    args = p.parse_args()

    analysis_args = (
        "--input {csv} --reports-dir {reports} --family {family} "
        f"--min-ltr-identity {args.min_ltr_identity}"
    )
    sched.submit(
        args,
        run_script="run_ltr_struct.py",
        upload_files=["run_ltr_struct.py", "te_ltr_struct.py",
                      "te_scheduler.py", "requirements.txt"],
        analysis_args=analysis_args,
        job_prefix="gameca_ltr_struct",
    )


if __name__ == "__main__":
    main()
