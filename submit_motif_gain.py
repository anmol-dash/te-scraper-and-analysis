#!/usr/bin/env python3
"""submit_motif_gain.py — submit run_motif_gain.py to HPC (LSF/Slurm/local)."""
import te_scheduler as sched


def main():
    p = sched.base_parser("Submit GAMECA TF motif gain/loss analysis")
    p.add_argument("--assembly", default="hg38")
    p.add_argument("--max-mm",   type=int, default=2)
    args = p.parse_args()

    analysis_args = (
        "--input {csv} --reports-dir {reports} --family {family} "
        f"--assembly {args.assembly} --max-mm {args.max_mm}"
    )
    sched.submit(
        args,
        run_script="run_motif_gain.py",
        upload_files=["run_motif_gain.py", "te_motif_gain.py",
                      "te_scheduler.py", "requirements.txt"],
        analysis_args=analysis_args,
        job_prefix="gameca_motif_gain",
    )


if __name__ == "__main__":
    main()
