#!/usr/bin/env python3
"""submit_divergence.py — submit run_divergence.py to HPC (LSF/Slurm/local)."""
import te_scheduler as sched


def main():
    p = sched.base_parser("Submit GAMECA divergence / repeat landscape analysis")
    p.add_argument("--assembly",    default="hg38")
    p.add_argument("--cpg-omega",   type=float, default=10.0)
    p.add_argument("--bin-width",   type=float, default=1.0)
    p.add_argument("--max-div",     type=float, default=40.0)
    args = p.parse_args()

    analysis_args = (
        "--input {csv} --reports-dir {reports} --family {family} "
        f"--assembly {args.assembly} --cpg-omega {args.cpg_omega} "
        f"--bin-width {args.bin_width} --max-div {args.max_div}"
    )
    sched.submit(
        args,
        run_script="run_divergence.py",
        upload_files=["run_divergence.py", "te_divergence.py",
                      "te_scheduler.py", "requirements.txt"],
        analysis_args=analysis_args,
        job_prefix="gameca_divergence",
    )


if __name__ == "__main__":
    main()
