#!/usr/bin/env python3
"""submit_grna_offtarget.py — submit run_grna_offtarget.py to HPC (LSF/Slurm/local)."""
import te_scheduler as sched


def main():
    p = sched.base_parser("Submit allele-aware gRNA off-target analysis")
    p.add_argument("--cas", default="SpCas9")
    p.add_argument("--grna-len", type=int, default=20)
    p.add_argument("--max-mm", type=int, default=2)
    p.add_argument("--background", default="")
    p.add_argument("--top-n", type=int, default=25)
    args = p.parse_args()

    bg = f"--background {args.background}" if args.background else ""
    analysis_args = (
        "--input {csv} --reports-dir {reports} --family {family} "
        f"--cas {args.cas} --grna-len {args.grna_len} --max-mm {args.max_mm} "
        f"--top-n {args.top_n} {bg}"
    ).strip()
    sched.submit(
        args,
        run_script="run_grna_offtarget.py",
        upload_files=["run_grna_offtarget.py", "te_scheduler.py", "requirements.txt"],
        analysis_args=analysis_args,
        job_prefix="gameca_grnaot",
    )


if __name__ == "__main__":
    main()
