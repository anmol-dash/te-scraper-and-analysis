#!/usr/bin/env python3
"""submit_ortholog_insertion.py — submit run_ortholog_insertion.py to HPC."""
import te_scheduler as sched


def main():
    p = sched.base_parser("Submit orthologous-insertion calling")
    p.add_argument("--chains", nargs="*", default=[], help="SPECIES=chain ...")
    p.add_argument("--liftover-cmd", default="liftOver")
    args = p.parse_args()
    chains = (" --chains " + " ".join(args.chains)) if args.chains else ""
    analysis_args = ("--input {csv} --reports-dir {reports} --family {family} "
                     f"--liftover-cmd {args.liftover_cmd}" + chains)
    sched.submit(args, run_script="run_ortholog_insertion.py",
                 upload_files=["run_ortholog_insertion.py", "te_overlay.py",
                               "te_scheduler.py", "requirements.txt"],
                 analysis_args=analysis_args, job_prefix="gameca_ortho")


if __name__ == "__main__":
    main()
