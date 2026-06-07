#!/usr/bin/env python3
"""submit_multiassembly_liftover.py — submit run_multiassembly_liftover.py to HPC."""
import te_scheduler as sched


def main():
    p = sched.base_parser("Submit multi-assembly liftover")
    p.add_argument("--source-assembly", default="hg38")
    p.add_argument("--chains", nargs="*", default=[], help="ASSEMBLY=chain ...")
    p.add_argument("--liftover-cmd", default="liftOver")
    args = p.parse_args()
    chains = (" --chains " + " ".join(args.chains)) if args.chains else ""
    analysis_args = ("--input {csv} --reports-dir {reports} --family {family} "
                     f"--source-assembly {args.source_assembly} "
                     f"--liftover-cmd {args.liftover_cmd}" + chains)
    sched.submit(args, run_script="run_multiassembly_liftover.py",
                 upload_files=["run_multiassembly_liftover.py", "te_overlay.py",
                               "te_scheduler.py", "requirements.txt"],
                 analysis_args=analysis_args, job_prefix="gameca_masm")


if __name__ == "__main__":
    main()
