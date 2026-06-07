#!/usr/bin/env python3
"""submit_epigenetic_overlay.py — submit run_epigenetic_overlay.py to HPC."""
import te_scheduler as sched


def main():
    p = sched.base_parser("Submit epigenetic / regulatory overlay")
    p.add_argument("--tracks", nargs="*", default=[], help="NAME=path.bed ...")
    args = p.parse_args()
    tracks = (" --tracks " + " ".join(args.tracks)) if args.tracks else ""
    analysis_args = ("--input {csv} --reports-dir {reports} --family {family}" + tracks)
    sched.submit(args, run_script="run_epigenetic_overlay.py",
                 upload_files=["run_epigenetic_overlay.py", "te_overlay.py",
                               "te_scheduler.py", "requirements.txt"],
                 analysis_args=analysis_args, job_prefix="gameca_epi")


if __name__ == "__main__":
    main()
