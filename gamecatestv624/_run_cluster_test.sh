#!/usr/bin/env bash
set -uo pipefail
cd /home/amodz/anmol/te-scraper-and-analysis
echo "Host: $(hostname)   Started: $(date)"
python3 /home/amodz/anmol/te-scraper-and-analysis/run_gameca_cluster_test.py --out-dir /home/amodz/anmol/gamecatestv624 --max-loci 300 --notify-email anmoldash@gmail.com 
echo "Finished: $(date)"
