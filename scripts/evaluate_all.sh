#!/usr/bin/env bash
# Score every file in results/raw and print the summary table (Tables 6 and 9).
set -euo pipefail
cd "$(dirname "$0")/.."

python evaluation/prepare_for_eval.py results/raw results/ready
python evaluation/evaluate.py results/ready -o results/scores
python evaluation/aggregate.py results/scores/aggregate_metrics.csv -o results/summary.csv
