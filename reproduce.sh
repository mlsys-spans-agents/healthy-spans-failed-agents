#!/usr/bin/env bash
# Rebuild every number in results/ from the raw datasets (no GPU, no API key).
set -euo pipefail
cd "$(dirname "$0")"
[ -f data/trail/gaia.parquet ] || ./fetch_data.sh
pip install -q -r requirements.txt
python -m hs.run_all
echo "wrote results/results.json and results/report.txt"
