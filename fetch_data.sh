#!/usr/bin/env bash
# Download the three public datasets into data/.
# TRAIL is gated: accept its terms at https://huggingface.co/datasets/PatronusAI/TRAIL
# and export HF_TOKEN before running.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/taubench data/whowhen data/trail

for f in gpt-4o-airline gpt-4o-retail sonnet-35-new-airline sonnet-35-new-retail; do
  curl -sfL -o "data/taubench/$f.json" \
    "https://raw.githubusercontent.com/sierra-research/tau-bench/main/historical_trajectories/$f.json"
done

for c in Algorithm-Generated Hand-Crafted; do
  curl -sfL -o "data/whowhen/$c.parquet" \
    "https://huggingface.co/datasets/Kevin355/Who_and_When/resolve/main/$c.parquet"
done

: "${HF_TOKEN:?set HF_TOKEN to download the gated TRAIL dataset}"
curl -sfL -H "Authorization: Bearer $HF_TOKEN" -o data/trail/gaia.parquet \
  "https://huggingface.co/datasets/PatronusAI/TRAIL/resolve/main/data/gaia-00000-of-00001-33a2e72d362d688a.parquet"
curl -sfL -H "Authorization: Bearer $HF_TOKEN" -o data/trail/swe_bench.parquet \
  "https://huggingface.co/datasets/PatronusAI/TRAIL/resolve/main/data/swe_bench-00000-of-00001-91aa04220f7198b4.parquet"
echo "data ready"
