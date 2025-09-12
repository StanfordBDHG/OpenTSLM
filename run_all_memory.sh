#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_all_memory.sh [--device cuda:0] [--results_csv memory_use.csv]

DEVICE_ARG=${1:-}
RESULTS_CSV=${2:-}

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=python

# Models and datasets to test
MODELS=(
  "EmbedHealthFlamingo"
  "EmbedHealthSP"
)

DATASETS=(
  "TSQADataset"
  "HARCoTQADataset"
  "SleepEDFCoTQADataset"
  "ECGQACoTQADataset"
)

# LLM IDs
LLM_IDS=(
  "meta-llama/Llama-3.2-1B"
  "meta-llama/Llama-3.2-3B"
  "google/gemma-3-270m"
  "google/gemma-3-1b"
)

RESULTS_FLAG="--results_csv ${RESULTS_CSV:-$REPO_DIR/memory_use.csv}"

if [[ -n "$DEVICE_ARG" ]]; then
  DEVICE_FLAG="--device ${DEVICE_ARG#--device }"
else
  DEVICE_FLAG=""
fi

echo "Writing results to: ${RESULTS_FLAG#--results_csv }"

for llm in "${LLM_IDS[@]}"; do
  for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      echo "[RUN] llm_id=$llm model=$model dataset=$dataset"
      set +e
      $PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "$llm" --model "$model" --dataset "$dataset" $DEVICE_FLAG $RESULTS_FLAG
      status=$?
      set -e
      if [[ $status -ne 0 ]]; then
        echo "[ERROR] Failed for llm_id=$llm model=$model dataset=$dataset (exit $status)"
      fi
    done
  done
done

echo "All runs completed."


