#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_all_datasets_missing.sh [--device cuda:0] [--results_csv memory_use.csv]

DEVICE_ARG=${1:-}
RESULTS_CSV=${2:-}

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=python

RESULTS_FLAG="--results_csv ${RESULTS_CSV:-$REPO_DIR/memory_use.csv}"

if [[ -n "$DEVICE_ARG" ]]; then
  DEVICE_FLAG="--device ${DEVICE_ARG#--device }"
else
  DEVICE_FLAG=""
fi

echo "Writing results to: ${RESULTS_FLAG#--results_csv }"

# Failed configurations from memory_use.csv that need to be rerun
echo "Rerunning failed dataset configurations..."

# meta-llama/Llama-3.2-1B + EmbedHealthSP + ECG-QA-CoT (CUDA OOM)
echo "[RETRY] meta-llama/Llama-3.2-1B + EmbedHealthSP + ECG-QA-CoT"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "meta-llama/Llama-3.2-1B" --model "EmbedHealthSP" --dataset "ECGQACoTQADataset" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for meta-llama/Llama-3.2-1B + EmbedHealthSP + ECG-QA-CoT (exit $status)"
fi

# meta-llama/Llama-3.2-3B + EmbedHealthSP + ECG-QA-CoT (CUDA OOM)
echo "[RETRY] meta-llama/Llama-3.2-3B + EmbedHealthSP + ECG-QA-CoT"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "meta-llama/Llama-3.2-3B" --model "EmbedHealthSP" --dataset "ECGQACoTQADataset" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for meta-llama/Llama-3.2-3B + EmbedHealthSP + ECG-QA-CoT (exit $status)"
fi

# meta-llama/Llama-3.2-3B + EmbedHealthFlamingo + TSQA (CUDA OOM)
echo "[RETRY] meta-llama/Llama-3.2-3B + EmbedHealthFlamingo + TSQA"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "meta-llama/Llama-3.2-3B" --model "EmbedHealthFlamingo" --dataset "TSQADataset" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for meta-llama/Llama-3.2-3B + EmbedHealthFlamingo + TSQA (exit $status)"
fi

# meta-llama/Llama-3.2-3B + EmbedHealthFlamingo + HAR-CoT (CUDA OOM)
echo "[RETRY] meta-llama/Llama-3.2-3B + EmbedHealthFlamingo + HAR-CoT"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "meta-llama/Llama-3.2-3B" --model "EmbedHealthFlamingo" --dataset "HARCoTQADataset" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for meta-llama/Llama-3.2-3B + EmbedHealthFlamingo + HAR-CoT (exit $status)"
fi

# meta-llama/Llama-3.2-3B + EmbedHealthFlamingo + SleepEDF-CoT (CUDA OOM)
echo "[RETRY] meta-llama/Llama-3.2-3B + EmbedHealthFlamingo + SleepEDF-CoT"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "meta-llama/Llama-3.2-3B" --model "EmbedHealthFlamingo" --dataset "SleepEDFCoTQADataset" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for meta-llama/Llama-3.2-3B + EmbedHealthFlamingo + SleepEDF-CoT (exit $status)"
fi

# meta-llama/Llama-3.2-3B + EmbedHealthFlamingo + ECG-QA-CoT (CUDA OOM)
echo "[RETRY] meta-llama/Llama-3.2-3B + EmbedHealthFlamingo + ECG-QA-CoT"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "meta-llama/Llama-3.2-3B" --model "EmbedHealthFlamingo" --dataset "ECGQACoTQADataset" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for meta-llama/Llama-3.2-3B + EmbedHealthFlamingo + ECG-QA-CoT (exit $status)"
fi

echo "All failed dataset configurations have been retried."
