#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_all_memory_missing.sh [--device cuda:0] [--results_csv memory_use.csv]

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

# Failed configurations that need to be rerun
echo "Rerunning failed configurations..."

# CUDA Out of Memory failures
echo "=== CUDA Out of Memory Failures ==="

# meta-llama/Llama-3.2-1B + EmbedHealthSP failures
echo "[RETRY] meta-llama/Llama-3.2-1B + EmbedHealthSP + Simulation-L10000-N3"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "meta-llama/Llama-3.2-1B" --model "EmbedHealthSP" --dataset "SimulationQADataset" --length "10000" --num_series "3" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for meta-llama/Llama-3.2-1B + EmbedHealthSP + Simulation-L10000-N3 (exit $status)"
fi

echo "[RETRY] meta-llama/Llama-3.2-1B + EmbedHealthSP + Simulation-L10000-N4"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "meta-llama/Llama-3.2-1B" --model "EmbedHealthSP" --dataset "SimulationQADataset" --length "10000" --num_series "4" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for meta-llama/Llama-3.2-1B + EmbedHealthSP + Simulation-L10000-N4 (exit $status)"
fi

echo "[RETRY] meta-llama/Llama-3.2-1B + EmbedHealthSP + Simulation-L10000-N5"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "meta-llama/Llama-3.2-1B" --model "EmbedHealthSP" --dataset "SimulationQADataset" --length "10000" --num_series "5" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for meta-llama/Llama-3.2-1B + EmbedHealthSP + Simulation-L10000-N5 (exit $status)"
fi

# meta-llama/Llama-3.2-3B + EmbedHealthSP failures
echo "[RETRY] meta-llama/Llama-3.2-3B + EmbedHealthSP + Simulation-L10000-N2"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "meta-llama/Llama-3.2-3B" --model "EmbedHealthSP" --dataset "SimulationQADataset" --length "10000" --num_series "2" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for meta-llama/Llama-3.2-3B + EmbedHealthSP + Simulation-L10000-N2 (exit $status)"
fi

echo "[RETRY] meta-llama/Llama-3.2-3B + EmbedHealthSP + Simulation-L10000-N3"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "meta-llama/Llama-3.2-3B" --model "EmbedHealthSP" --dataset "SimulationQADataset" --length "10000" --num_series "3" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for meta-llama/Llama-3.2-3B + EmbedHealthSP + Simulation-L10000-N3 (exit $status)"
fi

echo "[RETRY] meta-llama/Llama-3.2-3B + EmbedHealthSP + Simulation-L10000-N4"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "meta-llama/Llama-3.2-3B" --model "EmbedHealthSP" --dataset "SimulationQADataset" --length "10000" --num_series "4" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for meta-llama/Llama-3.2-3B + EmbedHealthSP + Simulation-L10000-N4 (exit $status)"
fi

echo "[RETRY] meta-llama/Llama-3.2-3B + EmbedHealthSP + Simulation-L10000-N5"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "meta-llama/Llama-3.2-3B" --model "EmbedHealthSP" --dataset "SimulationQADataset" --length "10000" --num_series "5" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for meta-llama/Llama-3.2-3B + EmbedHealthSP + Simulation-L10000-N5 (exit $status)"
fi

# google/gemma-3-270m + EmbedHealthSP failure
echo "[RETRY] google/gemma-3-270m + EmbedHealthSP + Simulation-L10000-N5"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "google/gemma-3-270m" --model "EmbedHealthSP" --dataset "SimulationQADataset" --length "10000" --num_series "5" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for google/gemma-3-270m + EmbedHealthSP + Simulation-L10000-N5 (exit $status)"
fi

# google/gemma-3-1b-pt + EmbedHealthSP failures
echo "[RETRY] google/gemma-3-1b-pt + EmbedHealthSP + Simulation-L10000-N4"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "google/gemma-3-1b-pt" --model "EmbedHealthSP" --dataset "SimulationQADataset" --length "10000" --num_series "4" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for google/gemma-3-1b-pt + EmbedHealthSP + Simulation-L10000-N4 (exit $status)"
fi

echo "[RETRY] google/gemma-3-1b-pt + EmbedHealthSP + Simulation-L10000-N5"
set +e
$PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "google/gemma-3-1b-pt" --model "EmbedHealthSP" --dataset "SimulationQADataset" --length "10000" --num_series "5" $DEVICE_FLAG $RESULTS_FLAG
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[ERROR] Failed for google/gemma-3-1b-pt + EmbedHealthSP + Simulation-L10000-N5 (exit $status)"
fi


echo "All failed configurations have been retried."

