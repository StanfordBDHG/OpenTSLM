#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_all_memory_missing.sh [--device cuda:0] [--results_csv memory_use.csv]

DEVICE_ARG=${1:-}
RESULTS_CSV=${2:-}

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=python

# Models and datasets to test
MODELS=(
    "OpenTSLMSP"
    "OpenTSLMFlamingo"
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

# SimulationQADataset parameters
SIMULATION_LENGTHS=(10 100 1000 10000)
SIMULATION_NUM_SERIES=(1 2 3 4 5)

RESULTS_FLAG="--results_csv ${RESULTS_CSV:-$REPO_DIR/memory_use.csv}"

if [[ -n "$DEVICE_ARG" ]]; then
  DEVICE_FLAG="--device ${DEVICE_ARG#--device }"
else
  DEVICE_FLAG=""
fi

echo "Writing results to: ${RESULTS_FLAG#--results_csv }"

# Failed configurations that need to be rerun
echo "Rerunning failed configurations..."

# llama3b OpenTSLMFlamingo for TSQA, HAR, Sleep, ECG_QA
LLAMA3B="meta-llama/Llama-3.2-3B"
FLAMINGO_MODEL="OpenTSLMFlamingo"
SP_MODEL="OpenTSLMSP"

SPECIFIC_DATASETS=(
  "TSQADataset"
  "HARCoTQADataset" 
  "SleepEDFCoTQADataset"
  "ECGQACoTQADataset"
)

for dataset in "${SPECIFIC_DATASETS[@]}"; do
  echo "[RUN] llm_id=$LLAMA3B model=$FLAMINGO_MODEL dataset=$dataset"
  set +e
  $PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "$LLAMA3B" --model "$FLAMINGO_MODEL" --dataset "$dataset" $DEVICE_FLAG $RESULTS_FLAG
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    echo "[ERROR] Failed for llm_id=$LLAMA3B model=$FLAMINGO_MODEL dataset=$dataset (exit $status)"
  fi
done

# OpenTSLMSP llama3b for ECG_QA
echo "[RUN] llm_id=$LLAMA3B model=$SP_MODEL dataset=ECGQACoTQADataset"
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

