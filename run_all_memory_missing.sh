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
    "EmbedHealthSP"
    "EmbedHealthFlamingo"
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
  "google/gemma-3-1b-pt"
)

# SimulationQADataset parameters
SIMULATION_LENGTHS=(10000)
SIMULATION_NUM_SERIES=(1 2 3 4 5)

# Extended parameters for google/gemma-3-1b-pt
GEMMA_1B="google/gemma-3-1b-pt"
SIMULATION_LENGTHS_EXTENDED=(1 10 100 1000 10000)
SIMULATION_NUM_SERIES_EXTENDED=(1 2 3 4 5)

RESULTS_FLAG="--results_csv ${RESULTS_CSV:-$REPO_DIR/memory_use.csv}"

if [[ -n "$DEVICE_ARG" ]]; then
  DEVICE_FLAG="--device ${DEVICE_ARG#--device }"
else
  DEVICE_FLAG=""
fi

echo "Writing results to: ${RESULTS_FLAG#--results_csv }"

# Run specific datasets for specific model/LLM combinations
echo "Running specific datasets for specific model/LLM combinations..."

# llama3b EmbedHealthFlamingo for TSQA, HAR, Sleep, ECG_QA
LLAMA3B="meta-llama/Llama-3.2-3B"
FLAMINGO_MODEL="EmbedHealthFlamingo"
SP_MODEL="EmbedHealthSP"

SPECIFIC_DATASETS=(
  "TSQADataset"
  "HARCoTQADataset" 
  "SleepEDFCoTQADataset"
  "ECGQACoTQADataset"
)


# Experiment 1: All 4 LLMs with length=10000 and num_series=1,2,3,4,5
echo "=========================================="
echo "Experiment 1: All LLMs with length=10000"
echo "=========================================="

for llm in "${LLM_IDS[@]}"; do
  for model in "${MODELS[@]}"; do
    for length in "${SIMULATION_LENGTHS[@]}"; do
      for num_series in "${SIMULATION_NUM_SERIES[@]}"; do
        echo "[RUN] llm_id=$llm model=$model dataset=SimulationQADataset length=$length num_series=$num_series"
        set +e
        $PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "$llm" --model "$model" --dataset "SimulationQADataset" --length "$length" --num_series "$num_series" $DEVICE_FLAG $RESULTS_FLAG
        status=$?
        set -e
        if [[ $status -ne 0 ]]; then
          echo "[ERROR] Failed for llm_id=$llm model=$model dataset=SimulationQADataset length=$length num_series=$num_series (exit $status)"
        fi
      done
    done
  done
done

# Experiment 2: Only google/gemma-3-1b-pt with lengths=1,10,100,1000,10000 and num_series=1,2,3,4,5
# Note: length=10000 is already covered above, so we skip it here to avoid duplication
echo "=========================================="
echo "Experiment 2: google/gemma-3-1b-pt with extended lengths (excluding 10000)"
echo "=========================================="

SIMULATION_LENGTHS_GEMMA_ONLY=(1 10 100 1000)  # Exclude 10000 to avoid duplication

for model in "${MODELS[@]}"; do
  for length in "${SIMULATION_LENGTHS_GEMMA_ONLY[@]}"; do
    for num_series in "${SIMULATION_NUM_SERIES_EXTENDED[@]}"; do
      echo "[RUN] llm_id=$GEMMA_1B model=$model dataset=SimulationQADataset length=$length num_series=$num_series"
      set +e
      $PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "$GEMMA_1B" --model "$model" --dataset "SimulationQADataset" --length "$length" --num_series "$num_series" $DEVICE_FLAG $RESULTS_FLAG
      status=$?
      set -e
      if [[ $status -ne 0 ]]; then
        echo "[ERROR] Failed for llm_id=$GEMMA_1B model=$model dataset=SimulationQADataset length=$length num_series=$num_series (exit $status)"
      fi
    done
  done
done


echo "=========================================="
echo "All simulation experiments completed!"
echo "=========================================="

# Summary of runs
echo "Summary:"
echo "- Experiment 1: 4 LLMs × 2 models × 1 length × 5 num_series = 40 runs"
echo "- Experiment 2: 1 LLM × 2 models × 4 lengths × 5 num_series = 40 runs"
echo "- Total: 80 simulation runs"

