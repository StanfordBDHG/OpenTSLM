#!/bin/bash

# List of models to evaluate
models=(
    "openai-gpt-4o"
    "meta-llama/Llama-3.2-3B"
    "google/gemma-2b"
    "google/gemma-2-2b"
    "google/gemma-3-4b-pt"
    "google/gemma-3-4b-it"
)

#echo "Starting evaluation of ${#models[@]} models on TSQA dataset..."
#for model in "${models[@]}"; do
#    echo "Evaluating $model on TSQA..."
#    python evaluate_tsqa.py "$model"
#done


#echo "Starting evaluation of ${#models[@]} models on PAMAP dataset..."
#for model in "${models[@]}"; do
#    echo "Evaluating $model on PAMAP..."
#    python evaluate_pamap.py "$model"
#done

echo "Starting evaluation of ${#models[@]} models on SleepEDF CoT dataset..."
for model in "${models[@]}"; do
    echo "Evaluating $model on SleepEDF CoT..."
    python evaluate_sleep_cot.py "$model"
done

echo "All evaluations completed!"