#!/bin/bash

# List of models to evaluate
models=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
    "google/gemma-2b"
    "google/gemma-2-2b"
    "google/gemma-3-4b-pt"
    "google/gemma-3-4b-it"
)

echo "Starting evaluation of ${#models[@]} models on TSQA dataset..."
for model in "${models[@]}"; do
    echo "Evaluating $model on TSQA..."
    python evaluate_tsqa.py "$model"
done

echo "Starting evaluation of ${#models[@]} models on PAMAP dataset..."
for model in "${models[@]}"; do
    echo "Evaluating $model on PAMAP..."
    python evaluate_pamap.py "$model"
done

echo "All evaluations completed!"