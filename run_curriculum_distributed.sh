#!/bin/bash

# Distributed Curriculum Learning Launch Script
# Usage: ./run_curriculum_distributed.sh [model] [num_gpus] [batch_size] [stages]

MODEL=${1:-"EmbedHealthFlamingo"}
NUM_GPUS=${2:-4}
BATCH_SIZE=${3:-8}
STAGES=${4:-"stage1_mcq stage2_captioning"}

echo "🚀 Launching distributed curriculum learning"
echo "Model: $MODEL"
echo "GPUs: $NUM_GPUS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Stages: $STAGES"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ CUDA not available. Please check your GPU setup."
    exit 1
fi

# Check if the model type is valid
if [[ "$MODEL" != "EmbedHealthFlamingo" && "$MODEL" != "EmbedHealthSP" ]]; then
    echo "❌ Invalid model type: $MODEL. Must be 'EmbedHealthFlamingo' or 'EmbedHealthSP'"
    exit 1
fi

# DDP training for both models (simpler and more reliable than FSDP)
echo "🔧 Starting DDP training..."
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS curriculum_learning.py \
    --model $MODEL \
    --stages $STAGES \
    --batch_size $BATCH_SIZE \
    --gradient_checkpointing

echo "✅ DDP training completed!"
echo "🎉 All training completed!" 