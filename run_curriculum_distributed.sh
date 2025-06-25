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

# FSDP training
echo "🔧 Starting FSDP training..."
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS curriculum_learning.py \
    --model $MODEL \
    --stages $STAGES \
    --batch_size $BATCH_SIZE \
    --fsdp \
    --fsdp_use_orig_params \
    --precision bf16 \
    --gradient_checkpointing

echo "✅ FSDP training completed!"

# Alternative: DDP training (uncomment to use)
# echo "🔧 Starting DDP training..."
# torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS curriculum_learning.py \
#     --model $MODEL \
#     --stages $STAGES \
#     --batch_size $BATCH_SIZE
# echo "✅ DDP training completed!" 