#!/bin/bash

# Distributed Curriculum Learning Launch Script
# Usage: ./run_curriculum_distributed.sh [model] [num_gpus] [batch_size] [stages] [precision] [sharding_strategy]

MODEL=${1:-"EmbedHealthFlamingo"}
NUM_GPUS=${2:-4}
BATCH_SIZE=${3:-8}
STAGES=${4:-"stage1_mcq stage2_captioning"}
PRECISION=${5:-"bf16"}
SHARDING_STRATEGY=${6:-"full"}

echo "🚀 Launching distributed curriculum learning"
echo "Model: $MODEL"
echo "GPUs: $NUM_GPUS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Stages: $STAGES"
echo "Precision: $PRECISION"
echo "Sharding strategy: $SHARDING_STRATEGY"

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

# Check if FSDP is supported for the model
if [[ "$MODEL" == "EmbedHealthSP" ]]; then
    echo "⚠️  FSDP is not supported for EmbedHealthSP. Switching to DDP training..."
    # DDP training for EmbedHealthSP
    echo "🔧 Starting DDP training..."
    torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS curriculum_learning.py \
        --model $MODEL \
        --stages $STAGES \
        --batch_size $BATCH_SIZE
    echo "✅ DDP training completed!"
else
    # FSDP training for EmbedHealthFlamingo
    echo "🔧 Starting FSDP training..."
    torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS curriculum_learning.py \
        --model $MODEL \
        --stages $STAGES \
        --batch_size $BATCH_SIZE \
        --fsdp \
        --fsdp_use_orig_params \
        --fsdp_sharding_strategy $SHARDING_STRATEGY \
        --precision $PRECISION \
        --gradient_checkpointing
    echo "✅ FSDP training completed!"
fi

echo "🎉 All training completed!" 