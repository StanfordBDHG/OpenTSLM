#!/bin/bash

# Distributed Curriculum Learning Launch Script
# Usage: ./run_curriculum_distributed.sh [model] [num_gpus] [batch_size] [stages] [llm_id]

MODEL=${1:-"EmbedHealthFlamingo"}
NUM_GPUS=${2:-4}
BATCH_SIZE=${3:-8}
STAGES=${4:-"stage1_mcq stage2_captioning"}
LLM_ID=${5:-"meta-llama/Llama-3.2-1B"}

echo "🚀 Launching distributed curriculum learning"
echo "Model: $MODEL"
echo "GPUs: $NUM_GPUS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Stages: $STAGES"
echo "LLM ID: $LLM_ID"
echo ""
echo "📊 Training Configuration:"
echo "   Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "   Learning rates and epochs are defined per stage in the code"
echo "   Stage 1 (MCQ): 30 epochs, LR varies by model"
echo "   Stage 2 (Captioning): 60 epochs, LR varies by model"
echo ""

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

# Validate LLM_ID if provided
if [[ -n "$LLM_ID" ]]; then
    echo "🔍 Validating LLM ID: $LLM_ID"
    
    # Check if it's a MedGemma model
    if [[ "$LLM_ID" == *"medgemma"* ]]; then
        echo "🏥 Detected MedGemma model - medical domain expertise enabled!"
        
        # Provide hardware recommendations for MedGemma
        if [[ "$LLM_ID" == *"medgemma-27b"* ]]; then
            echo "⚠️  MedGemma-27b detected - ensure you have sufficient GPU memory (48GB+ per GPU)"
        elif [[ "$LLM_ID" == *"medgemma-7b"* ]]; then
            echo "💡 MedGemma-7b detected - recommended for most use cases"
        elif [[ "$LLM_ID" == *"medgemma-2b"* ]]; then
            echo "⚡ MedGemma-2b detected - lightweight and fast"
        fi
    elif [[ "$LLM_ID" == *"llama"* ]]; then
        echo "🦙 Using Llama model - general purpose language model"
    fi
    
    # Check if it's a supported model type
    if [[ "$LLM_ID" != *"llama"* && "$LLM_ID" != *"medgemma"* && "$LLM_ID" != *"gemma"* ]]; then
        echo "⚠️  Warning: LLM ID '$LLM_ID' may not be fully tested. Supported models:"
        echo "   - meta-llama/Llama-3.2-1B (default)"
        echo "   - google/medgemma-2b"
        echo "   - google/medgemma-7b"
        echo "   - google/medgemma-27b"
        echo "   - Other Gemma-based models"
    fi
fi

# DDP training for both models (simpler and more reliable than FSDP)
echo "🔧 Starting DDP training..."

# Build the command
CMD="torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS curriculum_learning.py \
    --model $MODEL \
    --stages $STAGES \
    --batch_size $BATCH_SIZE \
    --gradient_checkpointing \
    --llm_id $LLM_ID"

echo "Executing: $CMD"
eval $CMD

echo "✅ DDP training completed!"
echo "🎉 All training completed!"
echo ""
echo "�� Tips for tuning:"
echo "   - Learning rates and epochs are defined per stage in curriculum_learning.py"
echo "   - For larger effective batch sizes, you may need to increase learning rates"
echo "   - MedGemma models are optimized for healthcare applications"
echo "   - Use gradient_checkpointing for memory efficiency with larger models"
echo ""
echo "📁 Results saved to: results/$MODEL/" 