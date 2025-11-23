#!/usr/bin/env python3
"""
Evaluate stage1_mcq checkpoint on TimeSeriesExam dataset.

This script loads the stage1 checkpoint (trained on TSQA MCQ) and evaluates
it on the TimeSeriesExam dataset to test if catastrophic forgetting from
stage2 captioning training is the cause of 0% accuracy.

Usage:
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python eval_stage1_on_tsexam.py \
        --batch_size 4 \
        --llm_id meta-llama/Llama-3.2-1B
"""

import argparse
import os
import sys
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from time_series_datasets.timeseriesexam.TimeSeriesExam1QADataset import TimeSeriesExam1QADataset
from time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from transformers import AutoTokenizer
from model_config import PATCH_SIZE


def calculate_accuracy(predictions, golds):
    """Calculate accuracy for MCQ predictions."""
    correct = 0
    total = len(predictions)

    for pred, gold in zip(predictions, golds):
        pred_clean = pred.strip().lower()
        gold_clean = gold.replace("<|end_of_text|>", "").strip().lower()

        # Check if prediction starts with correct answer pattern
        if gold_clean.startswith("("):
            # Gold is like "(b)"
            if pred_clean.startswith(gold_clean[:3]):  # Match "(b)"
                correct += 1
        else:
            # Gold is descriptive, check if prediction contains gold
            if gold_clean in pred_clean or pred_clean in gold_clean:
                correct += 1

    return correct / total if total > 0 else 0.0


def setup_distributed():
    """Initialize distributed training if applicable."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
    elif torch.cuda.device_count() > 1:
        # Multi-GPU without torchrun - not using DDP
        rank = 0
        world_size = 1
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def main():
    parser = argparse.ArgumentParser(description="Evaluate stage1 checkpoint on TimeSeriesExam")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU (must be 1 for TimeSeriesExam due to variable time series count)")
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B", help="LLM model ID")
    parser.add_argument("--stage1_checkpoint", type=str,
                        default="results/Llama_3_2_1B/OpenTSLMFlamingo/stage1_mcq/checkpoints/best_model.pt",
                        help="Path to stage1 checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/Llama_3_2_1B/OpenTSLMFlamingo/tsqa_on_ts_exam",
                        help="Output directory for results")
    args = parser.parse_args()

    # Setup distributed
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("EVALUATING STAGE1 CHECKPOINT ON TIMESERIESEXAM")
    print("=" * 80)
    print(f"Checkpoint: {args.stage1_checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print(f"Rank: {rank}, World size: {world_size}")
    print("=" * 80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_id)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )

    # Create model
    print("\n🔧 Creating OpenTSLMFlamingo model...")
    model = OpenTSLMFlamingo(
        cross_attn_every_n_layers=1,
        gradient_checkpointing=False,
        llm_id=args.llm_id,
        device=device,
    ).to(device)

    # Load stage1 checkpoint
    print(f"\n📂 Loading stage1 checkpoint: {args.stage1_checkpoint}")
    if not os.path.exists(args.stage1_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.stage1_checkpoint}")

    checkpoint = torch.load(args.stage1_checkpoint, map_location=device)
    model_state = checkpoint["model_state"]

    # Load state dict (no DDP wrapper needed since we're not using DDP)
    model.load_state_dict(model_state, strict=False)

    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"   Validation loss: {checkpoint.get('val_loss', 'unknown')}")

    # Load TimeSeriesExam test dataset
    print("\n📚 Loading TimeSeriesExam test dataset...")
    test_dataset = TimeSeriesExam1QADataset(
        split="test",
        EOS_TOKEN=tokenizer.eos_token,
    )
    print(f"   Test samples: {len(test_dataset)}")

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        ),
        num_workers=0,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Run evaluation
    print("\n🔍 Running evaluation...")
    model.eval()

    results_file = os.path.join(results_dir, f"test_predictions_rank_{rank}.jsonl")
    final_results_file = os.path.join(results_dir, "test_predictions.jsonl")

    print(f"Saving predictions to: {results_file}")

    all_predictions = []
    all_golds = []

    with open(results_file, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", disable=rank != 0):
                # Generate predictions with max_new_tokens=2000 (same as original)
                predictions = model.generate(batch, max_new_tokens=2000)

                # Collect results
                for sample, pred in zip(batch, predictions):
                    result = {
                        "pre_prompt": sample["pre_prompt"],
                        "time_series_text": sample["time_series_text"],
                        "post_prompt": sample["post_prompt"],
                        "generated": pred,
                        "gold": sample["answer"],
                    }

                    all_predictions.append(pred)
                    all_golds.append(sample["answer"])

                    # Stream write
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()

    # Calculate accuracy
    accuracy = calculate_accuracy(all_predictions, all_golds)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total samples: {len(all_predictions)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 80)

    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "total_samples": len(all_predictions),
        "checkpoint": args.stage1_checkpoint,
        "model_type": "OpenTSLMFlamingo",
        "stage": "stage1_mcq_on_tsexam",
    }

    metrics_file = os.path.join(results_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved to: {metrics_file}")

    # Merge rank files if using DDP (for now just copy since we're using single rank)
    if rank == 0:
        with open(final_results_file, "w", encoding="utf-8") as merged:
            with open(results_file, "r", encoding="utf-8") as rank_file:
                for line in rank_file:
                    merged.write(line)
        print(f"✅ Final predictions saved to: {final_results_file}")

    print("\n✅ Evaluation complete!")

    # Show some example predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (first 3)")
    print("=" * 80)
    for i in range(min(3, len(all_predictions))):
        print(f"\nSample {i+1}:")
        print(f"  Gold: {all_golds[i][:100]}")
        print(f"  Pred: {all_predictions[i][:100]}")


if __name__ == "__main__":
    main()
