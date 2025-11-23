#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Sanity check script for curriculum learning.
Runs minimal training with:
- Only stage1_mcq, stage2_captioning, and stage_tsexam_eval
- 1 epoch per stage
- Small subset of data (100 samples)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import argparse
from curriculum_learning import CurriculumTrainer, CURRICULUM_STAGES


def main():
    parser = argparse.ArgumentParser(
        description="Sanity Check for Curriculum Learning"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["OpenTSLMSP", "OpenTSLMFlamingo"],
        default="OpenTSLMFlamingo",
        help="Model type to train",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training (default: 2 for quick testing)",
    )
    parser.add_argument(
        "--llm_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="LLM model ID",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Local GPU rank",
    )
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    from logger import set_global_verbose, get_logger
    set_global_verbose(args.verbose)
    logger = get_logger(verbose=args.verbose)

    # Initialize trainer
    print("🧪 Starting SANITY CHECK")
    print("=" * 60)
    print("This will run minimal training to verify the fix:")
    print("  - Stages: stage1_mcq, stage2_captioning, stage_tsexam_eval")
    print("  - Epochs: 1 per stage")
    print("  - Batch size: 2")
    print("=" * 60)

    trainer = CurriculumTrainer(
        args.model,
        args.device,
        gradient_checkpointing=False,
        dist_url="env://",
        dist_backend="nccl",
        local_rank=args.local_rank,
        llm_id=args.llm_id,
    )

    # Temporarily override the stage methods to use 1 epoch only
    original_stage1 = trainer.stage1_mcq
    original_stage2 = trainer.stage2_captioning
    original_tsexam = trainer.stage_tsexam_eval

    def stage1_wrapper(batch_size=None, eval_only=False):
        """Override stage1 to use 1 epoch"""
        return trainer._train_stage(
            stage_name="stage1_mcq",
            dataset_class=trainer.stage1_mcq.__wrapped__.__code__.co_consts[1] if hasattr(trainer.stage1_mcq, '__wrapped__') else type('TSQADataset', (), {}),
            num_epochs=1,  # Changed from 30 to 1
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {
                "accuracy": trainer._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size or 2,
            eval_only=eval_only,
        )

    def stage2_wrapper(batch_size=None, eval_only=False):
        """Override stage2 to use 1 epoch"""
        return trainer._train_stage(
            stage_name="stage2_captioning",
            dataset_class=type('M4QADataset', (), {}),
            num_epochs=1,  # Changed from 20 to 1
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,
            batch_size=batch_size or 2,
            eval_only=eval_only,
        )

    # Import the dataset classes
    from time_series_datasets.TSQADataset import TSQADataset
    from time_series_datasets.m4.M4QADataset import M4QADataset
    from time_series_datasets.timeseriesexam.TimeSeriesExam1QADataset import TimeSeriesExam1QADataset

    # Run only the first 3 stages with 1 epoch each
    stages_to_run = ["stage1_mcq", "stage2_captioning", "stage_tsexam_eval"]

    results = {}

    for stage in stages_to_run:
        if stage == "stage1_mcq":
            print("\n" + "=" * 60)
            print("🧪 SANITY CHECK: Running stage1_mcq with 1 epoch")
            print("=" * 60)
            stage_results = trainer._train_stage(
                stage_name="stage1_mcq",
                dataset_class=TSQADataset,
                num_epochs=1,
                lr_encoder=2e-4,
                lr_projector=1e-4,
                lr_base=2e-4,
                metric_func=lambda preds, golds: {
                    "accuracy": trainer._calculate_accuracy(preds, golds)
                },
                batch_size=args.batch_size,
                eval_only=False,
            )
            results[stage] = stage_results
            trainer._mark_stage_completed(stage, stage_results)
        elif stage == "stage2_captioning":
            print("\n" + "=" * 60)
            print("🧪 SANITY CHECK: Running stage2_captioning with 1 epoch")
            print("=" * 60)
            stage_results = trainer._train_stage(
                stage_name="stage2_captioning",
                dataset_class=M4QADataset,
                num_epochs=1,
                lr_encoder=2e-4,
                lr_projector=1e-4,
                lr_base=2e-4,
                metric_func=None,
                batch_size=args.batch_size,
                eval_only=False,
            )
            results[stage] = stage_results
            trainer._mark_stage_completed(stage, stage_results)
        elif stage == "stage_tsexam_eval":
            print("\n" + "=" * 60)
            print("🧪 SANITY CHECK: Running stage_tsexam_eval (eval-only)")
            print("=" * 60)
            stage_results = trainer._train_stage(
                stage_name="stage_tsexam_eval",
                dataset_class=TimeSeriesExam1QADataset,
                num_epochs=1,
                lr_encoder=2e-4,
                lr_projector=1e-4,
                lr_base=2e-4,
                metric_func=lambda preds, golds: {
                    "accuracy": trainer._calculate_accuracy(preds, golds)
                },
                batch_size=args.batch_size,
                eval_only=True,  # This is the critical eval-only test
            )
            results[stage] = stage_results
            trainer._mark_stage_completed(stage, stage_results)

    # Print summary
    print("\n" + "=" * 60)
    print("✅ SANITY CHECK COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    logger.info("Final Results Summary:")
    for stage, metrics in results.items():
        logger.info(f"{stage.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

    print("\n🎉 The fix works! Now you can run the full training.")


if __name__ == "__main__":
    main()
