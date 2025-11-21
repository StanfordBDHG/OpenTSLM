#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""
Interleaved Training Script for TSQA and M4

This script trains a model by interleaving 1 epoch of TSQA followed by 1 epoch of M4,
repeated 30 times (60 total epochs), then evaluates on TimeSeriesExam without training on it.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import json
import argparse
import datetime
from typing import Dict, Any
from time_series_datasets.TSQADataset import TSQADataset
from time_series_datasets.m4.M4QADataset import M4QADataset
from time_series_datasets.timeseriesexam.TimeSeriesExam1QADataset import TimeSeriesExam1QADataset
from time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from logger import get_logger, set_global_verbose

from model_config import (
    BATCH_SIZE,
    EARLY_STOP_PAT,
    GRAD_CLIP_NORM,
    WARMUP_FRAC,
    WEIGHT_DECAY,
    PATCH_SIZE,
)


class InterleavedTrainer:
    """
    Trainer that interleaves TSQA and M4 training for 30 cycles (60 total epochs),
    then evaluates on TimeSeriesExam.
    """

    def __init__(
        self,
        model,
        device: str,
        results_dir: str,
        llm_id: str,
        batch_size: int = BATCH_SIZE,
        num_cycles: int = 30,
        lr_base: float = 2e-4,
    ):
        self.model = model
        self.device = device
        self.results_dir = results_dir
        self.llm_id = llm_id
        self.batch_size = batch_size
        self.num_cycles = num_cycles
        self.lr_base = lr_base
        self.logger = get_logger()

        # Create session-specific log directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(results_dir, f"interleaved_session_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)

        # Session log file
        self.session_log_file = os.path.join(self.session_dir, "session_log.txt")
        self._log_session(f"=" * 80)
        self._log_session(f"Interleaved Training Session Started: {timestamp}")
        self._log_session(f"Model: {llm_id}")
        self._log_session(f"Results Directory: {self.session_dir}")
        self._log_session(f"Num Cycles: {num_cycles} (Total epochs: {num_cycles * 2})")
        self._log_session(f"Batch Size: {batch_size}")
        self._log_session(f"Learning Rate: {lr_base}")
        self._log_session(f"=" * 80)
        self._log_session("")

    def _log_session(self, message: str):
        """Log message to session log file."""
        with open(self.session_log_file, "a") as f:
            f.write(f"{message}\n")
        print(message)

    def _save_loss_history(self, epoch: int, dataset_name: str, train_loss: float, val_loss: float):
        """Save loss history for each dataset."""
        loss_file = os.path.join(self.session_dir, f"{dataset_name}_loss_history.txt")

        # Create header if file doesn't exist
        if not os.path.exists(loss_file):
            with open(loss_file, "w") as f:
                f.write("Epoch\tTrain_Loss\tVal_Loss\n")
                f.write("-" * 30 + "\n")

        # Append loss
        with open(loss_file, "a") as f:
            f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")

    def _save_checkpoint(self, cycle: int, dataset_name: str, val_loss: float, optimizer, scheduler):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.session_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss": val_loss,
            "cycle": cycle,
            "dataset": dataset_name,
        }

        checkpoint_path = os.path.join(checkpoint_dir, f"cycle_{cycle}_{dataset_name}.pt")
        torch.save(checkpoint, checkpoint_path)
        self._log_session(f"💾 Checkpoint saved: {checkpoint_path}")

    def _train_epoch(self, train_loader, optimizer, scheduler, epoch: int, dataset_name: str):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch} [{dataset_name}] Training",
            disable=False,
        )

        for batch in progress_bar:
            # Compute loss (batch is already processed by collate_fn)
            loss = self.model.compute_loss(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        return avg_loss

    def _validate(self, val_loader, dataset_name: str):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[{dataset_name}] Validation", disable=False):
                # Compute loss (batch is already processed by collate_fn)
                loss = self.model.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def _evaluate_timeseriesexam(self):
        """Evaluate on TimeSeriesExam test set."""
        self._log_session("\n" + "=" * 80)
        self._log_session("Starting TimeSeriesExam Evaluation")
        self._log_session("=" * 80)

        # Load test dataset
        test_dataset = TimeSeriesExam1QADataset("test", EOS_TOKEN=self.model.get_eos_token())
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                batch, patch_size=PATCH_SIZE
            ),
        )

        self.model.eval()
        predictions = []
        gold_answers = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="TimeSeriesExam Evaluation"):
                # Generate prediction (batch is already processed by collate_fn)
                generated_texts = self.model.generate(batch, max_new_tokens=2000)

                # Collect predictions and gold answers
                for sample, pred in zip(batch, generated_texts):
                    predictions.append(pred)
                    gold_answers.append(sample["answer"])

        # Calculate accuracy
        accuracy = self._calculate_accuracy(predictions, gold_answers)

        # Save results
        results_file = os.path.join(self.session_dir, "timeseriesexam_results.json")
        results = {
            "accuracy": accuracy,
            "num_samples": len(predictions),
            "timestamp": datetime.datetime.now().isoformat(),
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save predictions
        predictions_file = os.path.join(self.session_dir, "timeseriesexam_predictions.jsonl")
        with open(predictions_file, "w") as f:
            for pred, gold in zip(predictions, gold_answers):
                f.write(json.dumps({"prediction": pred, "gold": gold}) + "\n")

        self._log_session(f"\n📊 TimeSeriesExam Results:")
        self._log_session(f"   Accuracy: {accuracy:.4f}")
        self._log_session(f"   Num Samples: {len(predictions)}")
        self._log_session(f"   Results saved to: {results_file}")
        self._log_session(f"   Predictions saved to: {predictions_file}")

        return results

    def _calculate_accuracy(self, predictions, gold_answers):
        """Calculate accuracy for multiple choice questions."""
        correct = 0
        for pred, gold in zip(predictions, gold_answers):
            # Extract answer from prediction (assumes format like "Answer: A" or just "A")
            pred_clean = pred.strip().upper()
            gold_clean = gold.strip().upper()

            # Check if prediction contains the gold answer
            if gold_clean in pred_clean:
                correct += 1
            # Also check if they match exactly
            elif pred_clean == gold_clean:
                correct += 1

        return correct / len(predictions) if predictions else 0.0

    def train_interleaved(self):
        """Main training loop with interleaved TSQA and M4."""
        self._log_session("\n" + "=" * 80)
        self._log_session("Starting Interleaved Training (TSQA <-> M4)")
        self._log_session("=" * 80 + "\n")

        # Load datasets
        tsqa_train = TSQADataset("train", EOS_TOKEN=self.model.get_eos_token())
        tsqa_val = TSQADataset("validation", EOS_TOKEN=self.model.get_eos_token())
        m4_train = M4QADataset("train", EOS_TOKEN=self.model.get_eos_token())
        m4_val = M4QADataset("validation", EOS_TOKEN=self.model.get_eos_token())

        # Create data loaders
        tsqa_train_loader = DataLoader(
            tsqa_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                batch, patch_size=PATCH_SIZE
            ),
        )
        tsqa_val_loader = DataLoader(
            tsqa_val,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                batch, patch_size=PATCH_SIZE
            ),
        )
        m4_train_loader = DataLoader(
            m4_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                batch, patch_size=PATCH_SIZE
            ),
        )
        m4_val_loader = DataLoader(
            m4_val,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                batch, patch_size=PATCH_SIZE
            ),
        )

        # Initialize optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr_base,
            weight_decay=WEIGHT_DECAY,
        )

        # Calculate total steps for both datasets (30 cycles = 60 epochs total)
        total_steps = self.num_cycles * (len(tsqa_train_loader) + len(m4_train_loader))
        warmup_steps = int(WARMUP_FRAC * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self._log_session(f"📈 Total training steps: {total_steps}")
        self._log_session(f"🔥 Warmup steps: {warmup_steps}")
        self._log_session(f"📊 TSQA train samples: {len(tsqa_train)}")
        self._log_session(f"📊 M4 train samples: {len(m4_train)}")
        self._log_session("")

        # Training loop - interleave TSQA and M4 for 30 cycles
        best_val_loss = float("inf")
        epoch_counter = 0

        for cycle in range(1, self.num_cycles + 1):
            self._log_session(f"\n{'='*80}")
            self._log_session(f"CYCLE {cycle}/{self.num_cycles}")
            self._log_session(f"{'='*80}")

            # Train on TSQA for 1 epoch
            epoch_counter += 1
            self._log_session(f"\n--- Training on TSQA (Epoch {epoch_counter}) ---")
            tsqa_train_loss = self._train_epoch(
                tsqa_train_loader, optimizer, scheduler, epoch_counter, "TSQA"
            )
            tsqa_val_loss = self._validate(tsqa_val_loader, "TSQA")

            self._log_session(f"TSQA Epoch {epoch_counter} - Train Loss: {tsqa_train_loss:.6f}, Val Loss: {tsqa_val_loss:.6f}")
            self._save_loss_history(epoch_counter, "TSQA", tsqa_train_loss, tsqa_val_loss)

            # Train on M4 for 1 epoch
            epoch_counter += 1
            self._log_session(f"\n--- Training on M4 (Epoch {epoch_counter}) ---")
            m4_train_loss = self._train_epoch(
                m4_train_loader, optimizer, scheduler, epoch_counter, "M4"
            )
            m4_val_loss = self._validate(m4_val_loader, "M4")

            self._log_session(f"M4 Epoch {epoch_counter} - Train Loss: {m4_train_loss:.6f}, Val Loss: {m4_val_loss:.6f}")
            self._save_loss_history(epoch_counter, "M4", m4_train_loss, m4_val_loss)

            # Calculate average validation loss for this cycle
            avg_val_loss = (tsqa_val_loss + m4_val_loss) / 2

            # Save checkpoint if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_checkpoint(cycle, "interleaved", avg_val_loss, optimizer, scheduler)
                self._log_session(f"✅ New best validation loss: {best_val_loss:.6f}")

        self._log_session(f"\n{'='*80}")
        self._log_session("Interleaved Training Completed!")
        self._log_session(f"Best Average Validation Loss: {best_val_loss:.6f}")
        self._log_session(f"{'='*80}\n")

        # Evaluate on TimeSeriesExam
        eval_results = self._evaluate_timeseriesexam()

        # Save final summary
        summary = {
            "training_completed": True,
            "num_cycles": self.num_cycles,
            "total_epochs": epoch_counter,
            "best_val_loss": best_val_loss,
            "timeseriesexam_accuracy": eval_results["accuracy"],
            "session_dir": self.session_dir,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        summary_file = os.path.join(self.session_dir, "training_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        self._log_session(f"\n📋 Training summary saved to: {summary_file}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Interleaved training on TSQA and M4, then evaluation on TimeSeriesExam"
    )
    parser.add_argument(
        "--llm_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model ID for LLM",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--num_cycles",
        type=int,
        default=30,
        help="Number of interleaving cycles (default: 30, total 60 epochs)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_interleaved",
        help="Directory to save results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set verbose mode
    if args.verbose:
        set_global_verbose(True)

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Initialize model
    print(f"🔧 Initializing OpenTSLMFlamingo with {args.llm_id}")
    model = OpenTSLMFlamingo(
        cross_attn_every_n_layers=1,
        gradient_checkpointing=False,
        llm_id=args.llm_id,
        device=args.device,
    ).to(args.device)

    # Initialize trainer
    trainer = InterleavedTrainer(
        model=model,
        device=args.device,
        results_dir=args.results_dir,
        llm_id=args.llm_id,
        batch_size=args.batch_size,
        num_cycles=args.num_cycles,
        lr_base=args.lr,
    )

    # Run training
    summary = trainer.train_interleaved()

    print("\n" + "=" * 80)
    print("✅ Training and Evaluation Complete!")
    print(f"📂 Results saved to: {summary['session_dir']}")
    print(f"📊 TimeSeriesExam Accuracy: {summary['timeseriesexam_accuracy']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
