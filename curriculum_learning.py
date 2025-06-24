import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import json
import os as _os
import argparse
from typing import List, Optional, Dict, Any, Callable
from time_series_datasets.TSQADataset import TSQADataset
from time_series_datasets.m4.M4QADataset import M4QADataset
from time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from model.llm.EmbedHealthFlamingo import EmbedHealthFlamingo
from model.llm.EmbedHealthSP import EmbedHealthSP
from model.projector.MLPProjector import MLPProjector
from model_config import (
    BATCH_SIZE,
    EARLY_STOP_PAT,
    GRAD_CLIP_NORM,
    LR_ENCODER,
    LR_PROJECTOR,
    NUM_EPOCHS,
    PATCH_SIZE,
    WARMUP_FRAC,
    WEIGHT_DECAY,
)


# Global stage configuration - users can modify this to mix and match stages
CURRICULUM_STAGES = ["stage1_mcq", "stage2_captioning"]


class CurriculumTrainer:
    """
    Curriculum learning trainer for EmbedHealth models.
    Trains models stage by stage with shared training logic.
    """
    
    def __init__(self, model_type: str, device: str = None):
        """
        Initialize the curriculum trainer.
        
        Args:
            model_type: Either 'EmbedHealthSP' or 'EmbedHealthFlamingo'
            device: Device to use for training ('cuda', 'mps', or 'cpu')
        """
        self.model_type = model_type
        self.device = device or self._get_device()
        self.model = self._initialize_model()
        self.results_dir = "results"
        self._create_results_dir()
        
    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _initialize_model(self):
        """Initialize the specified model type."""
        if self.model_type == "EmbedHealthSP":
            encoder = TransformerCNNEncoder().to(self.device)
            model = EmbedHealthSP(
                encoder=encoder, 
                projector_class=MLPProjector, 
                device=self.device
            ).to(self.device)
            
            # Freeze the LLM backbone for SP model
            for p in model.llm.parameters():
                p.requires_grad = False
                
        elif self.model_type == "EmbedHealthFlamingo":
            model = EmbedHealthFlamingo(
                device=self.device,
                cross_attn_every_n_layers=1,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        return model
    
    def _create_results_dir(self):
        """Create the results directory structure."""
        os.makedirs(self.results_dir, exist_ok=True)
        model_dir = os.path.join(self.results_dir, self.model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create stage directories based on global configuration
        for stage in CURRICULUM_STAGES:
            stage_dir = os.path.join(model_dir, stage)
            os.makedirs(stage_dir, exist_ok=True)
            os.makedirs(os.path.join(stage_dir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(stage_dir, "results"), exist_ok=True)
    
    def _get_optimizer(self):
        """Get optimizer for the model."""
        if self.model_type == "EmbedHealthSP":
            # Parameter groups with different learning rates for SP
            enc_params = list(self.model.encoder.parameters())
            proj_params = list(self.model.projector.projector.parameters())
            return AdamW([
                {"params": enc_params, "lr": LR_ENCODER, "weight_decay": WEIGHT_DECAY},
                {"params": proj_params, "lr": LR_PROJECTOR, "weight_decay": WEIGHT_DECAY},
            ])
        else:
            # For Flamingo, use grouped parameters
            params_to_optimize = self.model.named_parameters()
            params_to_optimize = list(
                filter(
                    lambda x: x[1].requires_grad
                    and not getattr(x[1], "exclude_from_optimizer", False),
                    params_to_optimize,
                )
            )
            
            # Group parameters for weight decay
            params_with_wd, params_without_wd = [], []
            for n, p in params_to_optimize:
                if "gated_cross_attn" in n:
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)
            
            return torch.optim.AdamW([
                {"params": params_with_wd, "weight_decay": 0.1},
                {"params": params_without_wd, "weight_decay": 0.0},
            ], lr=2e-4)
    
    def _merge_data_loaders(
        self, datasets: List[Dataset], shuffle: bool, batch_size: int, patch_size: int
    ) -> DataLoader:
        """Create a merged data loader from multiple datasets."""
        merged_ds = ConcatDataset(datasets)
        return DataLoader(
            merged_ds,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                batch, patch_size=patch_size
            ),
        )
    
    def _save_checkpoint(self, stage: str, epoch: int, val_loss: float, optimizer, scheduler):
        """Save model checkpoint for a specific stage."""
        checkpoint_dir = os.path.join(self.results_dir, self.model_type, stage, "checkpoints")
        
        if self.model_type == "EmbedHealthSP":
            checkpoint = {
                "encoder_state": self.model.encoder.state_dict(),
                "projector_state": self.model.projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            }
        else:
            checkpoint = {
                "model_state": self.model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            }
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pt"))
    
    def _load_checkpoint(self, stage: str, optimizer, scheduler):
        """Load model checkpoint for a specific stage."""
        checkpoint_path = os.path.join(
            self.results_dir, self.model_type, stage, "checkpoints", "best_model.pt"
        )
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if self.model_type == "EmbedHealthSP":
                self.model.encoder.load_state_dict(checkpoint["encoder_state"])
                self.model.projector.load_state_dict(checkpoint["projector_state"])
            else:
                self.model.load_state_dict(checkpoint["model_state"])
            
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            
            return checkpoint.get("epoch", "?"), checkpoint.get("val_loss", float("inf"))
        return None, float("inf")
    
    def _load_previous_stage_model(self, current_stage: str) -> Optional[Dict[str, Any]]:
        """Load the best model from the previous stage and return its metrics."""
        try:
            current_idx = CURRICULUM_STAGES.index(current_stage)
            if current_idx == 0:
                # First stage, no previous model to load
                return None
            
            previous_stage = CURRICULUM_STAGES[current_idx - 1]
            metrics_file = os.path.join(
                self.results_dir, self.model_type, previous_stage, "results", "metrics.json"
            )
            
            if not os.path.exists(metrics_file):
                raise RuntimeError(f"Previous stage {previous_stage} metrics file not found: {metrics_file}")
            
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            
            # Load the model weights from previous stage
            checkpoint_path = os.path.join(
                self.results_dir, self.model_type, previous_stage, "checkpoints", "best_model.pt"
            )
            
            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f"Previous stage {previous_stage} checkpoint not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if self.model_type == "EmbedHealthSP":
                self.model.encoder.load_state_dict(checkpoint["encoder_state"])
                self.model.projector.load_state_dict(checkpoint["projector_state"])
            else:
                self.model.load_state_dict(checkpoint["model_state"])
            
            return {
                "stage": previous_stage,
                "metrics": metrics,
                "epoch": checkpoint.get("epoch", "?"),
                "val_loss": checkpoint.get("val_loss", "?")
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to load previous stage model: {e}")
    
    def _calculate_accuracy(self, predictions: List[str], gold_answers: List[str]) -> float:
        """Calculate accuracy for MCQ tasks."""
        correct = 0
        total = len(predictions)
        
        for pred, gold in zip(predictions, gold_answers):
            # Clean up predictions and gold answers
            pred_clean = pred.strip().lower()
            gold_clean = gold.strip().lower()
            
            if pred_clean == gold_clean:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_stage(self, stage: str, test_loader: DataLoader, stage_name: str, 
                       metric_func: Callable = None) -> Dict[str, Any]:
        """Evaluate model on test set for a specific stage."""
        self.model.eval()
        results = []
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {stage_name}"):
                # Compute loss
                loss = self.model.compute_loss(batch)
                test_loss += loss.item()
                
                # Generate predictions
                predictions = self.model.generate(batch)
                
                # Collect results
                for sample, pred in zip(batch, predictions):
                    result = {
                        "pre_prompt": sample["pre_prompt"],
                        "time_series_text": sample["time_series_text"],
                        "post_prompt": sample["post_prompt"],
                        "generated": pred,
                        "gold": sample["answer"],
                    }
                    results.append(result)
        
        avg_test_loss = test_loss / len(test_loader)
        
        # Calculate stage-specific metrics
        metrics = {"test_loss": avg_test_loss}
        if metric_func:
            predictions = [r["generated"] for r in results]
            gold_answers = [r["gold"] for r in results]
            additional_metrics = metric_func(predictions, gold_answers)
            metrics.update(additional_metrics)
        
        # Save results
        results_file = os.path.join(
            self.results_dir, self.model_type, stage, "results", "test_predictions.jsonl"
        )
        with open(results_file, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        # Save metrics
        metrics_file = os.path.join(
            self.results_dir, self.model_type, stage, "results", "metrics.json"
        )
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ {stage_name} evaluation complete:")
        print(f"   Test predictions saved to: {results_file}")
        print(f"   Metrics saved to: {metrics_file}")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        return metrics
    
    def _train_stage(self, stage: str, stage_name: str, dataset_class, 
                    metric_func: Callable = None) -> Dict[str, Any]:
        """Generic training function for any stage."""
        print(f"\n🚀 Starting {stage_name} Training with {self.model_type}")
        print("=" * 60)
        
        # Load previous stage model and display metrics
        try:
            previous_stage_info = self._load_previous_stage_model(stage)
            if previous_stage_info:
                print(f"📂 Loading best model from {previous_stage_info['stage']}:")
                print(f"   Achieved at epoch: {previous_stage_info['epoch']}")
                print(f"   Validation loss: {previous_stage_info['val_loss']:.4f}")
                for metric, value in previous_stage_info['metrics'].items():
                    print(f"   {metric}: {value:.4f}")
                print()
            else:
                # Only allow fresh model for first stage
                if stage != CURRICULUM_STAGES[0]:
                    raise RuntimeError(f"Cannot start {stage} with fresh model. Previous stage {CURRICULUM_STAGES[CURRICULUM_STAGES.index(stage) - 1]} must be completed first.")
                print("🆕 Starting with fresh model (first stage)")
                print()
        except Exception as e:
            print(f"❌ Error loading previous stage: {e}")
            raise
        
        # Initialize optimizer and scheduler
        optimizer = self._get_optimizer()
        
        # Create data loaders
        train_loader = self._merge_data_loaders(
            [dataset_class("train", EOS_TOKEN=self.model.get_eos_token())],
            shuffle=True,
            batch_size=BATCH_SIZE,
            patch_size=PATCH_SIZE,
        )
        
        val_loader = self._merge_data_loaders(
            [dataset_class("validation", EOS_TOKEN=self.model.get_eos_token())],
            shuffle=False,
            batch_size=1,
            patch_size=PATCH_SIZE,
        )
        
        test_loader = self._merge_data_loaders(
            [dataset_class("test", EOS_TOKEN=self.model.get_eos_token())],
            shuffle=False,
            batch_size=1,
            patch_size=PATCH_SIZE,
        )
        
        # Scheduler
        total_steps = NUM_EPOCHS * len(train_loader)
        warmup_steps = int(WARMUP_FRAC * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Load previous checkpoint if exists (for resuming current stage)
        best_epoch, best_val_loss = self._load_checkpoint(stage, optimizer, scheduler)
        if best_epoch is not None:
            print(f"📂 Resuming {stage} from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
        
        # Training loop
        epochs_no_improve = 0
        
        for epoch in range(1, NUM_EPOCHS + 1):
            # Training
            self.model.train()
            running_loss = 0.0
            prog = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
            
            for batch in prog:
                optimizer.zero_grad()
                loss = self.model.compute_loss(batch)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
                
                running_loss += loss.item()
                prog.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )
            
            avg_train_loss = running_loss / len(train_loader)
            tqdm.write(f"Epoch {epoch} — train loss: {avg_train_loss:.4f}")
            
            # Validation
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    val_loss += self.model.compute_loss(batch).item()
            
            avg_val_loss = val_loss / len(val_loader)
            tqdm.write(f"Epoch {epoch} — val   loss: {avg_val_loss:.4f}\n")
            
            # Early stopping
            if avg_val_loss + 1e-4 < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                self._save_checkpoint(stage, epoch, avg_val_loss, optimizer, scheduler)
                tqdm.write("✔️  New best model saved.\n")
            else:
                epochs_no_improve += 1
                tqdm.write(f"No improvement for {epochs_no_improve}/{EARLY_STOP_PAT} epochs.")
                if epochs_no_improve >= EARLY_STOP_PAT:
                    tqdm.write("\nEarly stopping triggered.")
                    break
        
        # Load best model and evaluate
        best_epoch, _ = self._load_checkpoint(stage, optimizer, scheduler)
        if best_epoch is not None:
            print(f"📂 Loaded best checkpoint from epoch {best_epoch} for evaluation.")
        
        metrics = self._evaluate_stage(stage, test_loader, stage_name, metric_func)
        return metrics
    
    def stage1_mcq(self) -> Dict[str, Any]:
        """Stage 1: Multiple Choice Question Answering (TSQA)."""
        return self._train_stage(
            stage="stage1_mcq",
            stage_name="MCQ (TSQA)",
            dataset_class=TSQADataset,
            metric_func=lambda preds, golds: {"accuracy": self._calculate_accuracy(preds, golds)}
        )
    
    def stage2_captioning(self) -> Dict[str, Any]:
        """Stage 2: Caption Generation (M4)."""
        return self._train_stage(
            stage="stage2_captioning",
            stage_name="Captioning (M4)",
            dataset_class=M4QADataset,
            metric_func=None  # Only test loss for captioning
        )
    
    def run_curriculum(self, stages: List[str] = None):
        """Run the complete curriculum learning pipeline."""
        if stages is None:
            stages = CURRICULUM_STAGES
        
        print(f"🎓 Starting Curriculum Learning with {self.model_type}")
        print(f"📊 Stages: {', '.join(stages)}")
        print(f"💻 Device: {self.device}")
        print("=" * 80)
        
        results = {}
        
        for stage in stages:
            if stage == "stage1_mcq":
                results[stage] = self.stage1_mcq()
            elif stage == "stage2_captioning":
                results[stage] = self.stage2_captioning()
            else:
                print(f"⚠️  Unknown stage: {stage}, skipping...")
        
        # Save overall results
        overall_results_file = os.path.join(
            self.results_dir, self.model_type, "curriculum_results.json"
        )
        with open(overall_results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Curriculum Learning Complete!")
        print(f"📁 All results saved to: {self.results_dir}/{self.model_type}/")
        print(f"📊 Overall results: {overall_results_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Curriculum Learning for EmbedHealth Models")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["EmbedHealthSP", "EmbedHealthFlamingo"],
        required=True,
        help="Model type to train"
    )
    parser.add_argument(
        "--stages", 
        nargs="+", 
        choices=CURRICULUM_STAGES,
        default=CURRICULUM_STAGES,
        help="Stages to run (default: all stages)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to use (cuda, mps, cpu)"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CurriculumTrainer(args.model, args.device)
    
    # Run curriculum
    results = trainer.run_curriculum(args.stages)
    
    # Print summary
    print("\n📈 Final Results Summary:")
    print("=" * 40)
    for stage, metrics in results.items():
        print(f"\n{stage.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main() 