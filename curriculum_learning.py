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
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    FullStateDictConfig,
    StateDictType,
)
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from model.llm.EmbedHealthFlamingo import EmbedHealthFlamingo
from model.llm.EmbedHealthSP import EmbedHealthSP
from model.projector.MLPProjector import MLPProjector
import datetime

from src.model_config import (
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
    While this may look like a lot of code, it's actually quite modular.
    We simply train either EmbedHealthSP or EmbedHealthFlamingo, both using the same training loop.
    We train across different stages:
    - stage1_mcq: Trains the model on a time-series MCQ dataset (TSQA)
    - stage2_captioning: Trains the model on a time-series captioning dataset (M4 time series captioning)

    If you run this script, you should be able to reproduce our results from the paper.
    All datasets are automatically downloaded and processed.
    """
    
    def __init__(self, model_type: str, device: str = None, gradient_checkpointing: bool = False, dist_url: str = "env://", dist_backend: str = "nccl", local_rank: int = int(os.environ.get("LOCAL_RANK", 0))):
        """
        Initialize the curriculum trainer.
        
        Args:
            model_type: Either 'EmbedHealthSP' or 'EmbedHealthFlamingo'
            device: Device to use for training ('cuda', 'mps', or 'cpu')
            gradient_checkpointing: Enable gradient checkpointing
            dist_url: URL used to set up distributed training
            dist_backend: Distributed backend
            local_rank: Local GPU rank
        """
        self.model_type = model_type
        self.device = device or self._get_device()
        
        # Distributed training parameters
        self.gradient_checkpointing = gradient_checkpointing
        self.dist_url = dist_url
        self.dist_backend = dist_backend
        self.local_rank = local_rank
        
        # Initialize distributed training if needed
        self.rank = 0
        self.world_size = 1
        if self._should_use_distributed():
            self._init_distributed()
        
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
                gradient_checkpointing=self.gradient_checkpointing,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Use DDP for multi-GPU training (simpler and more reliable than FSDP)
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.local_rank] if torch.cuda.is_available() else None)
            if self.rank == 0:
                print(f"Wrapped {self.model_type} with DDP for distributed training")
            
        return model
    
    def _get_cast_dtype(self, precision: str):
        """Get cast dtype for mixed precision."""
        if precision == "bf16":
            return torch.bfloat16
        elif precision == "fp16":
            return torch.float16
        else:
            return None
    
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
    
    def _get_optimizer(self, batch_size: int = None, lr_encoder: float = None, lr_projector: float = None, lr_base: float = None):
        """Get optimizer for the model with configurable learning rates."""
        # Get the underlying model (handles DDP wrapping)
        model = self._get_model()
        
        if self.model_type == "EmbedHealthSP":
            # Parameter groups with different learning rates for SP
            enc_params = list(model.encoder.parameters())
            proj_params = list(model.projector.projector.parameters())
            
            # Use provided learning rates or defaults
            encoder_lr = lr_encoder if lr_encoder is not None else LR_ENCODER
            projector_lr = lr_projector if lr_projector is not None else LR_PROJECTOR
            
            if self.rank == 0:
                print(f"📊 Learning rates for {self.model_type}:")
                print(f"   Encoder LR: {encoder_lr:.2e}")
                print(f"   Projector LR: {projector_lr:.2e}")
            
            return AdamW([
                {"params": enc_params, "lr": encoder_lr, "weight_decay": WEIGHT_DECAY},
                {"params": proj_params, "lr": projector_lr, "weight_decay": WEIGHT_DECAY},
            ])
        else:
            # For Flamingo, use grouped parameters
            params_to_optimize = model.named_parameters()
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
            
            # Use provided base learning rate or default
            base_lr = lr_base if lr_base is not None else 2e-4
            
            if self.rank == 0:
                print(f"📊 Learning rate for {self.model_type}:")
                print(f"   Base LR: {base_lr:.2e}")
            
            return torch.optim.AdamW([
                {"params": params_with_wd, "weight_decay": 0.1},
                {"params": params_without_wd, "weight_decay": 0.0},
            ], lr=base_lr)
    
    def _merge_data_loaders(
        self, datasets: List[Dataset], shuffle: bool, batch_size: int, patch_size: int, distribute_data: bool = False
    ) -> DataLoader:
        """Create a merged data loader from multiple datasets."""
        merged_ds = ConcatDataset(datasets)
        
        # Use distributed sampler if distributed training is enabled
        if distribute_data and dist.is_initialized():
            sampler = DistributedSampler(
                merged_ds, 
                num_replicas=self.world_size, 
                rank=self.rank, 
                shuffle=shuffle
            )
            return DataLoader(
                merged_ds,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=patch_size
                ),
            )
        else:
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
        
        # Only save on rank 0 for distributed training
        if dist.is_initialized() and self.rank != 0:
            return
        
        # Get the underlying model (handles DDP wrapping)
        model = self._get_model()
        
        if self.model_type == "EmbedHealthSP":
            checkpoint = {
                "encoder_state": model.encoder.state_dict(),
                "projector_state": model.projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            }
        else:
            # Handle DDP or single GPU case for EmbedHealthFlamingo
            model_state = model.state_dict()
            if hasattr(self.model, 'module'):
                # Remove 'module.' prefix for DDP
                model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
            checkpoint = {
                "model_state": model_state,
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
            
            # Get the underlying model (handles DDP wrapping)
            model = self._get_model()
            
            if self.model_type == "EmbedHealthSP":
                model.encoder.load_state_dict(checkpoint["encoder_state"])
                model.projector.load_state_dict(checkpoint["projector_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            else:
                # Handle DDP or single GPU case for EmbedHealthFlamingo
                model_state = checkpoint["model_state"]
                if hasattr(self.model, 'module'):
                    # Add 'module.' prefix for DDP
                    model_state = {f'module.{k}': v for k, v in model_state.items()}
                
                # Load state dict with strict=False to handle missing keys
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(model_state, strict=False)
                    if missing_keys and self.rank == 0:
                        print(f"⚠️  Warning: Missing keys when loading checkpoint for {stage}:")
                        for key in missing_keys[:10]:  # Show first 10 missing keys
                            print(f"   - {key}")
                        if len(missing_keys) > 10:
                            print(f"   ... and {len(missing_keys) - 10} more keys")
                    if unexpected_keys and self.rank == 0:
                        print(f"⚠️  Warning: Unexpected keys when loading checkpoint for {stage}:")
                        for key in unexpected_keys[:10]:  # Show first 10 unexpected keys
                            print(f"   - {key}")
                        if len(unexpected_keys) > 10:
                            print(f"   ... and {len(unexpected_keys) - 10} more keys")
                except Exception as e:
                    raise RuntimeError(f"Failed to load model state from checkpoint for {stage}: {e}")
                
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
            
            # Get the underlying model (handles DDP wrapping)
            model = self._get_model()
            
            if self.model_type == "EmbedHealthSP":
                model.encoder.load_state_dict(checkpoint["encoder_state"])
                model.projector.load_state_dict(checkpoint["projector_state"])
            else:
                # Handle EmbedHealthFlamingo with graceful loading
                model_state = checkpoint["model_state"]
                if hasattr(self.model, 'module'):
                    # Add 'module.' prefix for DDP
                    model_state = {f'module.{k}': v for k, v in model_state.items()}
                
                # Load state dict with strict=False to handle missing keys
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(model_state, strict=False)
                    if missing_keys and self.rank == 0:
                        print(f"⚠️  Warning: Missing keys when loading previous stage {previous_stage}:")
                        for key in missing_keys[:5]:  # Show first 5 missing keys
                            print(f"   - {key}")
                        if len(missing_keys) > 5:
                            print(f"   ... and {len(missing_keys) - 5} more keys")
                        print(f"   This is normal when transitioning between stages with different model configurations.")
                    if unexpected_keys and self.rank == 0:
                        print(f"⚠️  Warning: Unexpected keys when loading previous stage {previous_stage}:")
                        for key in unexpected_keys[:5]:  # Show first 5 unexpected keys
                            print(f"   - {key}")
                        if len(unexpected_keys) > 5:
                            print(f"   ... and {len(unexpected_keys) - 5} more keys")
                except Exception as e:
                    raise RuntimeError(f"Failed to load model state from previous stage {previous_stage}: {e}")
                
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
            pred_clean = pred.strip()
            gold_clean = gold.strip()
            
            # Check if gold starts with the cleaned prediction (more robust matching)
            if gold_clean.startswith(pred_clean) or pred_clean == gold_clean:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_stage(self, stage: str, test_loader: DataLoader, stage_name: str, 
                       metric_func: Callable = None, epoch: int = None) -> Dict[str, Any]:
        """Evaluate model on test set for a specific stage."""
        # Only evaluate on rank 0 for distributed training
        if dist.is_initialized() and self.rank != 0:
            # Other ranks wait for evaluation to complete
            dist.barrier()
            return {"test_loss": 0.0}
        
        self.model.eval()
        results = []
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {stage_name}", disable=self.rank != 0):
                # Compute loss
                loss = self._get_model().compute_loss(batch)
                test_loss += loss.item()
                
                # Generate predictions
                predictions = self._get_model().generate(batch)
                
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
        if epoch is not None:
            metrics["epoch"] = epoch
        if metric_func:
            predictions = [r["generated"] for r in results]
            gold_answers = [r["gold"] for r in results]
            additional_metrics = metric_func(predictions, gold_answers)
            metrics.update(additional_metrics)
        
        # Save results only on rank 0
        if self.rank == 0:
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
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.4f}")
                else:
                    print(f"   {metric}: {value}")
        
        # Signal other ranks that evaluation is complete
        if dist.is_initialized():
            dist.barrier()
        
        return metrics
    
    def _train_stage(self, stage_name: str, dataset_class, num_epochs: int, 
                    lr_encoder: float, lr_projector: float, lr_base: float, 
                    metric_func: Callable = None, batch_size: int = None) -> Dict[str, Any]:
        """Generic training function for any stage."""
        # Use provided batch_size or default to global BATCH_SIZE
        if batch_size is None:
            batch_size = BATCH_SIZE
            
        print(f"\n🚀 Starting {stage_name} Training with {self.model_type}")
        print("=" * 60)
        print(f"📊 Stage Configuration:")
        print(f"   Epochs: {num_epochs}")
        if self.model_type == "EmbedHealthSP":
            print(f"   Encoder LR: {lr_encoder:.2e}")
            print(f"   Projector LR: {lr_projector:.2e}")
        else:
            print(f"   Base LR: {lr_base:.2e}")
        print(f"   Batch size per GPU: {batch_size}")
        if self.world_size > 1:
            print(f"   Effective batch size: {batch_size * self.world_size}")
        print()
        
        # Load previous stage model and display metrics
        try:
            previous_stage_info = self._load_previous_stage_model(stage_name)
            if previous_stage_info:
                if self.rank == 0:
                    print(f"📂 Loading best model from {previous_stage_info['stage']}:")
                    print(f"   Achieved at epoch: {previous_stage_info['epoch']}")
                    val_loss = previous_stage_info['val_loss']
                    if isinstance(val_loss, (int, float)):
                        print(f"   Validation loss: {val_loss:.4f}")
                    else:
                        print(f"   Validation loss: {val_loss}")
                    for metric, value in previous_stage_info['metrics'].items():
                        if isinstance(value, (int, float)):
                            print(f"   {metric}: {value:.4f}")
                        else:
                            print(f"   {metric}: {value}")
                    print()
            else:
                # Only allow fresh model for first stage
                if stage_name != CURRICULUM_STAGES[0]:
                    raise RuntimeError(f"Cannot start {stage_name} with fresh model. Previous stage {CURRICULUM_STAGES[CURRICULUM_STAGES.index(stage_name) - 1]} must be completed first.")
                if self.rank == 0:
                    print("🆕 Starting with fresh model (first stage)")
                    print()
        except Exception as e:
            if self.rank == 0:
                print(f"❌ Error loading previous stage: {e}")
            raise Exception(f"Error loading previous stage: {e}")
        
        # Initialize optimizer and scheduler
        optimizer = self._get_optimizer(batch_size, lr_encoder, lr_projector, lr_base)
        
        # Create data loaders
        train_loader = self._merge_data_loaders(
            [dataset_class("train", EOS_TOKEN=self._get_model().get_eos_token())],
            shuffle=True,
            batch_size=batch_size,
            patch_size=PATCH_SIZE,
            distribute_data=self.world_size > 1
        )
        
        val_loader = self._merge_data_loaders(
            [dataset_class("validation", EOS_TOKEN=self._get_model().get_eos_token())],
            shuffle=False,
            batch_size=1,
            patch_size=PATCH_SIZE,
            distribute_data=False  # Don't distribute validation
        )
        
        test_loader = self._merge_data_loaders(
            [dataset_class("test", EOS_TOKEN=self._get_model().get_eos_token())],
            shuffle=False,
            batch_size=1,
            patch_size=PATCH_SIZE,
            distribute_data=False  # Don't distribute test
        )
        
        # Scheduler
        total_steps = num_epochs * len(train_loader)
        warmup_steps = int(WARMUP_FRAC * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        if self.rank == 0:
            print(f"📈 Total training steps: {total_steps}")
            print(f"🔥 Warmup steps: {warmup_steps}")
        
        # Load previous checkpoint if exists (for resuming current stage)
        best_epoch, best_val_loss = self._load_checkpoint(stage_name, optimizer, scheduler)
        if best_epoch is not None:
            print(f"📂 Resuming {stage_name} from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
        else:
            print(f"🆕 Starting fresh training for {stage_name}")
            best_val_loss = float("inf")  # Ensure proper initialization
        
        # Training loop
        epochs_no_improve = 0
        
        for epoch in range(1, num_epochs + 1):
            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            # Training
            self.model.train()
            running_loss = 0.0
            prog = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", disable=self.rank != 0)
            
            for batch in prog:
                optimizer.zero_grad()
                loss = self._get_model().compute_loss(batch)
                loss.backward()
                
                # Handle gradient clipping for distributed training
                clip_grad_norm_(self._get_model().parameters(), GRAD_CLIP_NORM)
                
                optimizer.step()
                scheduler.step()
                
                running_loss += loss.item()
                if self.rank == 0:
                    prog.set_postfix(
                        loss=f"{loss.item():.4f}", 
                        lr=f"{scheduler.get_last_lr()[0]:.2e}"
                    )
            
            avg_train_loss = running_loss / len(train_loader)
            if self.rank == 0:
                tqdm.write(f"Epoch {epoch} — train loss: {avg_train_loss:.4f}")
            
            # Validation
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    val_loss += self._get_model().compute_loss(batch).item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Synchronize validation loss across all ranks
            if dist.is_initialized():
                val_loss_tensor = torch.tensor(avg_val_loss, device=self.device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                avg_val_loss = val_loss_tensor.item() / self.world_size
            
            if self.rank == 0:
                tqdm.write(f"Epoch {epoch} — val   loss: {avg_val_loss:.4f}")
                tqdm.write(f"Epoch {epoch} — best  loss: {best_val_loss:.4f}")
            
            # Early stopping - all ranks need to make the same decision
            should_save = avg_val_loss + 1e-4 < best_val_loss
            if dist.is_initialized():
                save_tensor = torch.tensor(1 if should_save else 0, device=self.device)
                dist.all_reduce(save_tensor, op=dist.ReduceOp.SUM)
                should_save = save_tensor.item() > 0  # If any rank thinks we should save, we save
            
            if should_save:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                self._save_checkpoint(stage_name, epoch, avg_val_loss, optimizer, scheduler)
                if self.rank == 0:
                    tqdm.write("✔️  New best model saved.\n")
            else:
                epochs_no_improve += 1
                if self.rank == 0:
                    tqdm.write(f"No improvement for {epochs_no_improve}/{EARLY_STOP_PAT} epochs.\n")
                
                # Synchronize early stopping decision across all ranks
                if epochs_no_improve >= EARLY_STOP_PAT:
                    if self.rank == 0:
                        tqdm.write(f"\nEarly stopping triggered after {epoch} epochs.")
                        tqdm.write(f"Final stats: best_val_loss={best_val_loss:.4f}, epochs_no_improve={epochs_no_improve}")
                    break
            
            # Synchronize best_val_loss and epochs_no_improve across all ranks
            if dist.is_initialized():
                best_loss_tensor = torch.tensor(best_val_loss, device=self.device)
                epochs_tensor = torch.tensor(epochs_no_improve, device=self.device)
                dist.broadcast(best_loss_tensor, src=0)
                dist.broadcast(epochs_tensor, src=0)
                best_val_loss = best_loss_tensor.item()
                epochs_no_improve = int(epochs_tensor.item())
        
        # Load best model and evaluate
        best_epoch, _ = self._load_checkpoint(stage_name, optimizer, scheduler)
        if best_epoch is not None:
            if self.rank == 0:
                print(f"📂 Loaded best checkpoint from epoch {best_epoch} for evaluation.")
        
        if self.rank == 0:
            print(f"🏁 Training completed for {stage_name}")
            print(f"   Total epochs run: {epoch}")
            print(f"   Best validation loss: {best_val_loss:.4f}")
            print(f"   Epochs without improvement: {epochs_no_improve}")
        
        metrics = self._evaluate_stage(stage_name, test_loader, stage_name, metric_func, best_epoch)
        
        return metrics
    
    def stage1_mcq(self, batch_size: int = None) -> Dict[str, Any]:
        """Stage 1: Multiple Choice Question Answering (TSQA).
        
        Configuration:
        - Epochs: 20
        - EmbedHealthSP: encoder_lr=2e-4, projector_lr=1e-4
        - EmbedHealthFlamingo: base_lr=2e-4
        - Metric: Accuracy
        """
        return self._train_stage(
            stage_name="stage1_mcq",
            dataset_class=TSQADataset,
            num_epochs=20,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {"accuracy": self._calculate_accuracy(preds, golds)},
            batch_size=batch_size
        )
    
    def stage2_captioning(self, batch_size: int = None) -> Dict[str, Any]:
        """Stage 2: Caption Generation (M4).
        
        Configuration:
        - Epochs: 15
        - EmbedHealthSP: encoder_lr=1e-4, projector_lr=5e-5 (lower for fine-tuning)
        - EmbedHealthFlamingo: base_lr=1e-4 (lower for fine-tuning)
        - Metric: Test loss only
        """
        return self._train_stage(
            stage_name="stage2_captioning",
            dataset_class=M4QADataset,
            num_epochs=15,
            lr_encoder=1e-4,
            lr_projector=5e-5,
            lr_base=1e-4,
            metric_func=None,  # Only test loss for captioning
            batch_size=batch_size
        )
    
    def run_curriculum(self, stages: List[str] = None, batch_size: int = None):
        """Run the complete curriculum learning pipeline."""
        if stages is None:
            stages = CURRICULUM_STAGES
        
        # Filter out completed stages
        incomplete_stages = []
        for stage in stages:
            if self._is_stage_completed(stage):
                if self.rank == 0:
                    print(f"⏭️  Skipping completed stage: {stage}")
            else:
                incomplete_stages.append(stage)
        
        if self.rank == 0:
            print(f"🎓 Starting Curriculum Learning with {self.model_type}")
            print(f"📊 All stages: {', '.join(stages)}")
            print(f"🔄 Incomplete stages: {', '.join(incomplete_stages)}")
            print(f"💻 Device: {self.device}")
            if batch_size:
                print(f"📦 Batch size: {batch_size}")
            if self.world_size > 1:
                print(f"🌐 Distributed training with {self.world_size} GPUs")
            print("=" * 80)
        
        results = {}
        
        # Run only incomplete stages
        for stage in incomplete_stages:
            # Synchronize all ranks before starting each stage
            if dist.is_initialized():
                dist.barrier()
                
            if stage == "stage1_mcq":
                stage_results = self.stage1_mcq(batch_size=batch_size)
                results[stage] = stage_results
                self._mark_stage_completed(stage, stage_results)
            elif stage == "stage2_captioning":
                stage_results = self.stage2_captioning(batch_size=batch_size)
                results[stage] = stage_results
                self._mark_stage_completed(stage, stage_results)
            else:
                if self.rank == 0:
                    print(f"⚠️  Unknown stage: {stage}, skipping...")
            
            # Synchronize all ranks after completing each stage
            if dist.is_initialized():
                dist.barrier()
        
        # Save overall results only on rank 0
        if self.rank == 0:
            overall_results_file = os.path.join(
                self.results_dir, self.model_type, "curriculum_results.json"
            )
            with open(overall_results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"\n🎉 Curriculum Learning Complete!")
            print(f"📁 All results saved to: {self.results_dir}/{self.model_type}/")
            print(f"📊 Overall results: {overall_results_file}")
        
        return results

    def _should_use_distributed(self) -> bool:
        """Check if distributed training should be used."""
        return (
            "WORLD_SIZE" in os.environ and 
            int(os.environ["WORLD_SIZE"]) > 1
        ) or (
            "LOCAL_RANK" in os.environ and 
            int(os.environ["LOCAL_RANK"]) >= 0
        )
        
    def _init_distributed(self):
        """Initialize distributed training."""
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
        elif "LOCAL_RANK" in os.environ:
            self.rank = int(os.environ["LOCAL_RANK"])
            
        # Initialize process group
        dist.init_process_group(
            backend=self.dist_backend,
            init_method=self.dist_url,
            world_size=self.world_size,
            rank=self.rank,
            timeout=datetime.timedelta(minutes=30),
        )
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        
        if self.rank == 0:
            print(f"Initialized distributed training with {self.world_size} GPUs")

    def _is_stage_completed(self, stage: str) -> bool:
        """Check if a stage is completed by looking for completion flag in metrics."""
        metrics_file = os.path.join(
            self.results_dir, self.model_type, stage, "results", "metrics.json"
        )
        
        if not os.path.exists(metrics_file):
            return False
            
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            return metrics.get("completed", False)
        except:
            return False
    
    def _mark_stage_completed(self, stage: str, metrics: Dict[str, Any]):
        """Mark a stage as completed by adding completion flag to metrics."""
        metrics["completed"] = True
        metrics["completion_epoch"] = metrics.get("epoch", "?")
        
        metrics_file = os.path.join(
            self.results_dir, self.model_type, stage, "results", "metrics.json"
        )
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        if self.rank == 0:
            print(f"✅ Stage {stage} marked as completed")

    def _get_model(self):
        """Get the underlying model (handles DDP wrapping)."""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model


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
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="Batch size for training (default: use value from model_config.py)"
    )
    
    # Distributed training arguments
    parser.add_argument(
        "--gradient_checkpointing", 
        default=False, 
        action="store_true",
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--dist_url", 
        default="env://", 
        type=str,
        help="URL used to set up distributed training"
    )
    parser.add_argument(
        "--dist_backend", 
        default="nccl", 
        type=str,
        help="Distributed backend"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Local GPU rank"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CurriculumTrainer(
        args.model, 
        args.device, 
        gradient_checkpointing=args.gradient_checkpointing,
        dist_url=args.dist_url,
        dist_backend=args.dist_backend,
        local_rank=args.local_rank
    )
    
    # Run curriculum
    results = trainer.run_curriculum(
        args.stages, 
        args.batch_size
    )
    
    # Print summary
    print("\n📈 Final Results Summary:")
    print("=" * 40)
    for stage, metrics in results.items():
        print(f"\n{stage.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")


if __name__ == "__main__":
    main() 