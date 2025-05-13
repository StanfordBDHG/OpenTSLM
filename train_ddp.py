import json
import os
import argparse
from typing import List

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from time_series_datasets.TSQADataset import TSQADataset
from time_series_datasets.monash.MonashSPO2QADataset import MonashSPO2QADataset
from time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

from model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from model.llm.TimeSeriesLLM import TimeSeriesLLM
from model.projector.MLPProjector import MLPProjector
from model_config import (
    PATCH_SIZE,
    RESULTS_FILE,
    WARMUP_FRAC,
    WEIGHT_DECAY,
    GRAD_CLIP_NORM,
)

# ---------------------------
# Argument parsing
# ---------------------------
parser = argparse.ArgumentParser(description="Distributed TimeSeries QA Training")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
parser.add_argument(
    "--patch-size",
    type=int,
    default=PATCH_SIZE,
    help="Patch size for time series aggregation",
)
# LOCAL_RANK is set by torch.distributed.launch utility
parser.add_argument(
    "--local_rank",
    type=int,
    default=int(os.environ.get("LOCAL_RANK", 0)),
    help="Local GPU rank",
)
args = parser.parse_args()

NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
PATCH_SIZE = args.patch_size


# ---------------------------
# Distributed setup
# ---------------------------
backend = "nccl" if torch.cuda.is_available() else "gloo"
dist.init_process_group(backend=backend, init_method="env://")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = args.local_rank

# Sanity check on available GPUs
num_gpus = torch.cuda.device_count()
if local_rank >= num_gpus:
    raise ValueError(f"Local rank {local_rank} exceeds available GPUs ({num_gpus})")

# Set device for this process
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cpu")

print(f"Rank {rank} initialized on device {device}")

# ---------------------------
# Model
# ---------------------------
encoder = TransformerCNNEncoder().to(device)
# instantiate model and retrieve EOS token before wrapping in DDP
base_model = TimeSeriesLLM(
    encoder=encoder, projector_class=MLPProjector, device=device
).to(device)
eos_token = base_model.get_eos_token()
# Freeze LLM backbone
for p in base_model.llm.parameters():
    p.requires_grad = False
# Wrap with DistributedDataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(base_model, device_ids=[local_rank] if torch.cuda.is_available() else None)

# ---------------------------
# Optimizer & Scheduler
# ---------------------------
enc_params = list(model.module.encoder.parameters())
proj_params = list(model.module.projector.projector.parameters())
optimizer = AdamW(
    [
        {
            "params": enc_params,
            "lr": float(os.environ.get("LR_ENCODER", 1e-4)),
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": proj_params,
            "lr": float(os.environ.get("LR_PROJECTOR", 1e-3)),
            "weight_decay": WEIGHT_DECAY,
        },
    ]
)


# ---------------------------
# Data loader helper
# ---------------------------
def make_loader(
    datasets: List[Dataset],
    shuffle: bool,
    batch_size: int,
    patch_size: int,
    distribute_data: bool,
):
    ds = ConcatDataset(datasets)
    if distribute_data and dist.is_initialized():
        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        return DataLoader(
            ds,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=lambda b: extend_time_series_to_match_patch_size_and_aggregate(
                b, patch_size=patch_size
            ),
        )
    else:
        return DataLoader(
            ds,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=lambda b: extend_time_series_to_match_patch_size_and_aggregate(
                b, patch_size=patch_size
            ),
        )


QA_DATASET_CLASSES = [TSQADataset]  # , MonashSPO2QADataset]

train_loader = make_loader(
    [cls("train", EOS_TOKEN=eos_token) for cls in QA_DATASET_CLASSES],
    shuffle=True,
    batch_size=BATCH_SIZE,
    patch_size=PATCH_SIZE,
    distribute_data=True,
)
val_loader = make_loader(
    [cls("validation", EOS_TOKEN=eos_token) for cls in QA_DATASET_CLASSES],
    shuffle=False,
    batch_size=BATCH_SIZE,
    patch_size=PATCH_SIZE,
    distribute_data=True,
)
test_loader = make_loader(
    [cls("test", EOS_TOKEN=eos_token) for cls in QA_DATASET_CLASSES],
    shuffle=False,
    batch_size=BATCH_SIZE,
    patch_size=PATCH_SIZE,
    distribute_data=False,
)

# Scheduler (linear warmup + decay)
TOTAL_STEPS = NUM_EPOCHS * len(train_loader)
WARMUP_STEPS = int(WARMUP_FRAC * TOTAL_STEPS)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=TOTAL_STEPS
)

total_train_loader_length_rank_0 = torch.tensor(len(train_loader), device=device)
dist.reduce(total_train_loader_length_rank_0, dst=0, op=dist.ReduceOp.SUM)
total_train_loader_length_rank_0 = total_train_loader_length_rank_0.item()


total_val_loader_length_rank_0 = torch.tensor(len(val_loader), device=device)
dist.reduce(total_val_loader_length_rank_0, dst=0, op=dist.ReduceOp.SUM)
total_val_loader_length_rank_0 = total_val_loader_length_rank_0.item()


print(f"train {total_train_loader_length_rank_0}, val {total_val_loader_length_rank_0}")


# ---------------------------
# Checkpoint helpers
# ---------------------------
def _save_best(epoch: int, val_loss: float):
    if rank == 0:
        # Get the state dict from the base model
        encoder_state = model.module.encoder.state_dict()
        projector_state = model.module.projector.state_dict()

        # Save the state dict
        torch.save(
            {
                "encoder_state": encoder_state,
                "projector_state": projector_state,
                "val_loss": val_loss,
                "epoch": epoch,
            },
            "best_encoder.pt",
        )
        print(f"Saved model checkpoint at epoch {epoch} with val_loss {val_loss:.4f}")

    if dist.is_initialized():
        dist.barrier()


def _load_best():
    if rank == 0 and os.path.exists("best_encoder.pt"):
        ckpt = torch.load("best_encoder.pt", map_location=device)
        # Load state into the base model
        model.module.encoder.load_state_dict(ckpt["encoder_state"])
        model.module.projector.load_state_dict(ckpt["projector_state"])
        print(
            f"Loaded model checkpoint from epoch {ckpt.get('epoch', '?')} with val_loss {ckpt.get('val_loss', '?'):.4f}"
        )

    # Ensure all processes have loaded the model
    if dist.is_initialized():
        dist.barrier()

        # Broadcast model state from rank 0 to all other ranks
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    return ckpt.get("epoch", None) if rank == 0 else None


# ---------------------------
# Evaluation
# ---------------------------
def _evaluate_test():
    model.eval()
    results = []

    # Ensure model is in eval mode and synchronized
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test inference"):
                # Move batch to device
                batch = [
                    {
                        k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in sample.items()
                    }
                    for sample in batch
                ]

                # Generate with the base model
                gens = model.module.generate(batch)

                # Only collect results from rank 0
                if rank == 0:
                    for sample, gen in zip(batch, gens):
                        results.append(
                            {
                                "pre_prompt": sample["pre_prompt"],
                                "time_series_text": sample["time_series_text"],
                                "post_prompt": sample["post_prompt"],
                                "generated": gen,
                                "gold": sample["answer"],
                            }
                        )

    # Ensure all processes have finished generation
    if dist.is_initialized():
        dist.barrier()

    # Save results only on rank 0
    if rank == 0:
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n✅  Test predictions saved to {RESULTS_FILE} (n={len(results)})")


# ---------------------------
# Training loop
# ---------------------------
def train():
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        if dist.is_initialized():
            train_loader.sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0

        if rank == 0:
            prog = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        else:
            prog = train_loader

        for batch in prog:
            optimizer.zero_grad()
            loss = model.module.compute_loss(batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if rank == 0:
                prog.set_postfix(
                    loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )

        train_loss_tensor = torch.tensor(running_loss, device=device)
        dist.reduce(train_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
        avg_train_loss = train_loss_tensor.item() / total_train_loader_length_rank_0

        if rank == 0:
            tqdm.write(f"Epoch {epoch} — train loss: {avg_train_loss:.4f}")

        local_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                local_val_loss += model.module.compute_loss(batch).item()

        val_loss_tensor = torch.tensor(local_val_loss, device=device)
        dist.reduce(val_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
        avg_val_loss = val_loss_tensor.item() / total_val_loader_length_rank_0
        if rank == 0:
            tqdm.write(f"Epoch {epoch} — val   loss: {avg_val_loss:.4f}\n")

        if avg_val_loss + 1e-4 < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            _save_best(epoch, avg_val_loss)
            if rank == 0:
                tqdm.write("✔️  New best model saved.\n")
        else:
            epochs_no_improve += 1
            if rank == 0:
                tqdm.write(
                    f"No improvement for {epochs_no_improve}/{os.environ.get('EARLY_STOP_PAT', 5)} epochs."
                )
            if epochs_no_improve >= int(os.environ.get("EARLY_STOP_PAT", 5)):
                if rank == 0:
                    tqdm.write("\nEarly stopping triggered.")
                break

    if rank == 0:
        tqdm.write("Training finished.\n")

    _load_best()
    _evaluate_test()


if __name__ == "__main__":
    train()
