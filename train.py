import json
import os
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from data import get_loader   # ← note: we use the data loader with test split
from model.time_series_model import TimeSeriesLLM

# ---------------------------
# Device setup
# ---------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ---------------------------
# Hyper‑parameters
# ---------------------------
BATCH_SIZE      = 8
PATCH_SIZE      = 4
NUM_EPOCHS      = 50          # allow many but we will early‑stop
EARLY_STOP_PAT  = 5           # stop if val loss hasn’t improved for this many epochs
LR_ENCODER      = 2e-4
LR_PROJECTOR    = 1e-4
WEIGHT_DECAY    = 1e-2
GRAD_CLIP_NORM  = 1.0
WARMUP_FRAC     = 0.03
MAX_SAMPLES     = None        # set to an int for quick experiments
RESULTS_FILE    = "test_predictions.jsonl"

# ---------------------------
# Model
# ---------------------------
model = TimeSeriesLLM(device=device).to(device)

# — Freeze the LLM backbone so we only update encoder + projector
for p in model.llm.parameters():
    p.requires_grad = False

# Parameter groups with different learning rates
enc_params  = list(model.encoder.parameters())
proj_params = list(model.projector.parameters())
optimizer = AdamW([
    {"params": enc_params,  "lr": LR_ENCODER,   "weight_decay": WEIGHT_DECAY},
    {"params": proj_params, "lr": LR_PROJECTOR, "weight_decay": WEIGHT_DECAY},
])

# ---------------------------
# Data loaders
# ---------------------------
train_loader = get_loader("train",      batch_size=BATCH_SIZE, patch_size=PATCH_SIZE, max_samples=MAX_SAMPLES, EOS_TOKEN=model.get_eos_token())
val_loader   = get_loader("validation", batch_size=1,         patch_size=PATCH_SIZE, EOS_TOKEN=model.get_eos_token())
test_loader  = get_loader("test",       batch_size=1,         patch_size=PATCH_SIZE, shuffle=False, EOS_TOKEN=model.get_eos_token())

# Scheduler (linear warmup + decay)
TOTAL_STEPS  = NUM_EPOCHS * len(train_loader)
WARMUP_STEPS = int(WARMUP_FRAC * TOTAL_STEPS)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=TOTAL_STEPS,
)

# ---------------------------
# Helpers
# ---------------------------

def _save_best(epoch: int, val_loss: float):
    torch.save(
        {
            "encoder_state":   model.encoder.state_dict(),
            "projector_state": model.projector.state_dict(),
            "val_loss":        val_loss,
            "epoch":           epoch,
        },
        "best_encoder.pt",
    )


def _load_best():
    if os.path.exists("best_encoder.pt"):
        ckpt = torch.load("best_encoder.pt", map_location=device)
        model.encoder.load_state_dict(ckpt["encoder_state"])
        model.projector.load_state_dict(ckpt["projector_state"])
        return ckpt.get("epoch", "?")
    return None


def _evaluate_test():
    """Run best model on test set and write prompt+generation+gold to JSONL."""
    model.eval()
    results = []
    with torch.no_grad():
        for ts_batch, prompts, answers in tqdm(test_loader, desc="Test inference"):
            gens = model(prompts, ts_batch)  # free‑generation path
            # each loader batch has batch_size=1
            results.append({
                "prompt":    prompts[0],
                "generated": gens[0],
                "gold":      answers[0],
            })

    # write JSONL
    with open(RESULTS_FILE, "w", encoding="utf‑8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n✅  Test predictions saved to {RESULTS_FILE} (n={len(results)})")

# ---------------------------
# Training loop with early stopping
# ---------------------------

def train():
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ----- training -----
        model.train()
        running_loss = 0.0
        prog = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")

        for ts_batch, prompts, answers in prog:
            optimizer.zero_grad()
            loss = model(prompts, ts_batch, answers)
            loss.backward()
            clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            prog.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_train_loss = running_loss / len(train_loader)
        tqdm.write(f"Epoch {epoch} — train loss: {avg_train_loss:.4f}")

        # ----- validation -----
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for ts_batch, prompts, answers in val_loader:
                loss = model(prompts, ts_batch, answers)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        tqdm.write(f"Epoch {epoch} — val   loss: {val_loss:.4f}\n")

        # ----- early stopping check -----
        if val_loss + 1e-4 < best_val_loss:   # little epsilon to avoid tiny oscillations
            best_val_loss = val_loss
            epochs_no_improve = 0
            _save_best(epoch, val_loss)
            tqdm.write("\u2714️  New best model saved.\n")
        else:
            epochs_no_improve += 1
            tqdm.write(f"No improvement for {epochs_no_improve}/{EARLY_STOP_PAT} epochs.")
            if epochs_no_improve >= EARLY_STOP_PAT:
                tqdm.write("\nEarly stopping triggered.")
                break

    tqdm.write("Training finished.\n")

    # ---------------- test evaluation ----------------
    best_epoch = _load_best()
    if best_epoch is not None:
        print(f"Loaded best checkpoint from epoch {best_epoch} for test evaluation.")
    _evaluate_test()


if __name__ == "__main__":
    train()
