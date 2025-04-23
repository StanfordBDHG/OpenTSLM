# train.py
import torch
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from data import get_loader
from model.time_series_model import TimeSeriesLLM

# device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# hyperparams
batch_size    = 8
patch_size    = 4
num_epochs    = 20
learning_rate = 8e-4
warmup_frac   = 0.01     # fraction of steps used for linear warmup
max_samples = 5000 

# model + LoRA
model = TimeSeriesLLM(device=device).to(device)
lora_config = LoraConfig(
    # Adapter rank
    r=16,
    # LoRA scaling α
    lora_alpha=32,
    # Which modules to inject adapters into
    target_modules=[
        'q_proj',     # query projection
        'k_proj',     # key projection
        'v_proj',     # value projection
        'o_proj'      # output projection
    ],
    # Whether to add LoRA bias; 'none' means no extra bias terms
    bias='none',
    # Task type for a causal language model
    task_type='CAUSAL_LM'
)
model.llm = get_peft_model(model.llm, lora_config)
optimizer = AdamW(model.parameters(), lr=learning_rate)

# data loaders
train_loader = get_loader('train',      batch_size, patch_size, max_samples=max_samples)
val_loader   = get_loader('validation', batch_size=1, patch_size=patch_size)

# linear scheduler with warmup
total_steps  = num_epochs * len(train_loader)
warmup_steps = int(warmup_frac * total_steps)
scheduler    = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

def train():
    for epoch in range(1, num_epochs + 1):
        model.train()
        prog = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        total_loss = 0.0

        for ts_batch, prompts, answers in prog:
            optimizer.zero_grad()
            loss = model(prompts, ts_batch, answers)  # differentiable CE loss
            loss.backward()
            optimizer.step()
            scheduler.step()                         # <-- step the LR scheduler

            total_loss += loss.item()
            prog.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg = total_loss / len(train_loader)
        tqdm.write(f"Epoch {epoch} — avg loss: {avg:.4f}")

        # validation
        model.eval()
        tqdm.write(f"\n--- Validation after Epoch {epoch} ---")
        with torch.no_grad():
            for i, (ts_batch, prompts, answers) in enumerate(val_loader, 1):
                if i > 5: break
                gen = model(prompts, ts_batch)  # inference path
                tqdm.write(f"Example {i}")
                tqdm.write(f"Prompt: {prompts[0]}")
                tqdm.write(f"Gen:    {gen[0]}")
                tqdm.write(f"Gold:   {answers[0]}\n")
        model.train()

    tqdm.write("Training complete.")

if __name__ == '__main__':
    train()
