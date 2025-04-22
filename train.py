import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig
from model.time_series_model import TimeSeriesLLM
from data import get_loader
from tqdm.auto import tqdm

# Configuration
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

batch_size = 2
patch_size = 4
num_epochs = 2
learning_rate = 2e-5

# Initialize model and LoRA
model = TimeSeriesLLM(device=device).to(device)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    bias='none',
    task_type='CAUSAL_LM'
)
model.llm = get_peft_model(model.llm, lora_config)
optimizer = AdamW(model.parameters(), lr=learning_rate)

# DataLoaders
train_loader = get_loader('train', batch_size=batch_size, patch_size=patch_size)
eval_loader = get_loader('train', batch_size=1, patch_size=patch_size)


def train():
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        # Epoch progress bar
        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
        for ts_batch, prompts, answers in epoch_bar:
            optimizer.zero_grad()
            outputs = model(prompts, ts_batch)

            # Simple token-overlap loss
            batch_loss = sum(
                0.0 if ans.lower() in out.lower() else 1.0
                for ans, out in zip(answers, outputs)
            ) / len(outputs)
            loss = torch.tensor(batch_loss, requires_grad=True, device=device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Update progress bar
            epoch_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        tqdm.write(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")

        # Evaluation after epoch
        model.eval()
        tqdm.write(f"\n--- Evaluation after Epoch {epoch} ---")
        with torch.no_grad():
            for i, (ts_batch, prompts, answers) in enumerate(eval_loader, start=1):
                if i > 5:
                    break
                output = model(prompts, ts_batch)
                tqdm.write(f"Example {i}:")
                tqdm.write(f"Prompt:  {prompts[0]}")
                tqdm.write(f"Model:   {output[0]}")
                tqdm.write(f"Answer:  {answers[0]}\n")
        model.train()

    tqdm.write("\nTraining complete.")


if __name__ == '__main__':
    train()