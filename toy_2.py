import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

# Utility to extract floats from model outputs (optional)
def extract_floats(text):
    return re.findall(r"(\d+\.\d+)", text)

# --- multi-sensor dataset with built-in reasoning targets ---
class MultiSensorDataset(Dataset):
    def __init__(self, num_samples=20000, seq_len=100):
        self.seq_len = seq_len
        t = np.linspace(0, 24, seq_len)  # 24h in hours
        self.X, self.y_insight = [], []
        for _ in range(num_samples):
            # Simulate modalities
            base_hr = 70 + 10 * np.sin(2 * np.pi * t / 24)
            hr = (base_hr + np.random.randn(seq_len) * 2).astype(np.float32)
            steps = (np.random.poisson(5, seq_len) * (np.random.rand(seq_len) < 0.1)).astype(np.float32)
            spo2 = (97 + np.random.randn(seq_len) * 0.5).astype(np.float32)
            sample = np.stack([hr, steps, spo2], axis=0)

            # Compute features
            avg_hr = float(hr.mean())
            peak_hr = float(hr.max())
            peak_hr_time = float(np.argmax(hr) * (24/seq_len))
            peak_step_time = float(np.argmax(steps) * (24/seq_len))
            min_spo2 = float(spo2.min())

            # Generate reasoning insight
            insight = (
                f"The heart rate peaked at {peak_hr:.0f} bpm around {peak_hr_time:.1f}h,"
                f" which coincides with a step burst at {peak_step_time:.1f}h."
                f" The average HR over 24h was {avg_hr:.1f} bpm, and SpO₂ stayed above {min_spo2:.1f}%."
            )

            self.X.append(sample.reshape(-1))
            self.y_insight.append(insight)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_insight[idx]

# --- Prefix encoder to map time-series to LLM prefix ---
class PrefixEncoder(nn.Module):
    def __init__(self, input_dim, prefix_len, hidden_size):
        super().__init__()
        self.prefix_len = prefix_len
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, prefix_len * hidden_size)
        )

    def forward(self, x):
        B = x.size(0)
        out = self.mlp(x)
        return out.view(B, self.prefix_len, -1)

# --- Training and evaluation ---
def evaluate(prefix_enc, model, tokenizer, dataset, device, prompt, max_examples=100):
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    prefix_enc.eval()
    total_bleu = []  # or any metric
    with torch.no_grad():
        count = 0
        for Xb, y_ins in loader:
            Xb = torch.tensor(Xb, device=device)
            pref = prefix_enc(Xb)
            toks = tokenizer([prompt] * len(Xb), return_tensors="pt", padding=True)
            input_ids = toks.input_ids.to(device)
            txt_emb = model.get_input_embeddings()(input_ids)
            inp_emb = torch.cat([pref, txt_emb], dim=1)
            attn = torch.cat([
                torch.ones(len(Xb), pref.size(1), device=device),
                toks.attention_mask.to(device)
            ], dim=1)
            gen = model.generate(
                inputs_embeds=inp_emb,
                attention_mask=attn,
                num_beams=3,
                max_new_tokens=50,
                early_stopping=True
            )
            dec = [tokenizer.decode(g, skip_special_tokens=True) for g in gen]
            # simple string match metric
            for pred, true in zip(dec, y_ins):
                total_bleu.append(1.0 if true.strip() == pred.strip() else 0.0)
                count += 1
                if count >= max_examples:
                    return float(np.mean(total_bleu))
    return float(np.mean(total_bleu))


def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model & tokenizer
    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(model_id)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    model.to(device)

    # hyperparameters
    SEQ_LEN    = 100
    MODALITIES = 3
    INPUT_DIM  = SEQ_LEN * MODALITIES
    PREFIX_LEN = 60
    HIDDEN_SZ  = model.config.hidden_size
    BATCH      = 8
    LR         = 2e-4
    EPOCHS     = 8
    PROMPT     = "Insight:"

    prompt_tok = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=False)
    PROMPT_LEN = prompt_tok.input_ids.size(1)

    # dataset
    full_ds = MultiSensorDataset(num_samples=20000, seq_len=SEQ_LEN)
    train_size = int(0.8 * len(full_ds))
    val_size   = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

    # prefix encoder
    prefix_enc = PrefixEncoder(INPUT_DIM, PREFIX_LEN, HIDDEN_SZ).to(device)
    optimizer = torch.optim.AdamW(prefix_enc.parameters(), lr=LR)

    # training loop
    for epoch in range(1, EPOCHS+1):
        prefix_enc.train()
        total_loss = 0.0
        for Xb, y_ins in tqdm(train_loader, desc=f"Epoch {epoch}"):
            Xb = torch.tensor(Xb, device=device)
            pref = prefix_enc(Xb)

            texts = [f"{PROMPT} {ins}" for ins in y_ins]
            toks = tokenizer(texts, return_tensors="pt", padding=True)
            ids  = toks.input_ids.to(device)
            emb  = model.get_input_embeddings()(ids)

            inp   = torch.cat([pref, emb], dim=1)
            attn  = torch.cat([
                torch.ones(len(Xb), PREFIX_LEN, device=device),
                toks.attention_mask.to(device)
            ], dim=1)

            labels = ids.clone()
            labels[:, :PROMPT_LEN] = -100
            labels = torch.cat([
                torch.full((len(Xb), PREFIX_LEN), -100, device=device),
                labels
            ], dim=1)

            out = model(inputs_embeds=inp, attention_mask=attn, labels=labels)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} avg loss: {total_loss/len(train_loader):.4f}")
        acc = evaluate(prefix_enc, model, tokenizer, val_ds, device, PROMPT)
        print(f"Validation exact-match accuracy: {acc:.3f}")

        # sample reasoning outputs
        prefix_enc.eval()
        print("--- Sample Reasoning ---")
        with torch.no_grad():
            for i in range(3):
                x, true_ins = val_ds[i]
                xb   = torch.tensor(x[None], device=device)
                pref = prefix_enc(xb)
                toks = tokenizer([PROMPT], return_tensors="pt", padding=True)
                emb = model.get_input_embeddings()(toks.input_ids.to(device))
                inp = torch.cat([pref, emb], dim=1)
                attn = torch.cat([
                    torch.ones(1, PREFIX_LEN, device=device),
                    toks.attention_mask.to(device)
                ], dim=1)
                out_ids = model.generate(
                    inputs_embeds=inp,
                    attention_mask=attn,
                    max_new_tokens=60,
                    num_beams=4,
                    early_stopping=True
                )
                pred = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                print("True Insight:", true_ins)
                print("Pred Insight:", pred, "\n")

if __name__ == "__main__":
    main()