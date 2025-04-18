import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

def extract_first_float(text):
    m = re.search(r"(\d+\.\d+)", text)
    return float(m.group(1)) if m else None

class SineDataset(Dataset):
    def __init__(self, num_samples=5000, seq_len=100, freq=1.0):
        self.seq_len = seq_len
        t = np.linspace(0, 1, seq_len)
        self.X, self.y_vals, self.y_str = [], [], []
        for _ in range(num_samples):
            A = np.random.uniform(0.5, 2.0)
            wave = (A * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            self.X.append(wave)
            self.y_vals.append(A)
            self.y_str.append(f"{A:.2f}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_vals[idx], self.y_str[idx]

class PrefixEncoder(nn.Module):
    def __init__(self, seq_len, prefix_len, hidden_size):
        super().__init__()
        self.prefix_len = prefix_len
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.ReLU(),
            nn.Linear(512, prefix_len * hidden_size)
        )

    def forward(self, x):
        B = x.size(0)
        out = self.mlp(x)
        return out.view(B, self.prefix_len, -1)


def evaluate(prefix_enc, model, tokenizer, dataset, device, prompt, max_examples=100):
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    prefix_enc.eval()
    maes = []
    with torch.no_grad():
        count = 0
        for Xb, y_vals, _ in loader:
            Xb = torch.as_tensor(Xb, device=device)
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
                num_beams=5,
                max_new_tokens=5,
                early_stopping=True
            )
            decoded = [tokenizer.decode(g, skip_special_tokens=True) for g in gen]
            for dec, trueA in zip(decoded, y_vals):
                pred = extract_first_float(dec)
                if pred is not None:
                    maes.append(abs(pred - trueA))
                count += 1
                if count >= max_examples:
                    return float(np.mean(maes)) if maes else float('nan')
    return float(np.mean(maes)) if maes else float('nan')


def main():
    # --- device selection ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}\n")

    # --- model & tokenizer ---
    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(model_id)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    # --- hyperparams ---
    SEQ_LEN    = 100
    PREFIX_LEN = 50
    HIDDEN_SZ  = model.config.hidden_size
    BATCH      = 16
    LR         = 1e-3
    EPOCHS     = 20
    PROMPT     = "The amplitude is:"

    # --- compute prompt length ---
    prompt_tok = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=False)
    PROMPT_LEN = prompt_tok.input_ids.size(1)

    # --- dataset & split ---
    full_ds = SineDataset(num_samples=5000, seq_len=SEQ_LEN)
    train_size = int(len(full_ds) * 0.8)
    val_size   = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

    # --- prefix encoder & optimizer ---
    prefix_encoder = PrefixEncoder(SEQ_LEN, PREFIX_LEN, HIDDEN_SZ).to(device)
    optimizer = torch.optim.AdamW(prefix_encoder.parameters(), lr=LR)

    # --- training loop ---
    for epoch in range(1, EPOCHS + 1):
        prefix_encoder.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for Xb, _, y_str in loop:
            Xb = torch.as_tensor(Xb, device=device)
            pref = prefix_encoder(Xb)

            # prepare texts and embeddings
            texts = [f"{PROMPT} {s}" for s in y_str]
            toks  = tokenizer(texts, return_tensors="pt", padding=True)
            ids   = toks.input_ids.to(device)
            emb   = model.get_input_embeddings()(ids)

            inp_emb = torch.cat([pref, emb], dim=1)
            attn    = torch.cat([
                torch.ones(len(Xb), PREFIX_LEN, device=device),
                toks.attention_mask.to(device)
            ], dim=1)

            # labels: only score amplitude tokens
            labels = ids.clone()
            labels[:, :PROMPT_LEN] = -100
            full_labels = torch.cat([
                torch.full((len(Xb), PREFIX_LEN), -100, device=device),
                labels
            ], dim=1)

            out = model(inputs_embeds=inp_emb,
                        attention_mask=attn,
                        labels=full_labels)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        train_mae = evaluate(prefix_encoder, model, tokenizer, train_ds, device, PROMPT, max_examples=200)
        val_mae   = evaluate(prefix_encoder, model, tokenizer, val_ds,   device, PROMPT, max_examples=200)

        print(f"\nEpoch {epoch} → avg loss: {avg_loss:.4f} | "
              f"train MAE: {train_mae:.3f} | val MAE: {val_mae:.3f}\n")

        # --- demo on 5 held-out examples ---
        print("Demo on 5 validation examples:")
        demo_idxs = np.random.choice(len(val_ds), size=5, replace=False)
        for idx in demo_idxs:
            X_demo, y_val_demo, _ = val_ds[idx]
            # print the full time series
            print("Time series:", X_demo.tolist())
            X_demo_tensor = torch.as_tensor(X_demo[None, :], device=device)
            pref_d = prefix_encoder(X_demo_tensor)
            toks_d = tokenizer([PROMPT], return_tensors="pt", padding=True)
            emb_d = model.get_input_embeddings()(toks_d.input_ids.to(device))
            inp_d = torch.cat([pref_d, emb_d], dim=1)
            attn_d = torch.cat([
                torch.ones(1, PREFIX_LEN, device=device),
                toks_d.attention_mask.to(device)
            ], dim=1)
            out_ids = model.generate(
                inputs_embeds=inp_d,
                attention_mask=attn_d,
                num_beams=5,
                max_new_tokens=5,
                early_stopping=True
            )
            out_txt = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            pred_val = extract_first_float(out_txt)
            print(f" True: {y_val_demo:.2f} | Pred: {pred_val:.2f} | Completion: “{out_txt}”")

    # end training loop

if __name__ == "__main__":
    main()
