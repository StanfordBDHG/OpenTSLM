import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM
from chronos import ChronosPipeline
from resnet import ResNet1Dv2  # your 1D ResNet adapter
from tqdm import tqdm
# -------------------------------------------------------------------
# 0) Device Configuration (with MPS support on macOS)
# -------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------
# 1) Toy Data Generation
# -------------------------------------------------------------------
def generate_toy_data(n_samples=500, seq_len=10, seed=0):
    rng = np.random.RandomState(seed)
    series = rng.randn(n_samples, seq_len).astype(np.float32)
    descs = []
    for ts in series:
        median = np.median(ts)
        e0 = ts[0]
        e5 = ts[5] if seq_len > 5 else ts[-1]
        descs.append(
            f"The median of the time series is {median:.2f}. "
            f"Element at index 0 is {e0:.2f}. "
            f"Element at index 5 is {e5:.2f}."
        )
    return series, descs

# -------------------------------------------------------------------
# 2) Dataset & Collation
# -------------------------------------------------------------------
class TS2TextDataset(Dataset):
    def __init__(self, series, descs, tokenizer, prompt="Describe the time series:"):
        self.series = series
        self.descs = descs
        self.tok = tokenizer
        self.prompt = prompt
        self.prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        self.prompt_len = len(self.prompt_ids)

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        ts = torch.from_numpy(self.series[idx])
        text = self.prompt + " " + self.descs[idx]
        enc = self.tok(text, return_tensors="pt", padding=False)
        ids = enc.input_ids.squeeze(0)
        mask = enc.attention_mask.squeeze(0)
        labels = ids.clone()
        labels[: self.prompt_len] = -100
        return ts, ids, mask, labels


def collate_fn(batch, pad_id):
    ts, ids, masks, labels = zip(*batch)
    ts = torch.stack(ts, 0)
    max_len = max(x.size(0) for x in ids)
    def pad(x, val):
        p = torch.full((max_len - x.size(0),), val, dtype=x.dtype, device=x.device)
        return torch.cat([x, p], 0)
    ids_p = torch.stack([pad(x, pad_id) for x in ids])
    masks_p = torch.stack([pad(x, 0) for x in masks])
    labels_p = torch.stack([pad(x, -100) for x in labels])
    return ts, ids_p, masks_p, labels_p

# -------------------------------------------------------------------
# 3) Model Builders
# -------------------------------------------------------------------
def build_chronos():
    chronos = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        torch_dtype=torch.float32
    )
    for p in chronos.model.parameters():
        p.requires_grad = False
    return chronos


def build_resnet(device):
    return ResNet1Dv2(
        in_channels=768,
        base_filters=32,
        kernel_size=3,
        stride=2,
        groups=1,
        n_block=6,
        n_classes=None
    ).to(device)

class CustomLlama(LlamaForCausalLM):
    def generate_from_embeddings(self, inputs_embeds, max_length=50, eos_token_id=None):
        self.eval()
        generated = []
        past = None
        for step in range(max_length):
            if past is None:
                out = self(
                    inputs_embeds=inputs_embeds,
                    use_cache=True
                )
            else:
                out = self(
                    input_ids=next_token,
                    past_key_values=past,
                    use_cache=True
                )
            logits = out.logits
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            generated.append(next_token)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            past = out.past_key_values
        return torch.cat(generated, dim=1)


def build_llama(model_id, device):
    llama = CustomLlama.from_pretrained(model_id)
    llama.to(device)
    for p in llama.parameters():
        p.requires_grad = False
    return llama

# -------------------------------------------------------------------
# 4) Training, Evaluation, and Generation
# -------------------------------------------------------------------
def train_epoch(ch, resnet, proj, llama, loader, opt, device):
    resnet.train()
    total_loss = 0.0
    prompt_len = loader.dataset.prompt_len
    for ts, ids, mask, labels in tqdm(loader, desc="Training", leave=False):
        ts_cpu = ts.cpu()
        ids, labels = ids.to(device), labels.to(device)
        with torch.no_grad():
            emb_cpu = ch.embed(ts_cpu)[0]
        emb = emb_cpu.permute(0,2,1).to(device)
        r = proj(resnet(emb).permute(0,2,1))
        B, orig_len = ids.size()
        desc_len = orig_len - prompt_len
        series_len = r.size(1)
        full_len = prompt_len + series_len + desc_len
        labels_full = torch.full((B, full_len), -100, device=device)
        labels_full[:, prompt_len+series_len:] = labels[:, prompt_len:]
        attn_mask = torch.ones((B, full_len), device=device)
        p_emb = llama.get_input_embeddings()(ids[:, :prompt_len])
        d_emb = llama.get_input_embeddings()(ids[:, prompt_len:])
        inputs_embeds = torch.cat([p_emb, r, d_emb], dim=1)
        out = llama(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels_full)
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss/len(loader)


def eval_epoch(ch, resnet, proj, llama, loader, device):
    resnet.eval()
    total_loss = 0.0
    prompt_len = loader.dataset.prompt_len
    with torch.no_grad():
        for ts, ids, mask, labels in tqdm(loader, desc="Evaluating", leave=False):
            ts_cpu = ts.cpu()
            ids, labels = ids.to(device), labels.to(device)
            emb_cpu = ch.embed(ts_cpu)[0]
            emb = emb_cpu.permute(0,2,1).to(device)
            r = proj(resnet(emb).permute(0,2,1))
            B, orig_len = ids.size()
            desc_len = orig_len - prompt_len
            series_len = r.size(1)
            full_len = prompt_len + series_len + desc_len
            labels_full = torch.full((B, full_len), -100, device=device)
            labels_full[:, prompt_len+series_len:] = labels[:, prompt_len:]
            attn_mask = torch.ones((B, full_len), device=device)
            p_emb = llama.get_input_embeddings()(ids[:, :prompt_len])
            d_emb = llama.get_input_embeddings()(ids[:, prompt_len:])
            inputs_embeds = torch.cat([p_emb, r, d_emb], dim=1)
            out = llama(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels_full)
            total_loss += out.loss.item()
    return total_loss/len(loader)


def generate_examples(ch, resnet, proj, llama, tok, series, prompt, device, n=3):
    print("\n--- Generations ---")
    prompt_ids = tok(prompt, return_tensors="pt").input_ids
    p_emb = llama.get_input_embeddings()(prompt_ids.to(device))
    for i in range(min(n, len(series))):
        # print the input time series
        print(f"Input series {i}: {series[i]}")
        ts = torch.from_numpy(series[i]).unsqueeze(0)
        emb_cpu = ch.embed(ts)[0]
        emb = emb_cpu.permute(0,2,1).to(device)
        r = proj(resnet(emb).permute(0,2,1))
        inp_embeds = torch.cat([p_emb, r], dim=1)
        gen_ids = llama.generate_from_embeddings(inputs_embeds=inp_embeds, max_length=50, eos_token_id=tok.eos_token_id)
        text = tok.decode(gen_ids[0], skip_special_tokens=True)
        print(f"Generated text: {text}\n")

# -------------------------------------------------------------------
# 5) Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-1B"
    series, descs = generate_toy_data(800, seq_len=10)
    train_s, val_s = series[:700], series[700:]
    train_d, val_d = descs[:700], descs[700:]
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    tok.pad_token = tok.eos_token
    train_ds = TS2TextDataset(train_s, train_d, tok)
    val_ds   = TS2TextDataset(val_s, val_d, tok)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=lambda b: collate_fn(b, tok.pad_token_id))
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, collate_fn=lambda b: collate_fn(b, tok.pad_token_id))
    chronos = build_chronos()
    llama   = build_llama(model_id, device)
    resnet  = build_resnet(device)
    res_ch  = resnet.basicblock_list[-1].out_channels
    hidden  = llama.config.hidden_size
    proj    = nn.Linear(res_ch, hidden).to(device)
    opt     = optim.Adam(proj.parameters(), lr=1e-3)
    epochs = 3
    prompt = train_ds.prompt
    for ep in range(1, epochs+1):
        print(f"\nEpoch {ep}/{epochs}")
        tr_loss = train_epoch(chronos, resnet, proj, llama, train_loader, opt, device)
        val_loss= eval_epoch(chronos, resnet, proj, llama, val_loader, device)
        print(f"Epoch {ep} - train_loss: {tr_loss:.4f}, val_loss: {val_loss:.4f}")
        generate_examples(chronos, resnet, proj, llama, tok, val_s, prompt, device)