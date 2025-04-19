import os
from dataset_generation.normal_dist_around_mean_generation import generate_test_data
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM
from chronos import ChronosPipeline
from resnet import ResNet1Dv2  # your 1D ResNet adapter
from transformers import DataCollatorWithPadding
from tqdm import tqdm


BATCH_SIZE = 128

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
def generate_toy_data():
    # seed = 42
    # n_samples = 200
    # seq_len = 100
    # rng = np.random.RandomState(seed)
    # series = rng.randn(n_samples, seq_len).astype(np.float32)
    # descs = []
    # for ts in series:
    #     mean = np.mean(ts)
    #     e0 = ts[0]
    #     e5 = ts[5] if seq_len > 5 else ts[-1]
    #     descs.append(f"The mean of the time series is {mean:.2f}.")

    # print(series, descs)
    series, descs = generate_test_data([0])
    # series, descs = generate_test_data([i for i in range(365)])
    # print(series, descs)
    return series, descs


# -------------------------------------------------------------------
# 2) Dataset & Collation
# -------------------------------------------------------------------
class TS2TextDataset(Dataset):
    def __init__(self, series, descs, tokenizer):
        self.series = series
        self.descs = descs
        self.tok = tokenizer

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        ts = torch.from_numpy(self.series[idx])
        text = self.descs[idx]
        enc = self.tok(text, return_tensors="pt", padding=False)
        ids = enc.input_ids.squeeze(0)
        mask = enc.attention_mask.squeeze(0)
        labels = ids.clone()
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
def build_resnet(device, in_channels=1):
    return ResNet1Dv2(
        in_channels=in_channels,
        base_filters=32,
        kernel_size=3,
        stride=2,
        groups=1,
        n_block=6,
        n_classes=None,
    ).to(device)


class CustomLlama(LlamaForCausalLM):
    def generate_from_embeddings(self, inputs_embeds, max_length=50, eos_token_id=None):
        self.eval()
        generated = []
        past = None
        for step in range(max_length):
            if past is None:
                out = self(inputs_embeds=inputs_embeds, use_cache=True)
            else:
                out = self(input_ids=next_token, past_key_values=past, use_cache=True)
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
def train_epoch(resnet, proj, llama, loader, opt, device):
    llama.eval()

    # 1) Enable training on ResNet + projection head
    resnet.train()
    proj.train()

    total_loss = 0.0

    for ts, ids, mask_ids, labels in tqdm(loader, desc="Training", leave=False):
        ts = ts.to(device).float().unsqueeze(1)  # [B, 1, T]

        ids, labels = ids.to(device), labels.to(device)
        mask_ids = mask_ids.to(device)  # [B, orig_len]

        # 1) Series → ResNet → projection
        res_out = resnet(ts)  # [B, C, T']
        r = proj(res_out.permute(0, 2, 1))  # [B, T', d_model]

        # 3) Build labels & masks
        B, orig_len = ids.size()
        desc_len = orig_len
        series_len = r.size(1)
        full_len = series_len + desc_len

        labels_full = torch.full((B, full_len), -100, device=device)
        labels_full[:, series_len:] = labels

        series_mask = torch.ones(
            (B, series_len), device=device, dtype=mask_ids.dtype
        )  # [B, series_len]
        attn_mask = torch.cat([series_mask, mask_ids], dim=1)

        # 4) Embed the prompt and description tokens
        d_emb = llama.get_input_embeddings()(ids)  # [B, desc_len,   d_model]

        # 5) Concatenate in order: prompt → series embedding → description
        inputs_embeds = torch.cat([r, d_emb], dim=1)  # [B, full_len, d_model]

        # 6) Forward pass (LLM frozen, but we want gradients back to proj+ResNet)
        out = llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels_full,
        )
        loss = out.loss

        # 7) Backprop only into ResNet+proj
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(resnet, proj, llama, loader, device):
    # Freeze everything
    llama.eval()
    resnet.eval()
    proj.eval()

    total_loss = 0.0

    for ts, ids, mask_ids, labels in tqdm(loader, desc="Evaluating", leave=False):
        ts = ts.to(device).float().unsqueeze(1)
        ids, labels = ids.to(device), labels.to(device)
        mask_ids = mask_ids.to(device)

        with torch.no_grad():
            # 1) Series embedding
            res_out = resnet(ts)
            r = proj(res_out.permute(0, 2, 1))

            # 2) Labels & masks
            B, orig_len = ids.size()
            desc_len = orig_len
            series_len = r.size(1)
            full_len = series_len + desc_len

            labels_full = torch.full((B, full_len), -100, device=device)
            labels_full[:, series_len:] = labels

            # prompt_mask = mask_ids[:, :pre_timeseries_len]  # token‐pad mask for prompt
            series_mask = torch.ones(
                (B, series_len), device=device, dtype=mask_ids.dtype
            )
            attn_mask = torch.cat([series_mask, mask_ids], dim=1)

            # 3) Embed text and concat
            d_emb = llama.get_input_embeddings()(ids)
            inputs_embeds = torch.cat([r, d_emb], dim=1)

            # 4) Forward and accumulate loss
            out = llama(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                labels=labels_full,
            )
            total_loss += out.loss.item()

    return total_loss / len(loader)


def generate_examples(resnet, proj, llama, tok, series, device, n=3):
    print("\n--- Generations ---")
    for i in range(min(n, len(series))):
        # print the input time series
        print(f"Input series {i}: {series[i]} item at position 0 is {series[i][0]}")
        ts = torch.from_numpy(series[i]).unsqueeze(0).to(device).float().unsqueeze(1)
        res_out = resnet(ts)
        r = proj(res_out.permute(0, 2, 1))
        gen_ids = llama.generate_from_embeddings(
            inputs_embeds=r, max_length=50, eos_token_id=tok.eos_token_id
        )

        text = tok.decode(gen_ids[0], skip_special_tokens=True)
        print(f"Generated text: {text}\n")

    # 1) A general description of the data, before the embedding
    pre_prompt = (
        "Below is an embedding of a multivariate time series.  "
        "This series contains measurements recorded over time—do not echo the embedding itself, "
        "but use it to understand and describe the underlying data."
    )

    # 2) A decoder‑prefix after the embedding to kick off generation
    post_prompt = (
        "Question: Based on the embedding, describe in plain English what patterns or values you observe.  "
        "Be concise and focus on the main features of the series.\n"
        "Description:"
    )

    # Tokenize and embed the two halves once
    enc_pre = tok(pre_prompt, return_tensors="pt", add_special_tokens=False)
    enc_post = tok(post_prompt, return_tensors="pt", add_special_tokens=False)

    ids_pre = enc_pre.input_ids.to(device)  # [1, pre_len]
    ids_post = enc_post.input_ids.to(device)  # [1, post_len]

    p_emb = llama.get_input_embeddings()(ids_pre)  # [1, pre_len, D]
    post_emb = llama.get_input_embeddings()(ids_post)  # [1, post_len, D]

    for i in range(min(n, len(series))):
        # Print the raw series for reference
        print(f"Input series {i}: {series[i]}")

        # Compute the series embedding
        ts = torch.from_numpy(series[i]).unsqueeze(0).to(device).float().unsqueeze(1)
        res_out = resnet(ts)
        r = proj(res_out.permute(0, 2, 1))  # [1, series_len, D]

        # Concatenate: pre_prompt → series embedding → post_prompt
        inputs_embeds = torch.cat([p_emb, r, post_emb], dim=1)

        # Generate up to 50 tokens beyond the post_prompt
        gen_ids = llama.generate_from_embeddings(
            inputs_embeds=inputs_embeds,
            max_length=inputs_embeds.size(1) + 50,
            eos_token_id=tok.eos_token_id,
        )

        # Decode and strip off everything up through "Description:"
        text = tok.decode(gen_ids[0], skip_special_tokens=True)
        description = text.split("Description:")[-1].strip()

        print(f"Generated description: {description}\n")


# -------------------------------------------------------------------
# 5) Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-1B"
    series, descs = generate_toy_data()
    val_perc = 10
    first_val_index = len(series) * (100 - val_perc) // 100
    train_s, val_s = series[:first_val_index], series[first_val_index:]
    train_d, val_d = descs[:first_val_index], descs[first_val_index:]
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    tok.pad_token = tok.eos_token
    train_ds = TS2TextDataset(train_s, train_d, tok)
    val_ds = TS2TextDataset(val_s, val_d, tok)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tok.pad_token_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tok.pad_token_id),
    )
    llama = build_llama(model_id, device)
    resnet = build_resnet(device)
    res_ch = resnet.basicblock_list[-1].out_channels
    hidden = llama.config.hidden_size
    proj = nn.Linear(res_ch, hidden).to(device)
    opt = optim.Adam(proj.parameters(), lr=1e-3)
    epochs = 100
    for ep in range(1, epochs + 1):
        print(f"\nEpoch {ep}/{epochs}")
        tr_loss = train_epoch(resnet, proj, llama, train_loader, opt, device)
        val_loss = eval_epoch(resnet, proj, llama, val_loader, device)
        print(f"Epoch {ep} - train_loss: {tr_loss:.4f}, val_loss: {val_loss:.4f}")
        generate_examples(
            resnet,
            proj,
            llama,
            tok,
            val_s,
            device,
        )
