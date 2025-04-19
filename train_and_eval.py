import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm

from dataset_generation.normal_dist_around_mean_generation import generate_test_data
from resnet import ResNet1Dv2  # your 1D ResNet adapter

BATCH_SIZE = 128
AUTO_EPOCHS = 10
LLM_EPOCHS = 90
LR_AE = 1e-3
LR_PROJ = 1e-4


# -------------------------------------------------------------------
# 0) Device Config
# -------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------------
# 1) Toy Data
# -------------------------------------------------------------------
def generate_toy_data():
    series, descs = generate_test_data([0])
    return series.astype(np.float32), descs


series, descs = generate_toy_data()
seq_len = series.shape[1]
n_samples = len(series)

# split train / val
val_split = int(n_samples * 0.1)
ae_train_series = series[val_split:]
ae_val_series = series[:val_split]
lm_train_series = series[val_split:]
lm_val_series = series[:val_split]
lm_train_descs = descs[val_split:]
lm_val_descs = descs[:val_split]


# -------------------------------------------------------------------
# 2) Datasets
# -------------------------------------------------------------------
class TSOnlyDataset(Dataset):
    def __init__(self, series):
        self.series = series

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        # returns a 1D tensor of length seq_len
        return torch.from_numpy(self.series[idx])


class TS2TextDataset(Dataset):
    def __init__(self, series, descs, tokenizer):
        self.series = series
        self.descs = descs
        self.tok = tokenizer
        self.prompt = ""  # no pre‑prompt for LLM phase
        self.prompt_len = 0

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        ts = torch.from_numpy(self.series[idx])
        text = self.descs[idx]
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
# 3) Models
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


class Decoder(nn.Module):
    """Simple MLP decoder: from global-pooled features back to raw series."""

    def __init__(self, in_dim, seq_len, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, seq_len)
        )

    def forward(self, x):
        return self.net(x)  # [B, seq_len]


class ConvTransposeDecoder1D(nn.Module):
    """
    ConvTranspose1d version of your MLP decoder:
      in_dim   → hidden_dim → seq_len
    but using convs so it can learn local structure if you ever expand it.
    """

    def __init__(self, in_dim, seq_len, hidden_dim=128):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            # 1) lift from [B, in_dim] → [B, hidden_dim, 1]
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            # 2) upsample from length 1 → length seq_len
            nn.ConvTranspose1d(
                hidden_dim,  # in channels
                1,  # out channels (we want a single‐channel series)
                kernel_size=seq_len,
                stride=1,
                padding=0,
                output_padding=0,
            ),
        )

    def forward(self, x):
        # x: [B, in_dim]
        x = x.unsqueeze(-1)  # [B, in_dim, 1]
        out = self.net(x)  # [B, 1, seq_len]
        return out.squeeze(1)  # [B, seq_len]


class ConvTransposeMLPDecoder1D(nn.Module):
    """
    ConvTranspose analogue of your MLP:
      B×in_dim  → B×hidden_dim  → B×seq_len
    """

    def __init__(self, in_dim, seq_len, hidden_dim=128):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            # 1) channel‑wise linear: [B,in_dim]→[B,hidden_dim] but as conv
            nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1),
            nn.ReLU(),
            # 2) “upsample” from length‑1 → length‑seq_len in one shot
            nn.ConvTranspose1d(
                in_channels=hidden_dim,
                out_channels=1,
                kernel_size=seq_len,
                stride=1,
                padding=0,
                output_padding=0,
            ),
        )

    def forward(self, x):
        # x: [B, in_dim]
        x = x.unsqueeze(-1)  # [B,in_dim,1]
        out = self.net(x)  # [B,1,seq_len]
        return out.squeeze(1)  # [B,seq_len]


class HierarchicalConvDecoder1D(nn.Module):
    def __init__(
        self, encoder_channels, seq_len, base_filters=32, kernel_size=3, num_layers=6
    ):
        super().__init__()
        layers = []
        in_ch = encoder_channels
        # Each layer upsamples by 2 and halves channels (until final → 1)
        for i in range(num_layers):
            out_ch = (
                base_filters * (2 ** (num_layers - i - 1)) if i < num_layers - 1 else 1
            )
            layers.append(
                nn.ConvTranspose1d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    output_padding=1,
                )
            )
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.ReLU())
            in_ch = out_ch

        self.net = nn.Sequential(*layers)
        self.seq_len = seq_len

    def forward(self, x):
        # x: [B, encoder_channels, T']  (T' ≈ seq_len / 2^num_layers)
        out = self.net(x)  # [B, 1, ≥ seq_len]
        # center‑crop if needed
        if out.size(2) != self.seq_len:
            diff = out.size(2) - self.seq_len
            start = diff // 2
            out = out[:, :, start : start + self.seq_len]
        return out.squeeze(1)  # [B, seq_len]


class HalvingConvTransposeDecoder1D(nn.Module):
    """
    A ConvTranspose1d decoder that
    - Takes encoder_channels → halves each layer → ends at 1
    - Doubles the time‐axis each layer → reaches seq_len (± cropping)
    """

    def __init__(self, encoder_channels, seq_len, kernel_size=3, num_layers=6):
        super().__init__()
        layers = []
        in_ch = encoder_channels

        for i in range(num_layers):
            # each layer: halve channels, double time
            out_ch = max(in_ch // 2, 1)
            layers.append(
                nn.ConvTranspose1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    output_padding=1,
                )
            )
            if i < num_layers - 1:
                layers.append(nn.ReLU())
            in_ch = out_ch

        self.net = nn.Sequential(*layers)
        self.seq_len = seq_len

    def forward(self, x):
        # x: [B, encoder_channels, T']
        out = self.net(x)  # [B, 1, ≥ seq_len]
        # center‐crop any overshoot
        if out.size(2) != self.seq_len:
            diff = out.size(2) - self.seq_len
            start = diff // 2
            out = out[:, :, start : start + self.seq_len]
        return out.squeeze(1)  # [B, seq_len]


class CustomLlama(LlamaForCausalLM):
    def generate_from_embeddings(self, inputs_embeds, max_length, eos_token_id):
        self.eval()
        generated = []
        past = None
        for _ in range(max_length):
            if past is None:
                out = self(inputs_embeds=inputs_embeds, use_cache=True)
            else:
                out = self(input_ids=next_token, past_key_values=past, use_cache=True)
            logits = out.logits
            next_token = logits[:, -1:].argmax(-1)
            generated.append(next_token)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            past = out.past_key_values
        return torch.cat(generated, dim=1)


def build_llama(model_id, device):
    llama = CustomLlama.from_pretrained(model_id).to(device)
    for p in llama.parameters():
        p.requires_grad = False
    return llama


# -------------------------------------------------------------------
# 4a) Autoencoder Training
# -------------------------------------------------------------------
def train_autoencoder_epoch(resnet, decoder, loader, opt, device):
    resnet.train()
    decoder.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for ts in tqdm(loader, desc="AE Train", leave=False):
        ts = ts.to(device).unsqueeze(1).float()  # [B,1,seq_len]
        res_out = resnet(ts)  # [B, C, T']
        recon = decoder(res_out)  # [B, seq_len]
        loss = criterion(recon, ts.squeeze(1))  # [B, seq_len]
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_autoencoder_epoch(resnet, decoder, loader, device):
    resnet.eval()
    decoder.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for ts in tqdm(loader, desc="AE  Eval", leave=False):
            ts = ts.to(device).unsqueeze(1).float()
            res_out = resnet(ts)
            recon = decoder(res_out)
            loss = criterion(recon, ts.squeeze(1))
            total_loss += loss.item()
    return total_loss / len(loader)


# -------------------------------------------------------------------
# 4b) LLM‑Alignment Training (fixed concatenation)
# -------------------------------------------------------------------
def train_epoch(resnet, proj, llama, loader, opt, device):
    llama.eval()  # frozen
    resnet.eval()  # frozen
    proj.train()  # only proj

    total_loss = 0.0

    for ts, ids, mask_ids, labels in tqdm(loader, desc="LLM Train", leave=False):
        # 1) Series → encoder
        ts = ts.to(device).unsqueeze(1).float()  # [B,1,seq_len]
        with torch.no_grad():
            res_out = resnet(ts)  # [B, C, T']
        # Global‐pool or however you want to reduce to one vector
        pooled = res_out.mean(dim=2)  # [B, C]
        r = proj(pooled)  # [B, hidden_size]
        r_seq = r.unsqueeze(1)  # [B, 1, hidden_size]

        # 2) Embed the text tokens
        ids = ids.to(device)  # [B, desc_len]
        mask_ids = mask_ids.to(device)  # [B, desc_len]
        labels = labels.to(device)  # [B, desc_len]
        d_emb = llama.get_input_embeddings()(ids)  # [B, desc_len, hidden_size]

        # 3) Concatenate **along seq‐dim** (dim=1)
        inputs_embeds = torch.cat([r_seq, d_emb], dim=1)  # [B, 1+desc_len, hidden_size]

        # 4) Build attention mask (also along seq‐dim)
        series_mask = torch.ones(
            (ids.size(0), 1), device=device, dtype=mask_ids.dtype
        )  # [B,1]
        attn_mask = torch.cat([series_mask, mask_ids], dim=1)  # [B, 1+desc_len]

        # 5) Shift labels over by one (if you want to predict the description tokens
        #    *after* the r_seq token, you may need to pad labels with -100 in front)
        labels_full = torch.cat(
            [torch.full((ids.size(0), 1), -100, device=device), labels], dim=1
        )

        # 6) Forward through the frozen LLM
        out = llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels_full,
        )
        loss = out.loss

        # 7) Backprop only into proj
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(resnet, proj, llama, loader, device):
    llama.eval()
    resnet.eval()
    proj.eval()

    total_loss = 0.0

    with torch.no_grad():
        for ts, ids, mask_ids, labels in tqdm(loader, desc="LLM  Eval", leave=False):
            # 1) Series encoding
            ts = ts.to(device).unsqueeze(1).float()
            res_out = resnet(ts)
            pooled = res_out.mean(dim=2)
            r = proj(pooled)  # [B, hidden_size]
            r_seq = r.unsqueeze(1)  # [B,1,hidden_size]

            # 2) Embed text
            ids = ids.to(device)
            mask_ids = mask_ids.to(device)
            labels = labels.to(device)
            d_emb = llama.get_input_embeddings()(ids)  # [B, desc_len, hidden_size]

            # 3) Concat on seq‐dim
            inputs_embeds = torch.cat([r_seq, d_emb], dim=1)
            series_mask = torch.ones(
                (ids.size(0), 1), device=device, dtype=mask_ids.dtype
            )
            attn_mask = torch.cat([series_mask, mask_ids], dim=1)

            # 4) Build labels_full
            labels_full = torch.cat(
                [torch.full((ids.size(0), 1), -100, device=device), labels], dim=1
            )

            # 5) Forward & accumulate loss
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
# 5) Main: two‑stage training
# -------------------------------------------------------------------
if __name__ == "__main__":
    # 5.1) Build encoder + decoder for autoencoder
    resnet = build_resnet(device)
    # res_ch = resnet.basicblock_list[-1].out_channels
    with torch.no_grad():
        dummy = torch.zeros(1, 1, seq_len, device=device)
        test_out = resnet(dummy)  # [1, 128, 64]
        encoder_channels = test_out.shape[1]  # <-- THIS must be 128, not shape[2]
    print(
        f"ResNet actually outputs {encoder_channels} channels (T'={test_out.shape[2]})"
    )

    # decoder = Decoder(in_dim=res_ch, seq_len=seq_len).to(device)
    # decoder = ConvTransposeDecoder1D(in_dim=res_ch, seq_len=seq_len, hidden_dim=128).to(
    #     device
    # )
    # encoder_ch = test_out.shape[1]  # e.g. 128
    # decoder = HierarchicalConvDecoder1D(
    #     encoder_channels=res_ch,
    #     seq_len=seq_len,
    #     base_filters=32,
    #     kernel_size=3,
    #     num_layers=6,
    # ).to(device)
    decoder = HalvingConvTransposeDecoder1D(
        encoder_channels=encoder_channels,  # e.g. 128
        seq_len=seq_len,  # your original time‑series length, e.g. 100+
        # base_filters=32,  # must match the “base_filters” you used in the ResNet
        kernel_size=3,  # same conv kernel you used in the encoder
        num_layers=6,  # same as n_block in your ResNet
    ).to(device)

    first_layer = decoder.net[0]
    print(
        "Decoder first layer expects:",
        first_layer.in_channels,
        "→",
        first_layer.out_channels,
    )

    ae_train_loader = DataLoader(
        TSOnlyDataset(ae_train_series), batch_size=BATCH_SIZE, shuffle=True
    )
    ae_val_loader = DataLoader(
        TSOnlyDataset(ae_val_series), batch_size=BATCH_SIZE, shuffle=False
    )

    ae_opt = optim.Adam(
        list(resnet.parameters()) + list(decoder.parameters()), lr=LR_AE
    )

    # -----  Stage 1: autoencoder -----
    for ep in range(1, AUTO_EPOCHS + 1):
        tr_loss = train_autoencoder_epoch(
            resnet, decoder, ae_train_loader, ae_opt, device
        )
        va_loss = eval_autoencoder_epoch(resnet, decoder, ae_val_loader, device)
        print(
            f"[AE] Epoch {ep}/{AUTO_EPOCHS}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}"
        )

    # 5.2) Build tokenizer, LLM, projection head for second stage
    model_id = "meta-llama/Llama-3.2-1B"
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    tok.pad_token = tok.eos_token

    llama = build_llama(model_id, device)
    for p in resnet.parameters():
        p.requires_grad = False
    for p in llama.parameters():
        p.requires_grad = False
    resnet.eval()

    proj = nn.Linear(encoder_channels, llama.config.hidden_size).to(device)
    lm_opt = optim.Adam(proj.parameters(), lr=LR_PROJ)

    # prepare LLM datasets
    train_ds = TS2TextDataset(lm_train_series, lm_train_descs, tok)
    val_ds = TS2TextDataset(lm_val_series, lm_val_descs, tok)
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

    # ----- Stage 2: LLM‑alignment -----
    for ep in range(1, LLM_EPOCHS + 1):
        tr_loss = train_epoch(resnet, proj, llama, train_loader, lm_opt, device)
        va_loss = eval_epoch(resnet, proj, llama, val_loader, device)
        print(
            f"[LLM] Epoch {ep}/{LLM_EPOCHS}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}"
        )

        # 5.3) (Optional) generate_examples(...)
        generate_examples(
            resnet,
            proj,
            llama,
            tok,
            lm_val_series,
            device,
        )
