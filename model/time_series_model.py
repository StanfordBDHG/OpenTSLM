# model/time_series_model.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class TimeSeriesEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.embedding = nn.Linear(patch_size, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len]
        B, L = x.shape
        if L % self.patch_size != 0:
            raise ValueError(f"Sequence length {L} not divisible by patch_size {self.patch_size}")
        # reshape into patches
        x = x.view(B, L // self.patch_size, self.patch_size)      # [B, num_patches, patch_size]
        x = self.embedding(x)                                     # [B, num_patches, embed_dim]
        x = self.encoder(x)                                       # [B, num_patches, embed_dim]
        return x

class TimeSeriesLLM(nn.Module):
    def __init__(self, llm_id='meta-llama/Llama-3.2-1B', embed_dim=256, device='cuda'):
        super().__init__()
        self.device = device

        # 1) tokenizer (ensure pad_token exists)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2) load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=torch.bfloat16,
            device_map={'': device}
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # 3) time-series encoder + projector
        self.encoder = TimeSeriesEncoder(embed_dim=embed_dim).to(device)
        self.projector = nn.Linear(embed_dim, self.llm.config.hidden_size).to(device)

    def forward(
        self,
        prompts: list[str],
        ts_data: torch.Tensor,
        answers: list[str] | None = None,
        max_new_tokens: int = 50
    ):
        B = len(prompts)

        # --- 1) tokenize prompts ---
        tok = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        prompt_input_ids   = tok.input_ids           # [B, P]
        prompt_attn_mask   = tok.attention_mask      # [B, P]
        prompt_embeds      = self.llm.get_input_embeddings()(prompt_input_ids)  # [B, P, H]

        # --- 2) encode & project TS ---
        ts_enc = self.encoder(ts_data.to(self.device))            # [B, N, emb]
        ts_proj = self.projector(ts_enc).to(prompt_embeds.dtype)  # [B, N, H]
        ts_attn_mask = torch.ones(B, ts_proj.size(1), device=self.device, dtype=torch.long)  # [B, N]

        # --- If no answers provided, just generate ---
        if answers is None:
            # combine embeddings
            inputs_embeds = torch.cat([prompt_embeds, ts_proj], dim=1)  # [B, P+N, H]
            attention_mask = torch.cat([prompt_attn_mask, ts_attn_mask], dim=1)  # [B, P+N]

            gen_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                temperature=0.8,
                top_p=0.9
            )
            return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # --- 3) training: tokenize answers ---
        ans_tok = self.tokenizer(
            answers,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        ans_input_ids    = ans_tok.input_ids           # [B, A]
        ans_attn_mask    = ans_tok.attention_mask      # [B, A]

        # --- 4) build full inputs_embeds and attention_mask ---
        inputs_embeds = torch.cat([prompt_embeds, ts_proj,
                                   self.llm.get_input_embeddings()(ans_input_ids)], dim=1)
        attention_mask = torch.cat([prompt_attn_mask, ts_attn_mask, ans_attn_mask], dim=1)  # [B, P+N+A]

        # --- 5) build labels with ignore_index = -100 everywhere except answer tokens ---
        # total length = P + N + A
        total_len = attention_mask.size(1)
        labels = torch.full((B, total_len), -100, dtype=torch.long, device=self.device)
        # mark answer positions with their true token IDs
        P = prompt_input_ids.size(1)
        N = ts_proj.size(1)
        labels[:, P+N : P+N+ans_input_ids.size(1)] = ans_input_ids

        # --- 6) forward through LLM, let it compute cross-entropy on the labels slice ---
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        # outputs.loss is already average over non-ignored positions
        return outputs.loss
