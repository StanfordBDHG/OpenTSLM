import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence

from model_config import ENCODER_OUTPUT_DIM
from model.llm.TimeSeriesLLM import TimeSeriesLLM
from model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from model.projector.MLPProjector import MLPProjector
from prompt.full_prompt import FullPrompt
from time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
 
class EmbedHealthSP(TimeSeriesLLM):
    def __init__(
        self,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        device: str = "cuda",
    ):
        super().__init__(device)

        # 1) tokenizer (ensure pad_token exists)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2) load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # 3) encoder + projector (now internal)
        self.encoder = TransformerCNNEncoder().to(device)
        self.projector = MLPProjector(
            ENCODER_OUTPUT_DIM, self.llm.config.hidden_size, device=device
        ).to(device)

        self.patch_size = 4

        # Freeze the LLM backbone for SP model (internally)
        for p in self.llm.parameters():
            p.requires_grad = False

    def pad_and_apply_batch(
        self,
        batch: List[Dict[str, any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TL;DR:
            This function is probably the most crucial part of EmbedHealth-SP, and also the hardest to understand.
            It's where the magic happens and legends are made.

            It batches and embeds all text and time series inputs in parallel,
            then reassembles them per sample to allow efficient GPU execution.
            Praise the PyTorch Wizards: ChatGPT-o4-mini-high, Patrick, and Thomas (listed in strictly descending order of skill).

        Long description:
            Processes a batch of training samples by embedding and aligning text and time series data
            for efficient parallel processing on the GPU.

            This method performs the following steps:

            1. Extracts all text components (pre_prompt, time_series_text, post_prompt) from each sample,
            and embeds them in a single batch using the LLM tokenizer and embedding layer. Padding and attention
            masks are applied to accommodate variable-length sequences.

            2. Gathers all time series segments across the batch and pads them
            into a single tensor of shape [N_ts_total, T_padded, D], where T_padded
            is the smallest multiple of `patch_size` ≥ the longest segment length.
            This tensor is then encoded and projected into the LLM hidden space.

            3. After all embeddings are extracted, the function reconstructs each original sample by interleaving its
            embedded pre_prompt, time series texts and corresponding time series embeddings, and the post_prompt, preserving original order.

            4. Pads all reassembled sequences to a uniform length across the batch to form the final input tensor
                and attention mask.

            5. All of this is only required for efficient processing.

        - pre_prompt: str
        - time_series_text: List[str]
        - time_series: Tensor [N_ts, T] or [N_ts, T, D]
        - post_prompt: str
        Returns (inputs_embeds, attention_mask)
        """
        device = self.device
        H = self.llm.config.hidden_size

        # 1) Gather all texts
        all_texts: List[str] = []
        text_ptrs: List[Tuple[int, int]] = []
        ts_counts: List[int] = []
        for sample in batch:
            start = len(all_texts)
            all_texts.append(sample["pre_prompt"])
            all_texts.extend(sample["time_series_text"])
            all_texts.append(sample["post_prompt"])
            end = len(all_texts)
            text_ptrs.append((start, end))
            ts_counts.append(len(sample["time_series_text"]))

        # 2) Tokenize & embed all texts
        tok = self.tokenizer(
            all_texts, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = tok.input_ids.to(device, non_blocking=True)
        attn_mask = tok.attention_mask.to(device, non_blocking=True)
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [N_all, P_max, H]

        # 3) Batch time-series encode & project
        ts_list: List[torch.Tensor] = []
        for sample in batch:
            for ts in sample["time_series"]:
                # ensure [T] → [T,1]
                if ts.dim() == 1:
                    ts = ts.unsqueeze(-1)
                ts_list.append(ts)

        if ts_list:
            ts_padded = pad_sequence(ts_list, batch_first=True).to(
                device, non_blocking=True
            )
            # ── pad time dim to multiple of patch_size ──
            T_max = ts_padded.size(1)
            rem = T_max % self.patch_size
            if rem:
                pad_len = self.patch_size - rem
                pad = ts_padded.new_zeros(ts_padded.size(0), pad_len, ts_padded.size(2))
                ts_padded = torch.cat([ts_padded, pad], dim=1)
            # ── now ts_padded: [N_ts_total, T_padded, 1]

            # ── key fix: squeeze out the feature dim so encoder sees [B, L] ──
            ts_enc = self.encoder(
                ts_padded.squeeze(-1)
            )  # [N_ts_total, N_patches, embed_dim]
            ts_proj = self.projector(ts_enc).to(
                text_embeds.dtype
            )  # [N_ts_total, N_patches, H]
        else:
            ts_proj = torch.empty(0, 0, H, device=device, dtype=text_embeds.dtype)

        # 4) Re‐assemble per sample
        all_seq_embeds, all_seq_masks = [], []
        ts_offset = 0
        for (start, end), n_ts in zip(text_ptrs, ts_counts):
            sample_embeds = text_embeds[start:end]  # [1+N_ts+1, P_max, H]
            sample_masks = attn_mask[start:end]  # [1+N_ts+1, P_max]
            seq_embeds, seq_masks = [], []

            # pre_prompt
            length = sample_masks[0].sum().item()
            seq_embeds.append(sample_embeds[0, :length, :])
            seq_masks.append(sample_masks[0, :length])

            # each (textᵢ, tsᵢ)
            for i in range(n_ts):
                idx = 1 + i
                length = sample_masks[idx].sum().item()
                seq_embeds.append(sample_embeds[idx, :length, :])
                seq_masks.append(sample_masks[idx, :length])

                proj = ts_proj[ts_offset + i]  # [N_patches, H]
                seq_embeds.append(proj)
                seq_masks.append(
                    torch.ones(proj.size(0), device=device, dtype=torch.long)
                )

            ts_offset += n_ts

            # post_prompt (fixed)
            length = sample_masks[-1].sum().item()
            seq_embeds.append(sample_embeds[-1, :length, :])
            seq_masks.append(sample_masks[-1, :length])

            all_seq_embeds.append(torch.cat(seq_embeds, dim=0))
            all_seq_masks.append(torch.cat(seq_masks, dim=0))

        # 5) Batch-pad the final sequences
        inputs_embeds = pad_sequence(all_seq_embeds, batch_first=True)  # [B, L_max, H]
        attention_mask = pad_sequence(all_seq_masks, batch_first=True)  # [B, L_max]

        return inputs_embeds, attention_mask

    def generate(
        self, batch: List[Dict[str, any]], max_new_tokens: int = 50, **generate_kwargs
    ) -> List[str]:
        inputs_embeds, attention_mask = self.pad_and_apply_batch(batch)
        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    def compute_loss(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        batch: same format as generate()
        answers: List[str] of length B
        """
        answers = [b["answer"] for b in batch]

        inputs_embeds, attention_mask = self.pad_and_apply_batch(batch)
        B, L, H = inputs_embeds.size()

        # tokenize answers
        ans_tok = self.tokenizer(
            answers, return_tensors="pt", padding=True, truncation=True
        )
        ans_ids = ans_tok.input_ids.to(self.device, non_blocking=True)
        ans_mask = ans_tok.attention_mask.to(self.device, non_blocking=True)
        ans_emb = self.llm.get_input_embeddings()(ans_ids)  # [B, A_max, H]

        # append
        inputs_embeds = torch.cat([inputs_embeds, ans_emb], dim=1)  # [B, L+A, H]
        attention_mask = torch.cat([attention_mask, ans_mask], dim=1)  # [B, L+A]

        # labels: only on the answer tokens
        total_len = attention_mask.size(1)
        labels = torch.full((B, total_len), -100, device=self.device, dtype=torch.long)
        labels[:, L:] = ans_ids

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs.loss

    def get_eos_token(self) -> str:
        return self.tokenizer.eos_token

    def store_to_file(self, path: str):
        torch.save(
            {
                "encoder_state": self.encoder.state_dict(),
                "projector_state": self.projector.state_dict(),
            },
            path,
        )

    def load_from_file(self, path: str):
        ckpt = torch.load("best_encoder.pt", map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder_state"])
        self.projector.load_state_dict(ckpt["projector_state"])
        print(f"Loaded best model from epoch {ckpt.get('epoch', '?')}")

    def eval_prompt(self, prompt: FullPrompt, max_new_tokens: int = 30000) -> str:
        """
        Evaluate a prompt and return the generated text.
        """
                
        batch = [prompt.to_dict()]
        self.eval()
        batch = extend_time_series_to_match_patch_size_and_aggregate(batch)
        output = self.generate(batch, max_new_tokens=max_new_tokens)
        return output[0]