# File: /Users/planger/Development/EmbedHealth/model/time_series_model.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class TimeSeriesEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.embedding = nn.Linear(patch_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len]
        Returns:
            Tensor of shape [batch_size, num_patches, embed_dim]
        """
        B, L = x.shape
        if L % self.patch_size != 0:
            raise ValueError(f"Sequence length ({L}) must be divisible by patch_size ({self.patch_size})")
        # Reshape into non-overlapping patches
        x = x.view(B, L // self.patch_size, self.patch_size)
        # Embed patches
        x = self.embedding(x)
        # Encode with Transformer
        x = self.encoder(x)
        return x

class TimeSeriesLLM(nn.Module):
    def __init__(
        self,
        llm_id: str = 'meta-llama/Llama-3.2-1B',
        embed_dim: int = 256,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        # Load tokenizer and ensure padding token exists
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=torch.bfloat16,
            device_map={'': self.device}
        )
        # If new tokens added, resize embeddings
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # Time-series encoder and projection, move to device
        self.encoder = TimeSeriesEncoder(embed_dim=embed_dim).to(self.device)
        self.projector = nn.Linear(embed_dim, self.llm.config.hidden_size).to(self.device)

    def forward(self, prompts: list[str], ts_data: torch.Tensor) -> list[str]:
        """
        Args:
            prompts: List of text prompts
            ts_data: Tensor [batch_size, seq_len]
        Returns:
            Generated text responses
        """
        # Tokenize prompts with padding/truncation
        tokenized = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)

        # Get text embeddings (bfloat16)
        text_embeds = self.llm.get_input_embeddings()(tokenized.input_ids)

        # Encode time-series (float32)
        ts_encoded = self.encoder(ts_data.to(self.device))  # [B, np, embed_dim]
        ts_projected = self.projector(ts_encoded)  # [B, np, hidden_size]
        # Cast to match text embedding dtype
        ts_projected = ts_projected.to(text_embeds.dtype)

        # Combine embeddings along sequence dimension
        combined_embeds = torch.cat([text_embeds, ts_projected], dim=1)
        # Create attention mask for time-series tokens
        ts_mask = torch.ones(ts_projected.size(0), ts_projected.size(1), device=self.device, dtype=torch.long)
        attention_mask = torch.cat([tokenized.attention_mask, ts_mask], dim=1)

        # Generate autoregressively
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=50
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

# EOF