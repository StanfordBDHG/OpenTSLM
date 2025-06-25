import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence

from src.model_config import ENCODER_OUTPUT_DIM

class TimeSeriesLLM(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

    
    def generate(
        self, batch: List[Dict[str, any]], max_new_tokens: int = 50, **generate_kwargs
    ) -> List[str]:
        
        raise NotImplementedError("Generate method should be implemented by the subclass")

    def compute_loss(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        batch: same format as generate()
        answers: List[str] of length B
        """
        raise NotImplementedError("Compute loss method should be implemented by the subclass")

    def get_eos_token(self) -> str:
        raise NotImplementedError("Get eos token method should be implemented by the subclass")
