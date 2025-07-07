from abc import abstractmethod
import torch
import torch.nn as nn

from model_config import ENCODER_OUTPUT_DIM


class TimeSeriesEncoderBase(nn.Module):
    def __init__(
        self,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.dropout = dropout

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
