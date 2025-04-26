from abc import abstractmethod
import torch
import torch.nn as nn

from model_config import ENCODER_INPUT_DIM, ENCODER_OUTPUT_DIM


class TimeSeriesEncoderBase(nn.Module):
    def __init__(
        self,
        input_dim: int = ENCODER_INPUT_DIM,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
