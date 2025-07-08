import torch.nn as nn


class LinearProjector(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()
        self.projector = nn.Linear(input_dim, output_dim).to(device)

    def forward(self, x):
        return self.projector(x)
