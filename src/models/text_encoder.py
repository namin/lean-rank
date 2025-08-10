import torch
from torch import nn
import torch.nn.functional as F

class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 64, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)
