import torch
from torch import nn
import torch.nn.functional as F

class MLPProd(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y = self.out(h).squeeze(-1)
        return y
