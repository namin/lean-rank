import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm

class GraphSAGEProd(nn.Module):
    def __init__(self, in_dim: int, hid: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.pre = nn.Linear(in_dim, hid)
        self.convs = nn.ModuleList([SAGEConv(hid, hid) for _ in range(layers)])
        self.norms = nn.ModuleList([BatchNorm(hid) for _ in range(layers)])
        self.dropout = dropout
        self.head = nn.Linear(hid, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.pre(x))
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        y = self.head(h).squeeze(-1)
        return y, h
