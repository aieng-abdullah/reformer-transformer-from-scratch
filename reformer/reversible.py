import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ReversibleBlock(nn.Module):
    def __init__(self, f_module, g_module):
        super().__init__()
        self.f = f_module
        self.g = g_module

    def forward(self, x1, x2):
        y1 = x1 + checkpoint(self.f, x2, use_reentrant=False)
        y2 = x2 + checkpoint(self.g, y1, use_reentrant=False)
        return y1, y2
