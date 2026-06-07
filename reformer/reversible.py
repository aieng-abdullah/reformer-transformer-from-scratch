import torch.nn as nn


class ReversibleBlock(nn.Module):
    def __init__(self, f_module, g_module):
        super().__init__()
        self.f = f_module
        self.g = g_module

    def forward(self, x1, x2):
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return y1, y2
