import torch
import torch.nn as nn

from reformer.attention import LSHSelfAttention
from reformer.feedforward import ChunkedFeedForward
from reformer.reversible import ReversibleBlock


class ReformerBlock(nn.Module):
    def __init__(self, dim, n_heads, bucket_size, ffn_chunks):
        super().__init__()
        half_dim = dim // 2
        self.f = LSHSelfAttention(half_dim, n_heads, bucket_size, n_hashes=4)
        self.g = ChunkedFeedForward(half_dim, dim * 2, ffn_chunks)
        self.reversible = ReversibleBlock(self.f, self.g)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        y1, y2 = self.reversible(x1, x2)
        return torch.cat([y1, y2], dim=-1)
