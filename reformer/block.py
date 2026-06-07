import torch
import torch.nn as nn

from reformer.attention import LSHSelfAttention
from reformer.feedforward import ChunkedFeedForward


class ReformerBlock(nn.Module):
    def __init__(self, dim, n_heads, bucket_size, ffn_chunks):
        super().__init__()
        self.attn = LSHSelfAttention(dim // 2, n_heads, bucket_size, n_hashes=4)
        self.ff = ChunkedFeedForward(dim // 2, dim * 2, ffn_chunks)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)

        y1 = x1 + self.attn(x2, use_lsh=True)
        y2 = x2 + self.ff(y1)

        return torch.cat([y1, y2], dim=-1)
