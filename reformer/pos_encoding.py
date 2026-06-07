import math

import torch
import torch.nn as nn


class AxialPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim

        # Compute grid dimensions: find factors closest to sqrt(max_seq_len)
        sqrt_len = int(math.sqrt(max_seq_len))
        while max_seq_len % sqrt_len != 0:
            sqrt_len -= 1
        self.n_rows = sqrt_len
        self.n_cols = max_seq_len // sqrt_len

        half_dim = dim // 2
        self.row_emb = nn.Embedding(self.n_rows, half_dim)
        self.col_emb = nn.Embedding(self.n_cols, half_dim)

    def forward(self, x):
        B, L, D = x.shape
        device = x.device

        positions = torch.arange(L, device=device)
        rows = positions // self.n_cols
        cols = positions % self.n_cols

        row_pe = self.row_emb(rows)
        col_pe = self.col_emb(cols)
        pe = torch.cat([row_pe, col_pe], dim=-1)

        if D > pe.shape[-1]:
            pe = torch.nn.functional.pad(pe, (0, D - pe.shape[-1]))
        pe = pe[:, :D]

        return x + pe.unsqueeze(0)
