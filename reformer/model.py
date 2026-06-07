import torch
import torch.nn as nn

from reformer.block import ReformerBlock
from reformer.pos_encoding import AxialPositionalEncoding


class ReformerModel(nn.Module):
    def __init__(
        self, vocab_size, dim, n_layers, n_heads, bucket_size, ffn_chunks, max_seq_len
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = AxialPositionalEncoding(dim, max_seq_len)
        self.layers = nn.ModuleList(
            [
                ReformerBlock(dim, n_heads, bucket_size, ffn_chunks)
                for _ in range(n_layers)
            ]
        )
        self.to_logits = nn.Linear(dim, vocab_size)

    def forward(self, x):
        b, t = x.shape
        x = self.token_emb(x)
        x = self.pos_emb(x)

        if x.shape[-1] % 2 != 0:
            raise ValueError("Embedding dim must be even for reversible residuals.")

        for layer in self.layers:
            x = layer(x)

        return self.to_logits(x)
