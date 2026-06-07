import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkedFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, L, D = x.shape
        outputs = []

        for i in range(0, L, self.chunk_size):
            chunk = x[:, i : i + self.chunk_size, :]
            chunk = F.relu(self.linear1(chunk))
            chunk = self.dropout(chunk)
            chunk = self.linear2(chunk)
            outputs.append(chunk)

        return torch.cat(outputs, dim=1)
