import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSHSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, bucket_size, n_hashes=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.last_attn_weights = None

    def hash_vectors(self, x, num_buckets):
        b, h, n, d = x.shape
        device = x.device

        num_buckets = max(2, num_buckets)
        half_buckets = max(1, num_buckets // 2)

        projections = torch.randn(
            self.n_hashes, d, half_buckets, device=device
        ) / math.sqrt(d)

        dots = torch.einsum("bhnd,rdk->bhnrk", x, projections)
        concat = torch.cat([dots, -dots], dim=-1)
        hashes = torch.argmax(concat, dim=-1)

        return hashes

    def attend_with_lsh(self, q, k, v):
        b, h, n, d = q.shape
        device = q.device

        num_buckets = max(1, n // self.bucket_size)

        q_hashes = self.hash_vectors(q, num_buckets)
        k_hashes = self.hash_vectors(k, num_buckets)

        q_hash = q_hashes[..., 0]
        k_hash = k_hashes[..., 0]

        q_sorted_idx = torch.argsort(q_hash, dim=-1)
        k_sorted_idx = torch.argsort(k_hash, dim=-1)

        q_sorted = torch.gather(
            q, 2, q_sorted_idx.unsqueeze(-1).expand(-1, -1, -1, d)
        )
        k_sorted = torch.gather(
            k, 2, k_sorted_idx.unsqueeze(-1).expand(-1, -1, -1, d)
        )
        v_sorted = torch.gather(
            v, 2, k_sorted_idx.unsqueeze(-1).expand(-1, -1, -1, d)
        )

        pad_len = (self.bucket_size - (n % self.bucket_size)) % self.bucket_size
        if pad_len > 0:
            q_sorted = F.pad(q_sorted, (0, 0, 0, pad_len))
            k_sorted = F.pad(k_sorted, (0, 0, 0, pad_len))
            v_sorted = F.pad(v_sorted, (0, 0, 0, pad_len))

        seq_len_padded = q_sorted.shape[2]

        q_buckets = q_sorted.reshape(b, h, -1, self.bucket_size, d)
        k_buckets = k_sorted.reshape(b, h, -1, self.bucket_size, d)
        v_buckets = v_sorted.reshape(b, h, -1, self.bucket_size, d)

        scores = torch.matmul(q_buckets, k_buckets.transpose(-2, -1))
        scores = scores / math.sqrt(d)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out_buckets = torch.matmul(attn, v_buckets)

        out_sorted = out_buckets.reshape(b, h, seq_len_padded, d)

        if pad_len > 0:
            out_sorted = out_sorted[:, :, :-pad_len]

        unsort_idx = torch.argsort(q_sorted_idx, dim=-1)
        out = torch.gather(
            out_sorted, 2, unsort_idx.unsqueeze(-1).expand(-1, -1, -1, d)
        )

        self.last_attn_weights = attn[:, :, -1].detach().cpu()

        return out

    def forward(self, x, use_lsh=True):
        b, n, d = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        head_dim = d // self.n_heads
        q = q.view(b, n, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(b, n, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(b, n, self.n_heads, head_dim).transpose(1, 2)

        if use_lsh and n > self.bucket_size:
            out = self.attend_with_lsh(q, k, v)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            self.last_attn_weights = attn.detach().cpu()
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(b, n, d)

        return self.to_out(out)
