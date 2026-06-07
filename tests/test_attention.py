import torch

from reformer.attention import LSHSelfAttention


def test_standard_attention_short_seq():
    dim, n_heads, batch, seq_len = 32, 4, 2, 8
    attn = LSHSelfAttention(dim, n_heads, bucket_size=16)
    x = torch.randn(batch, seq_len, dim)
    out = attn(x, use_lsh=False)
    assert out.shape == (batch, seq_len, dim)


def test_lsh_attention_long_seq():
    dim, n_heads, batch, seq_len = 32, 4, 2, 64
    attn = LSHSelfAttention(dim, n_heads, bucket_size=16)
    x = torch.randn(batch, seq_len, dim)
    out = attn(x, use_lsh=True)
    assert out.shape == (batch, seq_len, dim)


def test_attention_backwards():
    dim, n_heads, batch, seq_len = 32, 4, 2, 32
    attn = LSHSelfAttention(dim, n_heads, bucket_size=16)
    x = torch.randn(batch, seq_len, dim, requires_grad=True)
    out = attn(x, use_lsh=True)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.all(x.grad == 0)


def test_attention_stores_weights():
    dim, n_heads, batch, seq_len = 32, 4, 2, 32
    attn = LSHSelfAttention(dim, n_heads, bucket_size=16)
    x = torch.randn(batch, seq_len, dim)
    attn(x, use_lsh=True)
    assert attn.last_attn_weights is not None


def test_single_head():
    dim, batch, seq_len = 32, 2, 16
    attn = LSHSelfAttention(dim, n_heads=1, bucket_size=8)
    x = torch.randn(batch, seq_len, dim)
    out = attn(x)
    assert out.shape == (batch, seq_len, dim)
