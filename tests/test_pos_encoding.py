import torch
import torch.nn as nn

from reformer.pos_encoding import AxialPositionalEncoding


def test_output_shape():
    dim = 64
    max_len = 64
    pe = AxialPositionalEncoding(dim, max_len)
    x = torch.randn(2, max_len, dim)
    out = pe(x)
    assert out.shape == (2, max_len, dim)


def test_different_lengths():
    dim = 64
    max_len = 64
    pe = AxialPositionalEncoding(dim, max_len)

    for length in [16, 32, 48, 64]:
        x = torch.randn(1, length, dim)
        out = pe(x)
        assert out.shape == (1, length, dim), f"Failed for length={length}"


def test_positions_are_different():
    dim = 64
    max_len = 64
    pe = AxialPositionalEncoding(dim, max_len)
    x = torch.zeros(1, max_len, dim)
    out = pe(x)

    pos0 = out[0, 0]
    pos1 = out[0, 1]
    pos_last = out[0, -1]
    assert not torch.allclose(pos0, pos1), "Adjacent positions should differ"
    assert not torch.allclose(pos0, pos_last), "First and last positions should differ"


def test_requires_grad():
    dim = 64
    max_len = 64
    pe = AxialPositionalEncoding(dim, max_len)
    x = torch.randn(1, max_len, dim)
    out = pe(x)
    loss = out.sum()
    loss.backward()
    assert pe.row_emb.weight.grad is not None
    assert pe.col_emb.weight.grad is not None


def test_odd_dim_padding():
    dim = 65
    max_len = 64
    pe = AxialPositionalEncoding(dim, max_len)
    x = torch.randn(1, max_len, dim)
    out = pe(x)
    assert out.shape == (1, max_len, dim)
