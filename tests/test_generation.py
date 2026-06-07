import torch

from reformer import ReformerModel
from reformer.generation import generate_text


def test_generation_output_shape():
    model = ReformerModel(
        vocab_size=100, dim=64, n_layers=2,
        n_heads=4, bucket_size=16, ffn_chunks=8, max_seq_len=64
    )
    device = torch.device("cpu")
    start = torch.randint(1, 100, (1, 10))
    out = generate_text(model, start, 20, device)
    assert out.shape == (1, 20)


def test_generation_increases_length():
    model = ReformerModel(
        vocab_size=100, dim=64, n_layers=2,
        n_heads=4, bucket_size=16, ffn_chunks=8, max_seq_len=64
    )
    device = torch.device("cpu")
    start = torch.randint(1, 100, (1, 5))
    out = generate_text(model, start, 15, device)
    assert out.shape[1] == 15


def test_generation_preserves_input():
    model = ReformerModel(
        vocab_size=100, dim=64, n_layers=2,
        n_heads=4, bucket_size=16, ffn_chunks=8, max_seq_len=64
    )
    device = torch.device("cpu")
    start = torch.randint(1, 100, (1, 8))
    out = generate_text(model, start, 8, device)
    assert out.shape == (1, 8)
    assert torch.equal(out[:, :8], start)


def test_generation_no_grad():
    model = ReformerModel(
        vocab_size=100, dim=64, n_layers=2,
        n_heads=4, bucket_size=16, ffn_chunks=8, max_seq_len=64
    )
    device = torch.device("cpu")
    start = torch.randint(1, 100, (1, 10))
    out = generate_text(model, start, 20, device)
    assert not out.requires_grad
