import torch

from reformer.model import ReformerModel


def test_forward_shape():
    model = ReformerModel(
        vocab_size=100, dim=64, n_layers=2,
        n_heads=4, bucket_size=16, ffn_chunks=8, max_seq_len=64
    )
    x = torch.randint(0, 100, (2, 64))
    out = model(x)
    assert out.shape == (2, 64, 100)


def test_backward_all_params():
    model = ReformerModel(
        vocab_size=100, dim=64, n_layers=2,
        n_heads=4, bucket_size=16, ffn_chunks=8, max_seq_len=64
    )
    x = torch.randint(0, 100, (2, 64))
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_different_seq_lens():
    model = ReformerModel(
        vocab_size=100, dim=64, n_layers=2,
        n_heads=4, bucket_size=16, ffn_chunks=8, max_seq_len=64
    )
    for length in [16, 32, 48, 64]:
        x = torch.randint(0, 100, (1, length))
        out = model(x)
        assert out.shape == (1, length, 100), f"Failed for length={length}"


def test_odd_dim_raises():
    try:
        model = ReformerModel(
            vocab_size=100, dim=65, n_layers=2,
            n_heads=4, bucket_size=16, ffn_chunks=8, max_seq_len=64
        )
        x = torch.randint(0, 100, (1, 64))
        model(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "even" in str(e)


def test_parameter_count():
    model = ReformerModel(
        vocab_size=100, dim=64, n_layers=2,
        n_heads=4, bucket_size=16, ffn_chunks=8, max_seq_len=64
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
    print(f"  Total parameters: {n_params:,}")
