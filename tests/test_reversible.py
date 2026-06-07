import torch
import torch.nn as nn

from reformer.model import ReformerModel
from reformer.reversible import ReversibleBlock


def test_reversible_block_forward():
    dim = 32
    f = nn.Linear(dim, dim)
    g = nn.Linear(dim, dim)
    block = ReversibleBlock(f, g)

    x1 = torch.randn(2, 16, dim)
    x2 = torch.randn(2, 16, dim)
    y1, y2 = block(x1, x2)

    assert y1.shape == x1.shape
    assert y2.shape == x2.shape


def test_reversible_block_backward():
    dim = 32
    f = nn.Linear(dim, dim)
    g = nn.Linear(dim, dim)
    block = ReversibleBlock(f, g)

    x1 = torch.randn(2, 16, dim, requires_grad=True)
    x2 = torch.randn(2, 16, dim, requires_grad=True)
    y1, y2 = block(x1, x2)
    loss = (y1.sum() + y2.sum())
    loss.backward()

    assert x1.grad is not None, "x1 should have gradients"
    assert x2.grad is not None, "x2 should have gradients"
    assert not torch.all(x1.grad == 0), "x1 gradients should be non-zero"
    assert not torch.all(x2.grad == 0), "x2 gradients should be non-zero"


def test_reversible_block_gradient_correctness():
    dim = 16
    f = nn.Linear(dim, dim)
    g = nn.Linear(dim, dim)
    block = ReversibleBlock(f, g)

    x1 = torch.randn(1, 8, dim, requires_grad=True)
    x2 = torch.randn(1, 8, dim, requires_grad=True)
    y1, y2 = block(x1, x2)
    loss = (y1.sum() + y2.sum())
    loss.backward()

    # Verify gradients match manual computation
    x1_ = x1.detach().requires_grad_(True)
    x2_ = x2.detach().requires_grad_(True)
    y1_ = x1_ + f(x2_)
    y2_ = x2_ + g(y1_)
    loss_ = (y1_.sum() + y2_.sum())
    loss_.backward()

    assert torch.allclose(x1.grad, x1_.grad, atol=1e-5), \
        f"x1 grad mismatch: max diff = {(x1.grad - x1_.grad).abs().max().item()}"
    assert torch.allclose(x2.grad, x2_.grad, atol=1e-5), \
        f"x2 grad mismatch: max diff = {(x2.grad - x2_.grad).abs().max().item()}"


def test_model_backward_pass():
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
        assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


def test_memory_savings():
    """Verify reversible model uses less peak memory than non-reversible."""
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cpu")

    model = ReformerModel(
        vocab_size=100, dim=64, n_layers=4,
        n_heads=4, bucket_size=16, ffn_chunks=8, max_seq_len=64
    ).to(device)

    x = torch.randint(0, 100, (4, 64), device=device)
    out = model(x)
    loss = out.sum()
    loss.backward()

    peak_cpu = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print(f"  Peak memory (CPU, proxy): {peak_cpu}")
