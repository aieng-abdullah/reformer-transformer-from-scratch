# AGENTS.md

## What this is

PyTorch Reformer implementation (LSH attention, reversible layers, chunked feed-forward, axial positional encoding). Structured as a Python package with CLI training and a test suite.

## Package layout

```
reformer/          # Core package
scripts/train.py   # CLI entrypoint
tests/             # 24 tests
notebooks/         # Reference notebook
config.json        # All hyperparameters
```

## Key commands

```bash
pip install -e .                          # Install package + deps
python -m pytest tests/ -v                # Run all 24 tests
python scripts/train.py --mode single     # Train one config
python scripts/train.py --mode sweep      # Hyperparameter sweep (4 configs)
python scripts/train.py --device cpu      # Force CPU (default: cuda if available)
```

## Config (`config.json`)

Single source of truth for model architecture, training params, and sweep grid. CLI reads from this file. Defaults are tuned for 4GB VRAM (GTX 1050 Ti).

## Gotchas

- `dim` must be even (reversible layer splits input in half)
- `bucket_size` must divide `max_seq_len` cleanly for LSH bucketing
- AMP is enabled by default on CUDA — disable via `use_amp: false` in config
- Notebook is in `notebooks/` (trailing space in original filename was fixed)
- Sweep produces 4 configs: `dim × n_layers` (2×2), other params held fixed
