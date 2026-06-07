# AGENTS.md

## What this is

Single-notebook PyTorch implementation of the Reformer architecture (LSH attention, reversible layers, chunked feed-forward, axial positional encoding). No package structure, no tests, no build system.

## Key file

- `reformer_research_implementation .ipynb` — all code lives here (note the space in the filename)

## Running

```bash
# Install deps (no requirements.txt; infer from notebook imports)
pip install torch matplotlib seaborn

# Run the notebook
jupyter notebook "reformer_research_implementation .ipynb"
```

## Gotchas

- Filename has a trailing space before `.ipynb` — always quote or tab-complete it
- Python 3.10+, PyTorch 2.x required
- Designed for Google Colab with GPU; local runs need a CUDA GPU for reasonable performance
