<h1 align="center">
   Reformer Transformer — From Scratch
</h1>

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1200/1*Yz45yt-uqMPYxDGbZGavTw.png" width="60%">
</p>

<p align="center">
  <b>A PyTorch Reimplementation of the Reformer Architecture — With LSH Attention, Reversible Layers, and Chunked Feed-Forward Networks</b>
</p>

---

## What is Reformer?

Reformer is a transformer architecture designed for **scaling attention to ultra-long sequences** (e.g., 64K tokens), introduced by [Kitaev et al., 2020](https://arxiv.org/abs/2001.04451).

| Component | Purpose |
|---|---|
| **LSH Attention** | Reduces attention from O(n²) → O(n log n) via Locality-Sensitive Hashing |
| **Reversible Layers** | Recomputes activations during backward instead of storing them |
| **Chunked Feed-Forward** | Applies FFN on sequence chunks to reduce peak memory |
| **Axial Positional Encoding** | Decomposes PE into row + column for long sequences |

---

## Project Structure

```
reformer-transformer-from-scratch/
├── reformer/                    # Core package
│   ├── attention.py             # LSH self-attention
│   ├── feedforward.py           # Chunked feed-forward network
│   ├── reversible.py            # Reversible block (gradient checkpointing)
│   ├── pos_encoding.py          # Axial positional encoding
│   ├── block.py                 # ReformerBlock (attention + FFN)
│   ├── model.py                 # ReformerModel (full model)
│   ├── dataset.py               # CopyDataset for demo training
│   ├── training.py              # Train/evaluate loops
│   ├── generation.py            # Autoregressive text generation
│   └── utils.py                 # Checkpoint save/load
├── scripts/
│   └── train.py                 # CLI entrypoint
├── tests/                       # Test suite (24 tests)
├── notebooks/
│   └── reformer_research_implementation.ipynb
├── config.json                  # Hyperparameters + sweep config
├── pyproject.toml               # Package metadata + deps
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+, CUDA GPU recommended (tested on GTX 1050 Ti 4GB)

```bash
# Install dependencies
pip install -e .

# Run tests
python -m pytest tests/ -v
```

---

## Training

```bash
# Single training run (uses config.json defaults)
python scripts/train.py --mode single

# Hyperparameter sweep (4 configs)
python scripts/train.py --mode sweep

# Override device / seed
python scripts/train.py --mode single --device cpu --seed 42
```

### Config (`config.json`)

```json
{
  "model": { "dim": 64, "n_layers": 2, "n_heads": 4, "bucket_size": 16 },
  "training": { "batch_size": 16, "epochs": 2, "use_amp": true },
  "sweep": { "dim": [64, 128], "n_layers": [2, 4] }
}
```

AMP (mixed precision) is enabled by default for CUDA GPUs.

---

## Technical Highlights

### 1. LSH Self-Attention
Queries are hashed into buckets so only vectors in the same bucket attend to each other.

$$
\text{Attention}(Q_{B_k}, K_{B_k}, V_{B_k}) = \text{softmax}\left(\frac{Q_{B_k} K_{B_k}^T}{\sqrt{d}}\right) V_{B_k}
$$

### 2. Reversible Residual Layers
Recompute activations during backward instead of storing them:

$$
\begin{cases}
y_1 = x_1 + f(x_2) \\
y_2 = x_2 + g(y_1)
\end{cases} \quad \Rightarrow \quad
\begin{cases}
x_2 = y_2 - g(y_1) \\
x_1 = y_1 - f(x_2)
\end{cases}
$$

### 3. Axial Positional Encoding
Decomposes positional embeddings along row and column axes:

$$
PE_{(i,j)} = PE_\text{row}(i) + PE_\text{col}(j)
$$

---

## Tests

```bash
python -m pytest tests/ -v
```

24 tests covering:
- LSH attention (standard + LSH paths, backward, weight storage)
- Reversible block (forward, backward, gradient correctness, memory savings)
- Full model (forward shape, all-params-backward, sequence lengths, odd-dim error)
- Axial PE (shape, length variants, position uniqueness, gradient flow)
- Text generation (shape, length, input preservation, no-grad)

---

## Paper Reference

**Title:** Reformer: The Efficient Transformer
**Authors:** Nikita Kitaev, Lukasz Kaiser, Anselm Levskaya
**Published in:** ICLR 2020
**Paper:** [arXiv:2001.04451](https://arxiv.org/abs/2001.04451)

---

## Learn More

- [Reformer Explained (The AI Summer)](https://theaisummer.com/reformer/)
- [LSH Attention Visualized (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [Original Reformer GitHub (Google)](https://github.com/google/trax)

---

## License

MIT License

Author: Abdullah Al Arif
Email: aieng.abdullah.arif@gmail.com
