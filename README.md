<h1 align="center">
   Reformer Transformer — From Scratch
</h1>

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit-1200/1*Yz45yt-uqMPYxDGbZGavTw.png" width="60%">
</p>

<p align="center">
  <b>A PyTorch Reimplementation of the Reformer Architecture — With LSH Attention, Reversible Layers, and Chunked Feed-Forward Networks</b>
</p>

---

## What is Reformer?

Reformer is a transformer architecture designed for **scaling attention to ultra-long sequences** (e.g., 64K tokens), introduced by [Kitaev et al., 2020](https://arxiv.org/abs/2001.04451).  
It addresses the quadratic memory and computational bottleneck of standard attention using several key ideas:

| Component                        | Purpose                                                                 |
|----------------------------------|------------------------------------------------------------------------|
| **LSH Attention**                | Reduces complexity from $O(n^2)$ → $O(n \log n)$ via Locality-Sensitive Hashing |
| **Reversible Layers**            | Saves GPU memory by recomputing intermediate activations instead of storing them |
| **Chunked Feed-Forward**         | Applies feed-forward layers on sequence chunks to reduce peak memory usage |
| **Axial Positional Encoding**    | Enables long sequence encoding efficiently without large positional matrices |

---

## Paper Reference

**Title:** Reformer: The Efficient Transformer  
**Authors:** Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya  
**Published in:** ICLR 2020  
**Paper Link:** [arXiv:2001.04451](https://arxiv.org/abs/2001.04451)

---

## Project Goals

This project aims to:

- Provide a **line-by-line understanding** of Reformer internals  
- Offer a **modular, clean PyTorch implementation**  
- Serve as a base for **research experiments** and **AI portfolio projects**  
- Support **ML engineers, students, and researchers** in learning memory-efficient Transformers  

---

## Key Features

- Locality-Sensitive Hashing Attention  
- Reversible Residual Layers  
- Chunked Feed-Forward Networks  
- Axial Positional Encoding  
- Full PyTorch implementation from scratch  
- Clear documentation, visualizations, and metrics tracking  
- GPU-ready and Colab-compatible

---

## Technical Highlights

### 1. LSH Self-Attention
Queries are hashed into buckets so that only vectors in the same bucket attend to each other.  
Reduces attention complexity from $\mathcal{O}(n^2)$ to $\mathcal{O}(n \log n)$.  

Within a bucket $B_k$, attention is computed as:

$$
\text{Attention}(Q_{B_k}, K_{B_k}, V_{B_k}) = \text{softmax}\left(\frac{Q_{B_k} K_{B_k}^T}{\sqrt{d}}\right) V_{B_k}
$$

### 2. Reversible Residual Layers
Standard residual connections store activations for backprop, consuming memory.  
Reversible layers recompute $x_{l}$ from $x_{l+1}$:

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

### 3. Chunked Feed-Forward
Instead of applying feed-forward layers to the entire sequence:

$$
\text{FFN}(X) = W_2 \cdot \text{GELU}(W_1 \cdot X + b_1) + b_2
$$

The input $X$ is divided into chunks to limit peak memory usage.

### 4. Axial Positional Encoding
Decomposes positional embeddings along axes (row and column) for long sequences:

$$
PE_{(i,j)} = PE_\text{row}(i) + PE_\text{col}(j)
$$

Allows very long sequences to be encoded efficiently.

---

## Who Should Use This?

- AI/ML engineers building scalable transformer models  
- Students seeking a deep understanding of Reformer internals  
- Researchers experimenting with long-sequence modeling  
- AI practitioners wanting full control of attention mechanics  

---

## Tools & Frameworks

- Python 3.10+  
- PyTorch 2.x  
- Matplotlib / Seaborn (for visualizations)  
- Google Colab (GPU-supported)

---

## Learn More

- [Reformer Explained (Illustrated Transformer)](https://theaisummer.com/reformer/)  
- [LSH Attention Visualized (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)  
- [Original Reformer GitHub (Google)](https://github.com/google/trax)

---

## License

MIT License © 2025 — Built for educational, research, and experimentation purposes.

---

<p align="center">
  Built by an AI Engineer committed to code that teaches.
</p>

Author: Abdullah Al Arif  
Email: aieng.abdullah.arif@gmail.com
