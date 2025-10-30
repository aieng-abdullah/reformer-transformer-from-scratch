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

## 🚀 What is Reformer?

Reformer is a breakthrough transformer architecture designed for **scaling attention to ultra-long sequences** (e.g., 64K tokens), introduced by [Kitaev et al., 2020](https://arxiv.org/abs/2001.04451). It tackles the quadratic memory/time bottleneck of standard attention using smart architectural choices:

| ⚙️ Component                      | 💡 Purpose                                                                 |
|----------------------------------|----------------------------------------------------------------------------|
| 🔑 **LSH Attention**             | Reduces complexity from O(n²) → O(n log n) via Locality-Sensitive Hashing |
| ♻️ **Reversible Layers**         | Save GPU memory by recomputing instead of storing hidden states           |
| 🧱 **Chunked Feed-Forward**      | Breaks computation into chunks to reduce peak memory usage                |
| 📐 **Axial Positional Encoding** | Enables long sequence encoding without massive memory use                 |

---

## 📜 Paper Reference

> **Title:** Reformer: The Efficient Transformer  
> **Authors:** Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya  
> **Published in:** ICLR 2020  
> **Paper Link:** [arXiv:2001.04451](https://arxiv.org/abs/2001.04451)

---

## 🌟 Why This Project?

This is not just another implementation. This project aims to:

- 🧠 Teach the internals of Reformer, **line by line**
- 🧱 Provide a **modular**, clean PyTorch implementation
- 🧪 Serve as a base for **research experiments**, **MLOps pipelines**, or **AI portfolio showcases**
- 🎓 Help **ML engineers**, **students**, and **researchers** understand memory-efficient Transformers

---

## 🔍 Key Features

✅ **Locality-Sensitive Hashing Attention**  
✅ **Reversible Residual Connections**  
✅ **Chunked Feed-Forward Network**  
✅ **Axial Positional Encoding**  
✅ **Full PyTorch implementation from scratch**  
✅ **Clear comments, visualizations, and metrics tracking**  
✅ **GPU-ready & Colab-compatible**

---

## ⚡ Technical Highlights

### 1. LSH Self-Attention
- Hashes similar query vectors into the same bucket
- Only computes attention within buckets
- Complexity reduced to O(n log n)

### 2. Reversible Residual Layers
- Forwards can be reversed at backprop time
- Saves memory (no activations need to be stored)

### 3. Chunked Feed-Forward
- Applies FFN block on chunks instead of full sequences
- Minimizes memory peak during training

### 4. Axial Positional Encoding
- Efficient 2D position embedding
- Works well with very long sequences (e.g., 4096+)

---

## 🧠 Who Should Use This?

- **ML Engineers** needing a scalable transformer model
- **Students** looking to understand Reformer deeply
- **Researchers** running experiments on long-sequence modeling
- **Hackers** and **AI tinkerers** wanting full control of attention

---

## 🧰 Tools & Frameworks

- **Python 3.10+**
- **PyTorch 2.x**
- **Matplotlib / Seaborn** (for visualizations)
- **Google Colab** (GPU supported)

---

## 🎓 Learn More

- [Reformer Explained (by Illustrated Transformer author)](https://theaisummer.com/reformer/)
- [LSH Attention Visualized (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [Original Reformer GitHub (Google)](https://github.com/google/trax)

---

## 🧩 License

MIT License © 2025 — Built for educational, research, and experimentation purposes.

---

<p align="center">
  Built with 💥 by an ML Engineer who believes in code that teaches.
</p>

👨‍💻 Author
Abdullah Al Arif
[email:aieng.abdullah.arif@gmail.com]
