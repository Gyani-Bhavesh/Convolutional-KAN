# Convolutional-KAN

*A compact demo of Kolmogorov–Arnold Networks (KAN) combined with convolutional feature extractors for image classification (example: brain-cancer).*

> **What’s inside?**  
> This repo currently contains a single Jupyter notebook:
> - `brain-cancer-ckan.ipynb` — an end-to-end, notebook-only workflow that builds a **Convolutional-KAN** (CNN features → KAN classifier) for a medical-imaging style task.  
> (You can adapt the notebook to any image dataset by adjusting the data loader paths.)

---

## Table of Contents
- [Why KAN?](#why-kan)
- [How KAN Works (Detailed)](#how-kan-works-detailed)
- [Convolutional-KAN: Marrying CNNs with KANs](#convolutional-kan-marrying-cnns-with-kans)
- [Quickstart](#quickstart)
- [Repository Structure](#repository-structure)
- [Training & Evaluation](#training--evaluation)
- [Interpreting a Trained KAN](#interpreting-a-trained-kan)
- [Tips & Pitfalls](#tips--pitfalls)
- [Roadmap](#roadmap)
- [Citations & Further Reading](#citations--further-reading)
- [License](#license)

---

## Why KAN?

**Kolmogorov–Arnold Networks (KANs)** replace fixed node activations with **learnable 1-D functions on edges** (connections). Each edge carries a univariate function (typically a B-spline expansion) that’s trained end-to-end. In practice, KANs can reach strong accuracy with comparatively few parameters and offer **clear interpretability hooks** (visualize/prune per-edge functions; even derive symbolic formulas).

---

## How KAN Works (Detailed)

### 1) From the Kolmogorov–Arnold Representation to KAN Layers
The (smoothed) Kolmogorov–Arnold representation theorem says any continuous multivariate function on a bounded domain can be written using **sums of univariate functions of sums of univariate functions**. One convenient form (for \(f:[0,1]^n\to\mathbb{R}\)) is:
\[
f(x)=\sum_{q=1}^{2n+1}\Phi_q\!\left(\sum_{p=1}^{n}\phi_{q,p}(x_p)\right)
\]
This motivates a **layer** \(\mathbf{\Phi}\) that maps \(n_{\text{in}}\!\to n_{\text{out}}\) using a matrix of univariate functions \(\phi_{j,i}(\cdot)\) placed on **edges** \(i\to j\). A **KAN** stacks such layers:
\[
\text{KAN}(\mathbf{x}) = \mathbf{\Phi}_{L-1}\circ\cdots\circ \mathbf{\Phi}_1\circ \mathbf{\Phi}_0(\mathbf{x})
\]
In contrast, an MLP alternates **linear layers** and **fixed activations**. KAN removes linear weights and makes each connection a learnable **function**. :contentReference[oaicite:2]{index=2}

### 2) What’s on an Edge? (B-Spline Parameterization)
Each edge function is typically parameterized as a **B-spline** expansion over a 1-D grid:
\[
\phi_{j,i}(x)=\sum_{k} c_{j,i,k}\,B_k(x)
\]
where \(B_k\) are basis splines and \(c_{j,i,k}\) are learned coefficients. This gives **local, shape-aware nonlinearity**, efficient gradients, and simple visualization of how each input dimension bends through the network. Libraries like **`pykan`** expose these details and offer grid-refinement, sparsity, and pruning utilities. :contentReference[oaicite:3]{index=3}

### 3) Training, Sparsity & Interpretability
- **Optimization:** Standard backprop/optimizers over spline coefficients.  
- **Sparsity/Pruning:** Encourage edge sparsity (e.g., L1/entropy regularizers). Pruned KANs can be **reduced** to compact graphs and even **symbolic formulas** via basis fitting.  
- **Interpretability:** You can **plot** per-edge functions, see which inputs matter, and extract **closed-form** expressions for parts of the network. :contentReference[oaicite:4]{index=4}

> For a gentle hands-on tour (initialization, training, pruning, symbolic extraction), see the official **pykan docs** “Hello, KAN!” page. 

---

## Convolutional-KAN: Marrying CNNs with KANs

In image tasks, a practical pattern is:
1. **Convolutional backbone** (e.g., a small CNN or a torchvision model) to produce a feature vector \(\mathbf{z}\in\mathbb{R}^d\).
2. **KAN head** that maps \(\mathbf{z}\to\) logits via one or more KAN layers (e.g., width \([d, h, C]\) for \(C\) classes).

This “**Conv → KAN**” split keeps the spatial bias/efficiency of CNNs while leveraging KAN’s **edge-function interpretability** and compact parameterization on the classifier head. (For fully KAN-ified convs, see research repos exploring KAN-style conv operators.)

**Minimal PyTorch sketch (using `pykan`):**
```python
import torch, torch.nn as nn
from kan import KAN  # pip install pykan

class ConvBackbone(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(64, out_dim)
    def forward(self, x):
        x = self.net(x)          # (B,64,1,1)
        x = x.view(x.size(0), -1)
        return self.proj(x)      # (B,out_dim)

class ConvKAN(nn.Module):
    def __init__(self, feat_dim=256, num_classes=2):
        super().__init__()
        self.backbone = ConvBackbone(out_dim=feat_dim)
        # KAN with cubic splines (k=3) and 5 grid intervals
        self.kan = KAN(width=[feat_dim, 64, num_classes], grid=5, k=3, seed=0)
    def forward(self, x):
        z = self.backbone(x)
        return self.kan(z)       # logits
