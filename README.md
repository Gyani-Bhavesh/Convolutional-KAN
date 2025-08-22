# Convolutional-KAN

*A compact demo of Kolmogorov–Arnold Networks (KAN) combined with convolutional feature extractors for image classification (example: brain-cancer).*

> **What’s inside?**  
> This repo currently contains a single Jupyter notebook:
> - `brain-cancer-ckan.ipynb` — an end-to-end, notebook-only workflow that builds a **Convolutional-KAN** (CNN features → KAN classifier) for a medical-imaging style task.  
> (You can adapt the notebook to any image dataset by adjusting the data loader paths.)

---

## Table of Contents
- [Why KAN?]
- [How KAN Works (Detailed)]
- [Convolutional-KAN: Marrying CNNs with KANs]

---

## Why KAN?

**Kolmogorov–Arnold Networks (KANs)** replace fixed node activations with **learnable 1-D functions on edges** (connections). Each edge carries a univariate function (typically a B-spline expansion) that’s trained end-to-end.  
In practice, KANs can reach strong accuracy with comparatively few parameters and offer **clear interpretability hooks** (visualize/prune per-edge functions; even derive symbolic formulas).

---

## How KAN Works (Detailed)

### 1) From the Kolmogorov–Arnold Representation to KAN Layers
The (smoothed) Kolmogorov–Arnold representation theorem says any continuous multivariate function on a bounded domain can be written using **sums of univariate functions of sums of univariate functions**.  

One convenient form (for a function \(f:[0,1]^n \to \mathbb{R}\)) is:

$$
f(x) = \sum_{q=1}^{2n+1} \Phi_q\!\left(\sum_{p=1}^{n}\phi_{q,p}(x_p)\right)
$$

This motivates a **layer** \(\mathbf{\Phi}\) that maps \(n_\text{in} \to n_\text{out}\) using a matrix of univariate functions \(\phi_{j,i}(\cdot)\) placed on **edges** \(i \to j\).  

A **KAN** stacks such layers:

$$
\text{KAN}(\mathbf{x}) = \mathbf{\Phi}_{L-1} \circ \cdots \circ \mathbf{\Phi}_1 \circ \mathbf{\Phi}_0(\mathbf{x})
$$

In contrast, an MLP alternates **linear layers** and **fixed activations**. KAN removes linear weights and makes each connection a learnable **function**.

---

### 2) What’s on an Edge? (B-Spline Parameterization)

Each edge function is typically parameterized as a **B-spline** expansion over a 1-D grid:

$$
\phi_{j,i}(x) = \sum_{k} c_{j,i,k}\, B_k(x)
$$

where \(B_k\) are basis splines and \(c_{j,i,k}\) are learned coefficients.  
This gives **local, shape-aware nonlinearity**, efficient gradients, and simple visualization of how each input dimension bends through the network.  

Libraries like **`pykan`** expose these details and offer grid-refinement, sparsity, and pruning utilities.

---

### 3) Training, Sparsity & Interpretability

- **Optimization:** Standard backprop/optimizers over spline coefficients.  
- **Sparsity/Pruning:** Encourage edge sparsity (e.g., L1/entropy regularizers). Pruned KANs can be **reduced** to compact graphs and even **symbolic formulas** via basis fitting.  
- **Interpretability:** You can **plot** per-edge functions, see which inputs matter, and extract **closed-form** expressions for parts of the network.

> For a hands-on demo (initialization, training, pruning, symbolic extraction), see the official **pykan docs** “Hello, KAN!” page.

---
