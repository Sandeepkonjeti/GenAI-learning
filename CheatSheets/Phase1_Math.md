# Phase 1 — Mathematics for ML Cheat Sheet

[← CheatSheets Index](./README.md) | [Full Phase File](../02_Phase1_Mathematics_for_ML.md)

---

## Linear Algebra

| Concept | Key Formula / Fact |
|---------|-------------------|
| Matrix multiply | $(AB)_{ij} = \sum_k A_{ik} B_{kj}$ — not commutative |
| Transpose | $(AB)^T = B^T A^T$ |
| Inverse | $AA^{-1} = I$; only square, full-rank matrices |
| Determinant | Scalar volume scaling; $\det(AB) = \det(A)\det(B)$ |
| Eigenvalue | $Av = \lambda v$ — $v$ direction unchanged, scaled by $\lambda$ |
| SVD | $A = U\Sigma V^T$ — always exists, even non-square |
| Norm | L2: $\|v\| = \sqrt{\sum v_i^2}$ — L1: $\|v\|_1 = \sum |v_i|$ |
| Cosine similarity | $\cos\theta = \frac{a \cdot b}{\|a\|\|b\|}$ — used in embeddings |

### Why it matters for AI
- **Attention** = matrix multiplication (Q, K, V projections)
- **Backprop** = chain rule via Jacobians
- **SVD** = dimensionality reduction (PCA), LoRA weight decomposition
- **Cosine similarity** = embedding search

---

## Calculus

| Concept | Formula |
|---------|---------|
| Chain rule | $\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}$ |
| Gradient | $\nabla_\theta L$ — vector of partial derivatives |
| Gradient descent | $\theta \leftarrow \theta - \alpha \nabla_\theta L$ |
| Jacobian | Matrix of all partial derivatives: $J_{ij} = \frac{\partial f_i}{\partial x_j}$ |
| Softmax gradient | $\frac{\partial \text{softmax}(z)_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$ |

### Backpropagation in words
1. Forward pass: compute activations, cache intermediate values
2. Backward pass: apply chain rule from loss to each parameter
3. Gradient flows back: `dL/dW = dL/dy · dy/dW`

---

## Probability & Statistics

| Concept | Formula / Definition |
|---------|---------------------|
| Bayes' theorem | $P(A|B) = \frac{P(B|A) P(A)}{P(B)}$ |
| MLE | $\hat\theta = \arg\max_\theta \prod_i p(x_i;\theta)$ |
| KL divergence | $D_{KL}(P\|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)} \geq 0$ |
| Cross-entropy | $H(P,Q) = -\sum_x P(x)\log Q(x) = H(P) + D_{KL}(P\|Q)$ |
| Entropy | $H(X) = -\sum_x p(x)\log p(x)$ — measures uncertainty |
| Gaussian | $\mathcal{N}(\mu,\sigma^2)$; 68/95/99.7% within 1/2/3 $\sigma$ |
| Softmax | $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ |

---

## Key Relationships

```
Cross-entropy loss  ←→  negative log-likelihood  ←→  MLE
KL divergence       ←→  RLHF KL penalty (keeps model close to reference)
Gaussian prior      ←→  L2 regularization (MAP estimation)
Laplace prior       ←→  L1 regularization (sparsity)
```

---

## Information Theory (LLM Connection)

| Term | Definition | LLM Context |
|------|-----------|-------------|
| Perplexity | $2^{H(P,Q)}$ — model's surprise | Lower = better LLM |
| Bits per token | $-\log_2 p(\text{token})$ | GPT-4 ≈ 3–4 bits/token |
| Temperature | Scales logits before softmax: $\text{softmax}(z/T)$ | T→0: greedy; T→∞: uniform |

---

## Common Mistakes

- **Confusing $P(A|B)$ with $P(B|A)$** — the base rate fallacy
- **Matrix shapes**: $(m,n) @ (n,k) = (m,k)$ — inner dims must match
- **KL divergence is not symmetric**: $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$
- **log(0)**: undefined — add epsilon in practice: `log(p + 1e-8)`
- **Eigenvectors of non-symmetric matrices** are not orthogonal

---

## Interview Quick-Hits

**Q: Why does gradient descent minimize loss?**  
A: The negative gradient points in the direction of steepest descent. Each step moves parameters to reduce loss. Converges to local minimum (global for convex loss).

**Q: What is KL divergence and where is it used in LLMs?**  
A: $D_{KL}(P\|Q)$ measures how much Q diverges from P. In RLHF: the reward objective includes a KL penalty $-\beta D_{KL}(\pi\|\pi_{ref})$ to prevent the fine-tuned model from drifting too far from the base model.

**Q: What does cross-entropy loss measure?**  
A: How different the model's predicted distribution Q is from the true distribution P. For classification: $-\log(\text{predicted probability of correct class})$.

**Q: Why is SVD useful for LoRA?**  
A: LoRA approximates weight updates as $\Delta W = BA$ where B is $d \times r$ and A is $r \times k$, $r \ll \min(d,k)$. This is low-rank decomposition — the same idea as truncated SVD.
