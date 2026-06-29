# Phase 3 — Deep Learning & PyTorch Cheat Sheet

[← CheatSheets Index](./README.md) | [Full Phase Files](../04_Phase3_Part1_Deep_Learning_Theory.md)

---

## Backpropagation

```
Forward:   input → activations → loss
Backward:  dL/dW = dL/da · da/dz · dz/dW  (chain rule)

For linear layer (z = Wx + b):
  dL/dW = dL/dz · x^T
  dL/dx = W^T · dL/dz
  dL/db = sum(dL/dz, axis=0)
```

---

## Activation Functions

| Function | Formula | Use Case | Problem |
|----------|---------|----------|---------|
| ReLU | $\max(0, x)$ | Hidden layers (default) | Dying ReLU (x<0 → grad=0) |
| GELU | $x\Phi(x)$ | **Transformers** | Slower than ReLU |
| SiLU/Swish | $x \cdot \sigma(x)$ | LLaMA, Mistral | — |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | Binary output | Vanishing gradient |
| Softmax | $\frac{e^{z_i}}{\sum e^{z_j}}$ | Multi-class output | — |

---

## Core PyTorch Patterns

```python
import torch
import torch.nn as nn

# --- TRAINING LOOP ---
model = MyModel().to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding grads
        optimizer.step()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        # validation loop
        pass
```

```python
# --- CUSTOM MODULE ---
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

---

## Tensor Operations Quick Reference

| Operation | Code |
|-----------|------|
| Shape | `t.shape`, `t.size(0)` |
| Reshape | `t.view(b, -1)`, `t.reshape(b, s, d)` |
| Permute | `t.permute(0, 2, 1)` — reorder dims |
| Squeeze | `t.squeeze(1)` — remove dim-1 |
| Unsqueeze | `t.unsqueeze(0)` — add dim |
| Matmul | `t1 @ t2` or `torch.bmm(t1, t2)` for batched |
| Concat | `torch.cat([a, b], dim=-1)` |
| Stack | `torch.stack([a, b], dim=0)` — new dim |
| Move to GPU | `t.to("cuda")` or `t.cuda()` |
| Detach | `t.detach()` — no gradient tracking |

---

## Optimizers

| Optimizer | Best For | Key Params |
|-----------|---------|-----------|
| SGD + momentum | Image models (ResNet) | `lr`, `momentum=0.9` |
| **AdamW** | **Transformers (default)** | `lr`, `weight_decay=0.01` |
| Adam | General | `lr`, `betas=(0.9, 0.999)` |
| RMSProp | RNNs | `lr`, `alpha=0.99` |

```python
# AdamW with linear warmup (transformer standard)
from transformers import get_linear_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=total_steps
)
```

---

## Regularization Techniques

| Technique | Where | Code |
|-----------|-------|------|
| L2 weight decay | Optimizer | `weight_decay=0.01` in AdamW |
| Dropout | Hidden layers | `nn.Dropout(p=0.1)` |
| Batch norm | After linear/conv, before activation | `nn.BatchNorm1d(dim)` |
| Layer norm | **Transformers** | `nn.LayerNorm(dim)` |
| Gradient clipping | During training | `clip_grad_norm_(..., 1.0)` |
| Early stopping | Training loop | Track val loss, stop if no improvement |

---

## Common Mistakes

- **Forgetting `optimizer.zero_grad()`** — gradients accumulate across batches
- **Not calling `model.eval()` during validation** — affects dropout, batch norm
- **Forgetting `torch.no_grad()` during inference** — wastes memory tracking grads
- **Data on wrong device** — ensure `model` and data both on `"cuda"`
- **Batch norm on batch size 1** — use layer norm or group norm instead

---

## Interview Quick-Hits

**Q: Explain backpropagation.**  
A: The chain rule applied to neural networks. The gradient of the loss w.r.t. each parameter equals the product of gradients along the path from the loss to that parameter. PyTorch builds a computation graph in the forward pass and traverses it in reverse during `.backward()`.

**Q: Why AdamW over Adam?**  
A: Adam incorrectly applies L2 regularization through the gradient update rather than the weight. AdamW separates weight decay from the gradient-based update, which is mathematically correct and empirically better for transformers.

**Q: What is gradient clipping and why use it?**  
A: Clips gradient norms above a threshold (typically 1.0). Prevents exploding gradients in deep networks and recurrent networks. Essential for training transformers.

**Q: When use BatchNorm vs LayerNorm?**  
A: BatchNorm normalizes across the batch dimension — works well for CNNs where batch is large. LayerNorm normalizes across the feature dimension — used in transformers because it's batch-size-independent (works with batch size 1).
