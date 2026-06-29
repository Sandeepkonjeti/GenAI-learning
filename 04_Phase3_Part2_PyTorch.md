# Phase 3b: PyTorch
**Months 7–9 | Difficulty: 6/10 | Label: 🔴 Must Learn**

> **Previous Phase**: [Phase 3a — Deep Learning Theory](./04_Phase3_Part1_Deep_Learning_Theory.md)  
> **Next Phase**: [Phase 4a — NLP Fundamentals](./05_Phase4_Part1_NLP_Fundamentals.md)

---

## Table of Contents

- [Phase Overview](#phase-overview)
- [Why PyTorch?](#why-pytorch)
- [Topic 1: Tensors and Autograd](#topic-1-tensors-and-autograd)
  - [Tensor Fundamentals](#tensor-fundamentals)
  - [Autograd — Automatic Differentiation](#autograd--automatic-differentiation)
- [Topic 2: Building Models with nn.Module](#topic-2-building-models-with-nnmodule)
  - [Custom Layers and Models](#custom-layers-and-models)
  - [Standard Layers Reference](#standard-layers-reference)
- [Topic 3: Data Loading](#topic-3-data-loading)
  - [Dataset and DataLoader](#dataset-and-dataloader)
- [Topic 4: The Complete Training Loop](#topic-4-the-complete-training-loop)
- [Topic 5: GPU Training](#topic-5-gpu-training)
- [Topic 6: Model Saving and Loading](#topic-6-model-saving-and-loading)
- [Topic 7: Mixed Precision Training](#topic-7-mixed-precision-training)
- [Topic 8: torch.compile and Performance](#topic-8-torchcompile-and-performance)
- [Resources](#resources)
- [Phase 3b Projects](#phase-3b-projects)
- [Common Mistakes](#common-mistakes)
- [Mastery Checklist](#mastery-checklist)

---

## Phase Overview

| Attribute | Details |
|-----------|---------|
| **Duration** | Months 7–9 (4 weeks, concurrent with Phase 3a weeks 5–6) |
| **Daily Time** | 1.5–2 hours |
| **Difficulty** | 6/10 |
| **Label** | 🔴 Must Learn |
| **Prerequisites** | Phase 3a complete (backprop, optimizers, normalization — all from scratch) |
| **Outcome** | Can implement any neural network architecture in PyTorch; writes clean, production-ready training code |

---

## Why PyTorch?

PyTorch is the framework of choice for AI research and is rapidly becoming the standard for production:

- **Hugging Face**: all models implemented in PyTorch
- **Research papers**: >90% of AI papers use PyTorch
- **vLLM, TGI, llama.cpp**: LLM inference built on PyTorch
- **Fine-tuning**: TRL, Axolotl, Unsloth all use PyTorch
- **Industry**: Meta AI, OpenAI (internal), Mistral, all use PyTorch

TensorFlow was once competitive; it is now in steep decline for new development. JAX is growing in research but PyTorch fluency is non-negotiable for the path ahead.

---

## Topic 1: Tensors and Autograd

### Tensor Fundamentals

A PyTorch tensor is a multi-dimensional array that can live on CPU or GPU, and automatically tracks operations for gradient computation.

```python
import torch
import numpy as np

# ── Creating Tensors ──
# From Python lists
t1 = torch.tensor([1.0, 2.0, 3.0])
t2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# Initialization
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 2)
eye = torch.eye(4)                          # Identity matrix
rand_uniform = torch.rand(3, 3)             # Uniform [0, 1)
rand_normal = torch.randn(3, 3)             # Standard normal N(0,1)
arange = torch.arange(0, 10, 2)            # Like np.arange

# From NumPy (shares memory if on CPU)
arr = np.array([1.0, 2.0, 3.0])
t_from_numpy = torch.from_numpy(arr)        # Zero-copy
t_copy = torch.tensor(arr)                  # Full copy

# Back to NumPy
arr_back = t1.numpy()                       # Zero-copy (CPU only)
arr_back_copy = t1.detach().numpy()         # Safe even with grad

# ── Shapes and Dimensions ──
t = torch.randn(8, 512, 768)               # (batch, seq, d_model)
print(t.shape)                              # torch.Size([8, 512, 768])
print(t.ndim)                               # 3
print(t.dtype)                              # torch.float32
print(t.device)                             # cpu

# Reshape operations
t2d = t.view(8, -1)                        # (8, 512*768) — requires contiguous
t2d = t.reshape(8, -1)                     # (8, 393216) — works always
t_perm = t.permute(1, 0, 2)               # (512, 8, 768) — change axis order
t_trans = t.transpose(0, 1)               # (512, 8, 768)

# Adding/removing dimensions
t_4d = t.unsqueeze(0)                      # (1, 8, 512, 768)
t_3d = t_4d.squeeze(0)                    # (8, 512, 768)

# ── Operations ──
a = torch.randn(3, 4)
b = torch.randn(4, 5)

# Matrix multiply
c = a @ b                                   # (3, 5)
c = torch.mm(a, b)                         # same, 2D only
c = torch.matmul(a, b)                     # works for batched too

# Batched matrix multiply
A = torch.randn(8, 3, 4)
B = torch.randn(8, 4, 5)
C = torch.bmm(A, B)                        # (8, 3, 5)
C = A @ B                                   # also works with broadcasting

# Element-wise
x = torch.tensor([1.0, 2.0, 3.0])
print(x + x)
print(x * x)
print(x ** 2)
print(torch.exp(x))
print(torch.log(x))

# Reductions
print(x.sum())
print(x.mean())
print(x.max())
print(x.argmax())

# Along a dimension
t = torch.randn(8, 512, 768)
mean_per_token = t.mean(dim=-1)            # (8, 512)
max_per_batch = t.max(dim=0).values        # (512, 768)
```

---

### Autograd — Automatic Differentiation

This is the core of PyTorch. Understanding it lets you debug gradient issues.

```python
import torch

# ── requires_grad: tell PyTorch to track operations on this tensor ──
x = torch.tensor([2.0, 3.0], requires_grad=True)

# ── Forward pass: build computation graph ──
y = x ** 2 + 3 * x + 1    # y = [4+6+1, 9+9+1] = [11, 19]
loss = y.sum()              # scalar: 11 + 19 = 30

# ── Backward pass: compute gradients ──
loss.backward()

# Gradient: d(loss)/d(x) = d(x^2 + 3x + 1)/dx = 2x + 3
print(x.grad)  # tensor([7., 9.]) = [2*2+3, 2*3+3]

# ── Gradient accumulation gotcha ──
# PyTorch ACCUMULATES gradients — you must zero them between steps!
x.grad.zero_()   # Zero gradients before next backward pass
# This is why training loops call optimizer.zero_grad() every step

# ── Detach: stop gradient tracking ──
y = x * 2
z = y.detach()   # z is a new tensor, no gradient tracking
# Use when:
# - Computing metrics (don't need gradients)
# - Storing values for logging
# - Creating "frozen" copies of tensors

# ── torch.no_grad(): inference mode ──
with torch.no_grad():
    # No gradient computation — faster, less memory
    # Use for: validation, inference, evaluation
    predictions = model(X_val)
    # This is equivalent to: detach + torch.inference_mode()

# ── Gradient checking in PyTorch ──
from torch.autograd import gradcheck

def my_function(x):
    return (x * 3 + 2).sum()

x = torch.randn(3, 4, requires_grad=True, dtype=torch.float64)
check = gradcheck(my_function, (x,), eps=1e-6, atol=1e-4)
print(f"Gradient check passed: {check}")

# ── Understanding the computation graph ──
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2        # grad_fn = PowBackward
z = y + 1         # grad_fn = AddBackward
w = z.sqrt()      # grad_fn = SqrtBackward

print(w.grad_fn)                    # <SqrtBackward0>
print(w.grad_fn.next_functions)     # Traces back through the graph

w.backward()
print(f"x.grad = {x.grad}")  # dw/dx = (1/(2*sqrt(x^2+1))) * 2x
```

---

## Topic 2: Building Models with nn.Module

### Custom Layers and Models

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── The nn.Module base class ──
class LinearLayer(nn.Module):
    """Custom linear layer — shows the nn.Module pattern."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()  # Always call super().__init__()
        
        # nn.Parameter: tells PyTorch "this is a learnable parameter"
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.linear does: x @ weight.T + bias
        return F.linear(x, self.weight, self.bias)

# ── A complete MLP in PyTorch ──
class MLP(nn.Module):
    """
    Multi-layer perceptron — the "hello world" of PyTorch models.
    This pattern is the foundation for ALL larger architectures.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout_rate: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        # Build layers dynamically
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # Not the last layer
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.GELU() if activation == 'gelu' else nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
        
        # nn.Sequential: list of modules applied in order
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Instantiate and inspect
model = MLP(input_dim=784, hidden_dims=[512, 256], output_dim=10)

# ── Essential model inspection ──
print("Model architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Parameter shapes
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")
```

### Standard Layers Reference

```python
import torch.nn as nn

# Linear / Dense
nn.Linear(in_features=512, out_features=768)

# Normalization
nn.LayerNorm(normalized_shape=768)           # Transformers
nn.BatchNorm1d(num_features=512)             # CNNs, MLPs
nn.BatchNorm2d(num_features=64)              # 2D CNNs

# Activations
nn.ReLU()
nn.GELU()                                    # Standard in transformers
nn.SiLU()                                    # Swish — used in LLaMA
nn.Softmax(dim=-1)
nn.Sigmoid()

# Dropout
nn.Dropout(p=0.1)                            # General
nn.Dropout2d(p=0.1)                          # For 2D feature maps

# Convolutions (for CNNs / multimodal)
nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
nn.MaxPool2d(kernel_size=2, stride=2)

# Embeddings (critical for NLP and LLMs)
nn.Embedding(num_embeddings=50257, embedding_dim=768)
# num_embeddings = vocabulary size
# embedding_dim = vector dimension

# Attention (PyTorch's built-in)
nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1, batch_first=True)

# Transformer components
nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1)
nn.TransformerDecoderLayer(d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1)
```

---

## Topic 3: Data Loading

### Dataset and DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json

# ── Custom Dataset ──
class TextClassificationDataset(Dataset):
    """
    Custom dataset for text classification.
    This pattern is used for every ML project with custom data.
    """
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ── DataLoader: batching, shuffling, parallel loading ──
# Training dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,           # Shuffle training data each epoch
    num_workers=4,          # Parallel data loading (set to 0 on Windows if issues)
    pin_memory=True,        # Faster CPU→GPU transfer
    drop_last=True          # Drop incomplete last batch
)

# Validation dataloader
val_loader = DataLoader(
    val_dataset,
    batch_size=64,          # Larger batch for validation (no gradient memory needed)
    shuffle=False,          # Don't shuffle validation
    num_workers=4,
    pin_memory=True
)

# ── Iterating ──
for batch in train_loader:
    input_ids = batch['input_ids']          # shape: (32, 512)
    attention_mask = batch['attention_mask'] # shape: (32, 512)
    labels = batch['label']                 # shape: (32,)
    break

# ── LLM-Style Dataset: Next Token Prediction ──
class CharacterLanguageModelDataset(Dataset):
    """
    Dataset for autoregressive language modeling.
    Given a sequence, predict the next token at each position.
    This is how GPT is pretrained.
    """
    def __init__(self, text: str, block_size: int = 256):
        # Create character-level vocabulary
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # char → int
        self.itos = {i: ch for i, ch in enumerate(chars)}  # int → char
        
        # Encode text as integers
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.block_size = block_size
    
    def __len__(self) -> int:
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Input: block_size tokens
        x = self.data[idx : idx + self.block_size]
        # Target: same tokens shifted by 1 (next token prediction)
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y
```

---

## Topic 4: The Complete Training Loop

This is the canonical PyTorch training loop. Memorize this structure.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb  # Weights & Biases for experiment tracking

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    device: str = 'auto'
) -> dict:
    """
    Complete PyTorch training loop.
    This structure applies to everything from MNIST to LLMs.
    """
    # ── Setup ──
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    
    model = model.to(device)
    
    # Optimizer: AdamW is standard for transformers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    total_steps = n_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # ══════════════════════════════════════════
        # TRAINING PHASE
        # ══════════════════════════════════════════
        model.train()    # Enable training mode (dropout, batch norm use training behavior)
        train_losses = []
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)    # Move to GPU
            y_batch = y_batch.to(device)
            
            # ① Zero gradients — MUST do this each step
            optimizer.zero_grad()
            
            # ② Forward pass
            logits = model(X_batch)
            
            # ③ Compute loss
            loss = criterion(logits, y_batch)
            
            # ④ Backward pass — compute gradients
            loss.backward()
            
            # ⑤ Gradient clipping — prevents exploding gradients
            # Standard for transformer training: clip to 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # ⑥ Update parameters
            optimizer.step()
            
            # ⑦ Update learning rate
            scheduler.step()
            
            train_losses.append(loss.item())
        
        # ══════════════════════════════════════════
        # VALIDATION PHASE
        # ══════════════════════════════════════════
        model.eval()     # Disable training mode (dropout off, batch norm uses running stats)
        val_losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():  # No gradient computation during validation
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())
                
                predictions = logits.argmax(dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
        
        # ══════════════════════════════════════════
        # RECORD AND REPORT
        # ══════════════════════════════════════════
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        val_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:3d}/{n_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # ── Save best model ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pt')
    
    return history
```

---

## Topic 5: GPU Training

```python
import torch

# ── Check what's available ──
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Device management — the clean pattern ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to GPU
model = MyModel().to(device)

# Move tensors to GPU
X = torch.randn(32, 784).to(device)
y = torch.randint(0, 10, (32,)).to(device)

# ── Memory management — critical for LLM work ──
# Check current VRAM usage
if torch.cuda.is_available():
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"VRAM cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Free cache between experiments
torch.cuda.empty_cache()

# ── Common OOM (Out of Memory) fixes ──
# 1. Reduce batch size
# 2. Use gradient accumulation (simulate larger batch)
# 3. Use mixed precision (float16/bfloat16)
# 4. Use gradient checkpointing (recompute activations instead of storing)

# ── Gradient Accumulation ──
# Simulates batch_size = micro_batch * accumulation_steps
accumulation_steps = 4
optimizer.zero_grad()

for i, (X_batch, y_batch) in enumerate(train_loader):
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    
    logits = model(X_batch)
    loss = criterion(logits, y_batch) / accumulation_steps  # Scale loss!
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

## Topic 6: Model Saving and Loading

```python
import torch
import torch.nn as nn
from pathlib import Path

# ── Save: state_dict (recommended) ──
def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint — includes both model and training state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': model.config if hasattr(model, 'config') else None,
    }, path)

# ── Load: state_dict ──
def load_checkpoint(model, optimizer, path, device='cpu'):
    """Load model checkpoint and resume training."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

# ── For inference only (no optimizer needed) ──
def load_for_inference(model_class, config, weights_path, device='cpu'):
    model = model_class(config)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ── Hugging Face-style saving ──
# When working with Hugging Face models, use their save/load
model.save_pretrained('./my-model-checkpoint')
# model = MyModelClass.from_pretrained('./my-model-checkpoint')
```

---

## Topic 7: Mixed Precision Training

Mixed precision training uses float16 (or bfloat16) for computations while keeping float32 for gradient accumulation. This approximately **doubles training speed** and **halves memory usage** with minimal accuracy cost.

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# ── Automatic Mixed Precision (AMP) ──
scaler = GradScaler()  # Handles gradient scaling to prevent underflow

model = MyModel().to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for X_batch, y_batch in train_loader:
    X_batch = X_batch.to('cuda')
    y_batch = y_batch.to('cuda')
    
    optimizer.zero_grad()
    
    # Forward pass in float16
    with autocast(dtype=torch.float16):
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
    
    # Backward pass: scaler prevents gradient underflow
    scaler.scale(loss).backward()
    
    # Unscale before clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Step + update scaler
    scaler.step(optimizer)
    scaler.update()

# ── bfloat16 (better for LLMs on modern hardware) ──
# bfloat16 has the same exponent range as float32 (more stable than float16)
# Used by default for training LLaMA, Mistral, and other modern LLMs
with autocast(dtype=torch.bfloat16):
    logits = model(X_batch)
    loss = criterion(logits, y_batch)
```

---

## Topic 8: torch.compile and Performance

`torch.compile` (PyTorch 2.0+) significantly speeds up training and inference by compiling the model to optimized code.

```python
import torch

model = MyTransformerModel(config)

# Compile the model — can give 1.5-3x speedup
# Mode options:
#   'default'      — balanced (recommended to start)
#   'reduce-overhead' — faster for small batches
#   'max-autotune' — slowest compile, fastest runtime
compiled_model = torch.compile(model, mode='default')

# Use exactly like a regular model
outputs = compiled_model(inputs)

# ── Performance profiling ──
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    with record_function("model_inference"):
        output = model(input_tensor)

# Print top operations by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## Resources

| Rank | Resource | Type | Cost | Why |
|------|----------|------|------|-----|
| 1 | [PyTorch Official Tutorials](https://pytorch.org/tutorials/) | Docs | Free | The definitive source. Work through "Learn the Basics" series first. |
| 2 | [Zero to Mastery PyTorch (Daniel Bourke)](https://www.learnpytorch.io/) | Course | Free | Best structured PyTorch course. Excellent pacing and exercises. |
| 3 | [Andrej Karpathy's YouTube series](https://www.youtube.com/@AndrejKarpathy) | YouTube | Free | Watch all of makemore series (parts 1–6). Builds a language model step by step. |
| 4 | [PyTorch documentation](https://pytorch.org/docs/stable/) | Docs | Free | Official reference. Use `torch.nn`, `torch.optim`, `torch.utils.data` sections. |
| 5 | [fast.ai course](https://course.fast.ai/) | Course | Free | Uses PyTorch underneath; top-down approach is excellent for intuition. |

---

## Phase 3b Projects

### Project 4: MNIST Classifier in PyTorch (Clean Architecture)
**Difficulty**: 4/10 | **Time**: 1 week

Requirements:
- Full `nn.Module` architecture
- Custom `Dataset` and `DataLoader`
- Complete training loop with validation and early stopping
- Learning rate scheduling (cosine annealing)
- Model checkpointing
- Training curves plot
- Final test accuracy >98%

```python
# Target architecture
model = MLP(
    input_dim=784,
    hidden_dims=[512, 256, 128],
    output_dim=10,
    dropout_rate=0.1
)
```

**Extension**: Use `torch.compile` and measure speedup. Add mixed precision training.

---

### Project 5: Character-Level GPT (Following Karpathy)
**Difficulty**: 7/10 | **Time**: 2 weeks

Implement from scratch following [Karpathy's "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY):
- Multi-head self-attention
- Transformer block (attention + FFN + residual + LayerNorm)
- Full GPT architecture
- Train on Shakespeare or any text corpus
- Generate new text

This project bridges Phase 3b and Phase 4b. It's the most important project in the roadmap.

```python
# Architecture to implement
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)         # (B, T, n_embd)
        pos_emb = self.position_embedding(
            torch.arange(T, device=idx.device)
        )                                           # (T, n_embd)
        x = tok_emb + pos_emb                      # (B, T, n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                   # (B, T, vocab_size)
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            return logits, loss
        return logits, None
```

---

### Project 6: Custom Dataset Pipeline
**Difficulty**: 4/10 | **Time**: 3 days

Build a production-quality data pipeline:
1. Custom `Dataset` with caching (save tokenized data to avoid re-tokenizing)
2. Stratified sampling for imbalanced classes
3. On-the-fly data augmentation
4. Multi-worker loading with `num_workers=4`
5. Benchmark: compare 1 worker vs 4 workers loading speed

---

## Common Mistakes

| Mistake | Why It's a Problem | Fix |
|---------|-------------------|-----|
| Forgetting `optimizer.zero_grad()` | Gradients accumulate → wrong updates → model diverges | First line of every training step |
| Not calling `model.eval()` during validation | Dropout stays active → inconsistent validation metrics | Always set `model.eval()` before validation |
| Not using `torch.no_grad()` during validation | Wastes memory building computation graph | Wrap validation in `with torch.no_grad():` |
| `.to(device)` only on model, not tensors | CUDA runtime error: tensor on wrong device | Move ALL tensors: `X.to(device)`, `y.to(device)` |
| Using `.data` instead of `.detach()` | `.data` bypasses autograd unsafely | Always use `.detach()` to stop gradient flow |
| `.view()` on non-contiguous tensors | RuntimeError | Use `.reshape()` instead |
| Not `pin_memory=True` in DataLoader | Slower CPU→GPU transfer | Always use `pin_memory=True` when training on GPU |
| Not clipping gradients | Exploding gradients in transformers | `clip_grad_norm_(model.parameters(), 1.0)` |

---

## Mastery Checklist

### Tensors and Autograd
- [ ] Can create, reshape, and manipulate tensors fluently
- [ ] Understands `requires_grad`, `.backward()`, `.grad` attribute
- [ ] Knows when to use `.detach()`, `torch.no_grad()`, `model.eval()`
- [ ] Can explain why `optimizer.zero_grad()` is needed

### Model Building
- [ ] Can write any `nn.Module` subclass from scratch
- [ ] Uses `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict` correctly
- [ ] Can count model parameters and inspect architecture

### Training Loop
- [ ] Writes the 7-step training loop from memory (zero_grad → forward → loss → backward → clip → step → scheduler)
- [ ] Implements validation loop with `model.eval()` + `torch.no_grad()`
- [ ] Implements model checkpointing (save best model)
- [ ] Uses learning rate scheduling

### Performance
- [ ] Trains on GPU with `.to(device)`
- [ ] Uses mixed precision (`autocast`)
- [ ] Uses gradient accumulation for effective large batches

### Projects
- [ ] MNIST classifier: clean training loop, >98% accuracy
- [ ] Character GPT: working text generation (following Karpathy)
- [ ] Custom data pipeline: multi-worker, cached, augmented

---

## Moving to Phase 4

**Before proceeding to [Phase 4a: NLP Fundamentals](./05_Phase4_Part1_NLP_Fundamentals.md), confirm:**

- [ ] MNIST classifier working with complete training loop
- [ ] Character-level GPT from scratch working (even character-level is fine)
- [ ] Comfortable writing `nn.Module` subclasses without tutorials
- [ ] Understands autograd deeply enough to debug gradient issues

**Why Phase 4 comes next**: You now have the tools (PyTorch) to build language models. Phase 4 gives you the domain knowledge (NLP) and the architecture (transformers) to build them correctly. The character-level GPT from Project 5 is a bridge directly into Phase 4.

---

## Phase Completion & Readiness Assessment

> Complete this assessment **before** moving to Phase 4. PyTorch mastery is the tool foundation for everything that follows — transformers, fine-tuning, and LLMs are all built in PyTorch.

---

### 1. Knowledge Checklist

**Tensors & Autograd**
- [ ] Tensor creation: `torch.tensor`, `torch.zeros`, `torch.randn`, `torch.arange`
- [ ] Tensor operations: reshape, view, squeeze, unsqueeze, permute, contiguous
- [ ] Device management: `.to(device)`, `torch.cuda.is_available()`, `.cuda()`, `.cpu()`
- [ ] `requires_grad=True`, gradient accumulation, `.grad` attribute
- [ ] `.backward()`, `retain_graph`, gradient of a scalar vs. vector
- [ ] `torch.no_grad()` and `model.eval()` — when and why
- [ ] `detach()` — removes tensor from computational graph

**nn.Module**
- [ ] `__init__` vs `forward` — what goes where
- [ ] `nn.Linear`, `nn.Conv2d`, `nn.LSTM`, `nn.Embedding`
- [ ] `nn.Sequential` for simple sequential models
- [ ] `parameters()` vs `named_parameters()` vs `state_dict()`
- [ ] `model.train()` vs `model.eval()` — effect on BatchNorm and Dropout
- [ ] Custom `nn.Module`: when and how to write one

**Training Loop**
- [ ] The 7-step training loop (from scratch, without a framework)
- [ ] `optimizer.zero_grad()` — why you must call this
- [ ] `loss.backward()` — what it does
- [ ] `optimizer.step()` — what it does
- [ ] Gradient clipping with `clip_grad_norm_`
- [ ] Learning rate schedulers: `CosineAnnealingLR`, `ReduceLROnPlateau`, `OneCycleLR`

**Data Loading**
- [ ] `Dataset` class: `__len__` and `__getitem__`
- [ ] `DataLoader`: `batch_size`, `shuffle`, `num_workers`, `pin_memory`
- [ ] `collate_fn` — why and when to write a custom one
- [ ] Data augmentation in `torchvision.transforms`

**Mixed Precision & Performance**
- [ ] `torch.cuda.amp.autocast` — what it does, when to use it
- [ ] `GradScaler` — why needed with FP16
- [ ] `torch.compile` — what it does, how to use it
- [ ] Memory profiling: `torch.cuda.memory_allocated()`

---

### 2. Practical Skills Checklist

- [ ] Write the complete 7-step training loop from memory (no reference)
- [ ] Build a CNN with `nn.Conv2d`, `nn.BatchNorm2d`, `nn.ReLU`, `nn.MaxPool2d` from scratch
- [ ] Implement a custom `Dataset` class for any tabular CSV dataset
- [ ] Use `torch.no_grad()` correctly during evaluation
- [ ] Load and save a model with `state_dict` (both saving and loading)
- [ ] Implement mixed precision training with `autocast` and `GradScaler`
- [ ] Use TensorBoard or W&B to log loss, accuracy, learning rate, and gradient norms
- [ ] Add gradient clipping to a training loop
- [ ] Write a custom `collate_fn` for variable-length sequences

---

### 3. Coding Challenges

**Challenge A — Full Training Pipeline**
```python
# Build a complete, production-quality training script for CIFAR-10:
# Requirements:
#   - ResNet-18 style architecture (implement residual blocks yourself)
#   - Mixed precision training (autocast + GradScaler)
#   - CosineAnnealingLR scheduler
#   - Gradient clipping (max_norm=1.0)
#   - W&B logging: loss, accuracy, lr, gradient norm per step
#   - Checkpoint saving: save best model by val accuracy
#   - Early stopping if no improvement for 10 epochs
#   - Final test accuracy must exceed 90%
```

**Challenge B — Custom Dataset**
```python
# Build a custom PyTorch Dataset for text classification:
# - Reads from a CSV with columns: text, label
# - Tokenises using a simple whitespace tokeniser
# - Builds vocabulary from training data only
# - Converts tokens to integer IDs with padding to max_len
# - Returns: (token_ids tensor, attention_mask tensor, label tensor)
# - Implement collate_fn that handles variable-length sequences
# - Test with DataLoader(batch_size=32, num_workers=2)
```

**Challenge C — Debug This Training Loop**
```python
# The following training loop has 5 deliberate bugs.
# Find and fix all 5 without running it first.
for epoch in range(epochs):
    for batch in train_loader:
        X, y = batch
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()    # Bug: missing zero_grad
    
    # Evaluation
    val_losses = []
    for batch in val_loader:
        X, y = batch
        output = model(X)  # Bug: no no_grad context
        val_loss = criterion(output, y)
        val_losses.append(val_loss)  # Bug: appending tensor, not .item()
    
    print(f"Val loss: {sum(val_losses)}")  # Bug: summing tensors
    scheduler.step(sum(val_losses))        # Bug: model in train mode during val
```

---

### 4. Mini Project

**Character Language Model in PyTorch**: Build a character-level RNN language model:
- Architecture: Embedding → LSTM (2 layers) → Linear → softmax
- Train on Shakespeare text (~1MB)
- Implement the training loop completely from scratch (no Lightning, no HuggingFace)
- Generate text with temperature sampling
- Plot: training and validation loss curves
- Save/load checkpoint, resume training from checkpoint

---

### 5. Capstone Project

**ResNet Image Classifier** (Project 7 from the roadmap, but now done fully):
- Implement ResNet-18 from scratch: BasicBlock with skip connections
- Mixed precision training with `autocast`
- W&B experiment tracking with full logging
- Comparison: with and without residual connections (show training curves)
- Hyperparameter sweep: learning rate × batch size grid (6 combinations)
- Achieve >93% CIFAR-10 test accuracy
- Export to ONNX and verify inference matches PyTorch output

---

### 6. Interview Questions

**Beginner**

1. **Q: What is a PyTorch tensor and how is it different from a NumPy array?**
   A: Both are N-dimensional arrays. Key differences: (1) tensors can live on GPU; (2) tensors track gradient computations; (3) PyTorch tensors are the primary unit of computation in neural networks. Most NumPy operations have PyTorch equivalents.

2. **Q: What does optimizer.zero_grad() do and why must it be called?**
   A: It sets all parameter gradients to zero (or None). PyTorch accumulates gradients by default — each `.backward()` call *adds* to existing gradients. Without zeroing, gradients from previous batches contaminate the current batch's gradients.

3. **Q: What is the difference between model.train() and model.eval()?**
   A: `train()` enables dropout (random masking) and uses batch statistics in BatchNorm. `eval()` disables dropout and switches BatchNorm to use running statistics. Always call `eval()` during validation and inference; always call `train()` before the training loop.

4. **Q: How do you move a model and data to GPU?**
   A: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. Then `model = model.to(device)` and in the loop: `X, y = X.to(device), y.to(device)`. Tensors and models must be on the same device.

5. **Q: What is `torch.no_grad()` and when is it used?**
   A: A context manager that disables gradient computation. Used during evaluation and inference — no gradients are needed, so this reduces memory usage and speeds up computation. Always wrap validation and prediction loops in `with torch.no_grad():`.

6. **Q: How do you save and load a PyTorch model?**
   A: Save: `torch.save(model.state_dict(), 'model.pt')`. Load: `model.load_state_dict(torch.load('model.pt'))`. Always save the `state_dict` (parameters), not the entire model object — the model object contains code references that may break across environments.

7. **Q: What is a DataLoader and what does num_workers do?**
   A: DataLoader wraps a Dataset and handles batching, shuffling, and parallel data loading. `num_workers` specifies how many worker processes to use for data loading in parallel with training. Set to 4–8 on machines with many CPU cores to avoid GPU starvation.

**Intermediate**

8. **Q: What is mixed precision training and how does it speed up training?**
   A: Mixed precision uses FP16 (16-bit float) for most operations but FP32 for accumulations. Benefits: (1) tensors use half the memory → larger batches; (2) modern GPUs (A100, V100) have Tensor Cores optimised for FP16 → 2-8x faster. GradScaler prevents gradients from underflowing to zero in FP16.

9. **Q: What is the difference between `view` and `reshape` in PyTorch?**
   A: `view` requires contiguous memory — it returns a view (no data copy). `reshape` works on non-contiguous tensors — it may copy data if required. Prefer `view` for efficiency; use `reshape` when you're unsure about contiguity (after `permute`, `transpose`).

10. **Q: How does torch.compile work and when should you use it?**
    A: `torch.compile` traces the computation graph and compiles it to optimised machine code using TorchDynamo + TorchInductor. Provides 10-50% speedup with a one-time compilation cost. Use for training runs that will run for many hours. Don't use for development/debugging — compilation makes stack traces harder to read.

11. **Q: What is gradient accumulation and why is it needed?**
    A: Gradient accumulation runs N "micro-batches" without calling `optimizer.step()`, only stepping after N batches. This simulates a batch size of N × micro_batch_size with limited GPU memory. Used when target batch size doesn't fit in VRAM.

12. **Q: What is pin_memory in DataLoader and when does it help?**
    A: `pin_memory=True` allocates data in page-locked (pinned) host memory, enabling faster CPU-to-GPU transfers. Useful when the data loading is a bottleneck. Only effective with CUDA — use with `non_blocking=True` in `.to(device)`.

13. **Q: How would you implement multi-GPU training in PyTorch?**
    A: The simplest approach: `torch.nn.DataParallel(model)` splits each batch across GPUs. Better: `torch.nn.parallel.DistributedDataParallel` (DDP) — each GPU gets its own process, no inter-GPU bottleneck. DDP is preferred for large-scale training. Use `torchrun` to launch DDP.

**Advanced**

14. **Q: Explain the difference between DataParallel and DistributedDataParallel.**
    A: DataParallel: single process, multiple threads, master GPU copies model, splits batch, collects gradients — bottlenecked by the master GPU and GIL. DDP: multiple processes, each owns one GPU, all-reduce communication for gradient synchronisation — scales linearly with GPU count.

15. **Q: How does autocast choose between FP16 and FP32 for different operations?**
    A: Operations are classified as: (1) "amp-safe" (matmul, conv → FP16), (2) "requires full precision" (softmax, LayerNorm, loss functions → FP32). The autocast context applies FP16 to safe ops automatically. You never need to manually cast tensors.

16. **Q: What is a custom collate_fn and why might you need one for NLP tasks?**
    A: The default collate_fn stacks tensors of equal shape. For variable-length sequences (sentences of different lengths), you need custom collation: pad shorter sequences, create attention masks, batch them appropriately. This is exactly what HuggingFace's `DataCollatorWithPadding` does internally.

17. **Q: Explain how nn.Module's forward method differs from __call__.**
    A: `__call__` (what runs when you do `model(x)`) calls `forward` but also runs registered hooks: forward pre-hooks, forward hooks, and full forward-hooks. This is how gradient checkpointing, activation hooks, and feature extraction are implemented. Never override `__call__`, only `forward`.

18. **Q: What is a PyTorch hook and give a practical ML use case?**
    A: Hooks are callbacks registered on modules or tensors. `register_forward_hook(fn)` calls fn(module, input, output) during every forward pass. Use cases: (1) extract intermediate layer activations for visualisation; (2) implement gradient surgery; (3) monitor gradient statistics per layer.

19. **Q: How would you profile a PyTorch training loop to find the bottleneck?**
    A: Use `torch.profiler.profile`: wrap the training loop, trace CPU + CUDA operations, export to TensorBoard or Chrome trace. Common bottlenecks: (1) data loading (slow CPU workers → increase num_workers); (2) CPU-GPU transfers (batch on CPU before transfer); (3) small kernel launches (use torch.compile); (4) memory bandwidth (use flash attention or larger batch sizes).

20. **Q: What is gradient checkpointing and when is it used?**
    A: Gradient checkpointing discards intermediate activations during the forward pass and recomputes them during backpropagation. Reduces memory from O(depth) to O(sqrt(depth)) at the cost of ~33% more computation. Essential for fine-tuning very large models on limited VRAM. PyTorch: `torch.utils.checkpoint.checkpoint_sequential`.

---

### 7. Self-Assessment Quiz

- [ ] Write the 7-step PyTorch training loop from memory.
- [ ] What is the shape of the output of `nn.Linear(784, 256)` given input shape `(32, 784)`?
- [ ] What happens if you forget `optimizer.zero_grad()`?
- [ ] What is the difference between `loss.backward()` and `torch.autograd.grad()`?
- [ ] How do you freeze layers in a PyTorch model?
- [ ] What does `model.parameters()` return?
- [ ] What is the purpose of `state_dict` vs saving the entire model?
- [ ] What is `torch.einsum('bnh,hd->bnd', x, W)` computing?
- [ ] How do you detect if your DataLoader is a bottleneck?
- [ ] What is gradient explosion and how do you detect it?
- [ ] Write a custom `Dataset.__getitem__` for a CSV file with text and labels.
- [ ] What is the difference between `nn.CrossEntropyLoss` and `nn.NLLLoss`?
- [ ] When should you use `torch.sigmoid` vs `nn.Sigmoid`?
- [ ] What is `nn.Embedding` and what is its `weight` shape?
- [ ] What does `contiguous()` do and when is it needed?
- [ ] What is the default initialisation of `nn.Linear`?
- [ ] How do you implement weight tying in a language model?
- [ ] What is `torchvision.transforms.Compose` and give 3 transforms you'd use for CIFAR-10?
- [ ] What is the difference between `torch.Tensor.item()` and `torch.Tensor.detach()`?
- [ ] How do you get the output shape of a model without running data through it?
- [ ] What does `model.half()` do?
- [ ] What is the purpose of `torch.manual_seed(42)`?
- [ ] What is `ReduceLROnPlateau` and when does it reduce the learning rate?
- [ ] What is `nn.ModuleList` vs a Python list of modules?
- [ ] What is `torch.inference_mode()` vs `torch.no_grad()`?

**Scoring**: 22–25 ✅ = Ready. 17–21 = Review weak areas. Below 17 = Spend more time on Phase 3b.

---

### 8. Common Mistakes

| Mistake | Why It Happens | How to Avoid |
|---------|---------------|--------------|
| Forgetting `zero_grad()` | It's easy to miss | Put it as step 1 in every training loop; add a comment |
| Not calling `model.eval()` during validation | train mode is default | Always pair: `model.train()` before train loop, `model.eval()` before val loop |
| Moving data to GPU inside the model | Seems logical | Always move data in the training loop; models should be device-agnostic |
| Accumulating tensors (not `.item()`) in loss lists | Leaks memory | Always call `loss.item()` when recording losses; tensors keep their computation graph |
| Using `model.state_dict()` but saving with wrong map_location | File loads on wrong device | Always specify `map_location=device` when loading checkpoints |
| Not shuffling training data | Easy oversight | Always `DataLoader(shuffle=True)` for training; never shuffle validation |
| Forgetting `retain_graph=True` for multiple backward passes | Needed for GANs, meta-learning | Know that the default frees the graph after one `.backward()` |
| Not detaching losses from the graph for logging | Memory leak | `loss.detach().item()` or `loss.item()` when logging |

---

### 9. Readiness Criteria

You are ready for Phase 4 when **all** of the following are true:

- [ ] I can write the 7-step training loop from memory in under 5 minutes
- [ ] I trained a CNN to >93% on CIFAR-10 with mixed precision
- [ ] I implemented the character-level LSTM language model (mini project)
- [ ] I found and fixed all 5 bugs in the Coding Challenge C
- [ ] I scored 22/25 or higher on the Self-Assessment Quiz
- [ ] I can answer at least 16/20 Interview Questions correctly
- [ ] I understand exactly what `.backward()` does under the hood (from Phase 3a)

---

### 10. Revision Summary

```
THE 7-STEP TRAINING LOOP
─────────────────────────────────────────────────────
1. optimizer.zero_grad()          # clear accumulated gradients
2. output = model(X)              # forward pass
3. loss = criterion(output, y)    # compute loss
4. loss.backward()                # backprop: compute all gradients
5. clip_grad_norm_(params, 1.0)   # optional: clip gradients
6. optimizer.step()               # update parameters
7. scheduler.step()               # update learning rate (may be per epoch)

KEY RULES
─────────────────────────────────────────────────────
model.train()  → before training loop (enables dropout + batch stats)
model.eval()   → before validation (disables dropout + uses running stats)
no_grad()      → wrap all evaluation and inference
.item()        → convert single-element tensor to Python scalar
state_dict()   → save model weights; load_state_dict() to restore

MIXED PRECISION
─────────────────────────────────────────────────────
scaler = GradScaler()
with autocast():
    loss = model(X)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

PERFORMANCE
─────────────────────────────────────────────────────
num_workers=4          → parallel data loading (avoid GPU starvation)
pin_memory=True        → faster CPU→GPU transfer
torch.compile(model)   → kernel fusion, 10-50% speedup
gradient_accumulation  → simulate large batch with small VRAM
```

---

### 11. Next Phase Prerequisites

**What Phase 4 (NLP & Transformers) requires from Phase 3b:**

| Phase 3b Skill | How Phase 4 Uses It |
|---------------|---------------------|
| `nn.Module` and `forward()` | Every transformer component is an `nn.Module` |
| `nn.Embedding` | The first layer of every language model |
| Custom Dataset + DataLoader | Text datasets, variable-length sequences, attention masks |
| Training loop | GPT training loop is identical — just different architecture |
| `torch.einsum` | Attention computation: Q @ K^T is `einsum('bqd,bkd->bqk', Q, K)` |
| Gradient clipping | Essential for transformer training stability |
| Mixed precision | GPT training always uses FP16/BF16 |
| `model.eval()` and `torch.no_grad()` | Text generation (inference) |

**The critical dependency**: Phase 4 projects (especially Project 9: GPT from scratch) are PyTorch projects. The GPT architecture uses everything from Phase 3b: `nn.Module`, `nn.Embedding`, `nn.Linear`, `nn.LayerNorm`, `nn.Dropout`, the training loop, and gradient clipping. You need to be fluent before reading the first line of transformer code.

---

*Phase 3b | Part of the [GenAI Engineer Roadmap](./00_README.md)*
