# Phase 4b: Transformers
**Months 10–11 | Difficulty: 9/10 | Label: 🔴 Must Learn (Most Important Phase)**

> **Previous Phase**: [Phase 4a — NLP Fundamentals](./05_Phase4_Part1_NLP_Fundamentals.md)  
> **Next Phase**: [Phase 5a — LLMs & Prompt Engineering](./06_Phase5_Part1_LLMs_and_Prompt_Engineering.md)

---

## Table of Contents

- [Phase Overview](#phase-overview)
- [Why Transformers Are The Most Important Topic](#why-transformers-are-the-most-important-topic)
- [Topic 1: The Transformer Architecture](#topic-1-the-transformer-architecture)
- [Topic 2: Scaled Dot-Product Attention](#topic-2-scaled-dot-product-attention)
  - [Query, Key, Value Intuition](#query-key-value-intuition)
  - [Mathematical Derivation](#mathematical-derivation)
  - [Full Implementation](#full-implementation)
- [Topic 3: Multi-Head Attention](#topic-3-multi-head-attention)
- [Topic 4: Positional Encoding](#topic-4-positional-encoding)
  - [Absolute Positional Encoding](#absolute-positional-encoding)
  - [Rotary Positional Encoding (RoPE)](#rotary-positional-encoding-rope)
- [Topic 5: Transformer Block](#topic-5-transformer-block)
  - [Feed-Forward Network](#feed-forward-network)
  - [Residual Connections and Layer Norm](#residual-connections-and-layer-norm)
  - [Pre-Norm vs. Post-Norm](#pre-norm-vs-post-norm)
- [Topic 6: Encoder vs. Decoder vs. Encoder-Decoder](#topic-6-encoder-vs-decoder-vs-encoder-decoder)
  - [Encoder-Only: BERT](#encoder-only-bert)
  - [Decoder-Only: GPT](#decoder-only-gpt)
  - [Encoder-Decoder: T5](#encoder-decoder-t5)
  - [Causal Masking](#causal-masking)
- [Topic 7: KV Cache](#topic-7-kv-cache)
- [Topic 8: Full GPT Implementation in PyTorch](#topic-8-full-gpt-implementation-in-pytorch)
- [Topic 9: Tokenization in Practice](#topic-9-tokenization-in-practice)
- [Resources](#resources)
- [The Essential Paper](#the-essential-paper)
- [Projects](#projects)
- [Common Mistakes](#common-mistakes)
- [Mastery Checklist](#mastery-checklist)

---

## Phase Overview

| Attribute | Details |
|-----------|---------|
| **Duration** | Months 10–11 (4 weeks) |
| **Daily Time** | 1.5–2 hours |
| **Difficulty** | 9/10 |
| **Label** | 🔴 Must Learn |
| **Prerequisites** | Phase 4a (attention precursor, tokenization), Phase 3b (PyTorch nn.Module) |
| **Outcome** | Can implement transformers from scratch; understands QKV, causal masking, positional encoding, KV cache |

---

## Why Transformers Are The Most Important Topic

> *"Attention Is All You Need"* (Vaswani et al., 2017) is the most important AI paper ever written.

Every major AI system deployed today is a transformer:
- **GPT-4, Claude, Gemini** — language models
- **DALL-E 3, Stable Diffusion** — image generation (U-Net uses attention)
- **Whisper** — speech recognition
- **AlphaFold** — protein structure prediction
- **Codex** — code generation
- **CLIP** — vision-language models

If you deeply understand transformers at the implementation level, you can:
- Read any AI research paper
- Understand why models fail in specific ways
- Design novel architectures
- Debug training instability
- Optimize inference

**This is the phase where engineering intuition becomes exponentially more valuable than surface-level API knowledge.**

---

## Topic 1: The Transformer Architecture

### High-Level Structure

```
Input Text
    ↓
[Tokenization]
    ↓
[Token Embeddings]  +  [Positional Encoding]
                     ↓
           ┌─────────────────────────┐
           │   Transformer Block × N │
           │  ┌────────────────────┐ │
           │  │  Multi-Head Attn   │ │
           │  │  + Residual + LN   │ │
           │  ├────────────────────┤ │
           │  │  Feed-Forward Net  │ │
           │  │  + Residual + LN   │ │
           │  └────────────────────┘ │
           └─────────────────────────┘
                     ↓
           [Layer Norm]
                     ↓
           [Linear Projection]
                     ↓
           [Softmax] → Next Token Probabilities
```

**Key hyperparameters** (GPT-2 small as reference):

| Parameter | GPT-2 Small | GPT-2 Medium | GPT-2 Large | LLaMA-7B |
|-----------|:-----------:|:------------:|:-----------:|:--------:|
| `d_model` (n_embd) | 768 | 1024 | 1280 | 4096 |
| `n_layers` | 12 | 24 | 36 | 32 |
| `n_heads` | 12 | 16 | 20 | 32 |
| `d_head` | 64 | 64 | 64 | 128 |
| `d_ff` | 3072 | 4096 | 5120 | 11008 |
| `vocab_size` | 50257 | 50257 | 50257 | 32000 |
| `context_length` | 1024 | 1024 | 1024 | 4096 |
| **Parameters** | **117M** | **345M** | **762M** | **6.7B** |

---

## Topic 2: Scaled Dot-Product Attention

### Query, Key, Value Intuition

The Q/K/V framework is a soft, differentiable database lookup:

```
Database analogy:
  Key   = database keys (what information is available)
  Value = database values (the actual information)
  Query = what you're looking for

Traditional database: exact match → return value
Attention: soft match (similarity) → weighted average of values
```

In a transformer processing the sentence "**The cat sat on the mat**":

- Each token produces a **Query**: "What information do I need?"
- Each token produces a **Key**: "What information do I have?"
- Each token produces a **Value**: "What information I will contribute if attended to"

When processing the word "sat":
- Its Query says: "I need to know the subject and location"
- "cat"'s Key matches the Query well → high attention weight
- "mat"'s Key also matches → moderate attention weight
- "The"'s Key barely matches → low attention weight

The final representation of "sat" = weighted mix of all tokens' Values.

**This is the key insight**: every token can directly access information from every other token in the sequence, regardless of distance. No vanishing gradient problem.

---

### Mathematical Derivation

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Why divide by $\sqrt{d_k}$?**

When $d_k$ is large, the dot products $QK^T$ grow in magnitude (sum of $d_k$ products). Large values push softmax into regions with tiny gradients. Dividing by $\sqrt{d_k}$ keeps the variance of the dot products approximately 1.

**Proof**: If Q and K have components drawn from $\mathcal{N}(0,1)$, then each dot product is a sum of $d_k$ terms with variance 1. Total variance = $d_k$. Standard deviation = $\sqrt{d_k}$. Dividing by $\sqrt{d_k}$ normalizes the variance to 1.

---

### Full Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Core attention mechanism.
    
    Shapes:
    - Q: (batch, n_heads, seq, d_k)
    - K: (batch, n_heads, seq, d_k)
    - V: (batch, n_heads, seq, d_v)  [d_v = d_k in most models]
    - mask: (batch, 1, seq, seq) or (batch, n_heads, seq, seq)
    - output: (batch, n_heads, seq, d_v)
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = Q.size(-1)
        
        # ① Compute attention scores: (batch, heads, seq_q, seq_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # ② Apply mask (causal or padding)
        if mask is not None:
            # Where mask == 0, set score to -inf so softmax → ~0
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # ③ Softmax → attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # ④ Dropout on attention weights (regularization)
        attention_weights = self.dropout(attention_weights)
        
        # ⑤ Weighted sum of values: (batch, heads, seq_q, d_v)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
```

---

## Topic 3: Multi-Head Attention

Instead of one attention head with $d_{model}$ dimensions, use $h$ heads each with $d_k = d_{model}/h$ dimensions.

**Why multiple heads?**
Each head can attend to different types of relationships:
- Head 1: syntactic relationships (subject-verb)
- Head 2: coreference ("it" → "the cat")
- Head 3: positional proximity
- Head 4: semantic similarity

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention from scratch.
    
    This is the exact implementation used in all transformers.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads    # Dimension per head
        
        # Four projection matrices: Q, K, V, and output
        # Note: typically implemented as single matrices for efficiency
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Projects to all heads' Q
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # Output projection
        
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split d_model dimension into (n_heads, d_k).
        (batch, seq, d_model) → (batch, n_heads, seq, d_k)
        """
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq, d_k)
    
    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge (n_heads, d_k) back to d_model.
        (batch, n_heads, seq, d_k) → (batch, seq, d_model)
        """
        batch_size, n_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2)  # (batch, seq, n_heads, d_k)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(
        self,
        x: torch.Tensor,           # Input: (batch, seq, d_model)
        context: torch.Tensor = None,  # For cross-attention (encoder output)
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # For self-attention, context = x
        # For cross-attention, context = encoder output
        if context is None:
            context = x
        
        # Project inputs to Q, K, V
        Q = self.W_q(x)        # (batch, seq, d_model)
        K = self.W_k(context)  # (batch, ctx_seq, d_model)
        V = self.W_v(context)  # (batch, ctx_seq, d_model)
        
        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch, n_heads, seq, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Compute attention for all heads simultaneously
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        # attn_output: (batch, n_heads, seq, d_k)
        
        # Merge heads back
        attn_output = self.merge_heads(attn_output)  # (batch, seq, d_model)
        
        # Final output projection
        output = self.W_o(attn_output)  # (batch, seq, d_model)
        
        return output, attn_weights

# Verify shapes
d_model, n_heads = 768, 12
attn = MultiHeadAttention(d_model, n_heads)

batch_size, seq_len = 4, 128
x = torch.randn(batch_size, seq_len, d_model)

output, weights = attn(x)
print(f"Input:  {x.shape}")           # (4, 128, 768)
print(f"Output: {output.shape}")      # (4, 128, 768)
print(f"Weights: {weights.shape}")    # (4, 12, 128, 128)

# Count parameters in MHA
n_params = sum(p.numel() for p in attn.parameters())
print(f"Parameters in MHA: {n_params:,}")  # 4 * d_model * d_model = 4 * 768^2 ≈ 2.4M
```

---

## Topic 4: Positional Encoding

Attention is **permutation-invariant** — it doesn't know whether "cat sat on mat" or "mat on sat cat". Positional encoding injects position information.

### Absolute Positional Encoding

The original transformer paper uses sinusoidal encoding:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

class SinusoidalPositionalEncoding(nn.Module):
    """
    Absolute positional encoding from "Attention Is All You Need".
    
    Key properties:
    1. Each position gets a unique pattern of sin/cos values
    2. The distance between any two positions can be computed as a linear function
    3. Works for sequences longer than training (extrapolation)
    4. No learnable parameters
    """
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Compute positional encodings
        PE = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # Frequency terms: 1 / 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        PE[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        PE[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        
        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('PE', PE.unsqueeze(0))  # (1, max_seq_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.PE[:, :x.size(1), :]
        return self.dropout(x)

# Visualize positional encodings
d_model = 128
pos_enc = SinusoidalPositionalEncoding(d_model)
PE_matrix = pos_enc.PE[0].numpy()  # (max_seq_len, d_model)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(PE_matrix[:50, :64], aspect='auto', cmap='RdBu')
plt.colorbar()
plt.xlabel('Embedding Dimension')
plt.ylabel('Position')
plt.title('Sinusoidal Positional Encoding\n(First 50 positions, 64 dims)')

plt.subplot(1, 2, 2)
# Show that similar positions have similar encodings
# and the similarity decreases with distance (which is what we want)
similarities = PE_matrix[:20] @ PE_matrix[:20].T
plt.imshow(similarities, cmap='Blues')
plt.colorbar()
plt.title('Position Similarity Matrix\n(Nearby positions more similar)')
plt.tight_layout()
```

### Rotary Positional Encoding (RoPE)

RoPE is used by LLaMA, Mistral, Falcon, and most modern LLMs. It encodes position by *rotating* the query and key vectors:

```python
class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) — used by LLaMA, Mistral, Falcon.
    
    Key advantages over sinusoidal PE:
    1. Applied to Q and K (not token embeddings) — encodes relative position in attention
    2. Works better for longer sequences
    3. More naturally expresses relative position
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        # Inverse frequencies for each dimension pair
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Pre-compute for efficiency
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)
        
        self.register_buffer('cos', emb.cos()[None, None, :, :])  # (1, 1, seq, dim)
        self.register_buffer('sin', emb.sin()[None, None, :, :])
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of x to create the rotation."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(
        self,
        q: torch.Tensor,   # (batch, n_heads, seq, d_k)
        k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]
        
        # Apply rotation: x_rotated = x * cos + rotate_half(x) * sin
        q_rotated = (q * cos) + (self.rotate_half(q) * sin)
        k_rotated = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_rotated, k_rotated
```

---

## Topic 4b: Attention Efficiency — MQA and GQA

> This topic is distinct from positional encoding but belongs here, after multi-head attention, because it directly modifies the head structure.

### The KV Cache Memory Problem

In standard Multi-Head Attention (MHA) with `n_heads` heads, the KV cache grows as:

$$\text{KV cache bytes} = 2 \times n_{layers} \times n_{heads} \times d_{head} \times T \times B \times \text{dtype\_bytes}$$

For LLaMA-3-70B with batch=8, sequence length=4096, float16:
- $2 \times 80 \times 64 \times 128 \times 4096 \times 8 \times 2 = \textbf{68 GB}$ — just for the KV cache.

**Solution**: Share key and value projections across groups of query heads.

### MHA vs. MQA vs. GQA

```
Standard MHA (all heads independent):
  Q₁  K₁  V₁   Q₂  K₂  V₂   Q₃  K₃  V₃   Q₄  K₄  V₄
  (4 query heads, 4 KV pairs)

MQA — Multi-Query Attention (1 shared KV):
  Q₁  Q₂  Q₃  Q₄
        K   V
  (4 query heads, 1 KV pair → 4× smaller KV cache)

GQA — Grouped-Query Attention (G shared KV groups, 1 < G < H):
  Q₁  Q₂  |  Q₃  Q₄
   K₁  V₁  |   K₂  V₂
  (4 query heads, 2 KV groups → 2× smaller KV cache)
```

| Variant | KV Heads | KV Cache Size | Quality |
|---------|:--------:|:-------------:|:-------:|
| MHA (standard) | = n_heads | Baseline | Best |
| GQA | < n_heads, > 1 | n_heads/n_kv_heads × smaller | Near-MHA |
| MQA | 1 | n_heads × smaller | Slightly worse |

### PyTorch Implementation of GQA

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention as used in LLaMA-3, Mistral, Gemma.
    n_kv_heads divides n_heads evenly (e.g., 32 query heads, 8 KV heads = 4 Q per KV group).
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads   # Q heads per KV group
        self.d_head = d_model // n_heads
        
        # Query: full n_heads; K/V: only n_kv_heads — this is the savings
        self.W_q = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_head, d_model, bias=False)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape
        
        # Standard projections
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)     # (B, n_heads, T, d_head)
        k = self.W_k(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)  # (B, n_kv_heads, T, d_head)
        v = self.W_v(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)  # (B, n_kv_heads, T, d_head)
        
        # Expand KV to match query head count by repeating each KV head n_groups times
        # (B, n_kv_heads, T, d_head) → (B, n_heads, T, d_head)
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)
        
        # Standard scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)   # (B, n_heads, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        return self.W_o(out)

# Verify parameter count reduction
mha = GroupedQueryAttention(d_model=512, n_heads=8, n_kv_heads=8)  # Standard MHA
gqa = GroupedQueryAttention(d_model=512, n_heads=8, n_kv_heads=2)  # GQA
mqa = GroupedQueryAttention(d_model=512, n_heads=8, n_kv_heads=1)  # MQA

def count_kv_params(m): return sum(p.numel() for n, p in m.named_parameters() if 'W_k' in n or 'W_v' in n)
print(f"MHA KV params: {count_kv_params(mha):,}")   # 131,072
print(f"GQA KV params: {count_kv_params(gqa):,}")   # 32,768  (4× fewer)
print(f"MQA KV params: {count_kv_params(mqa):,}")   # 16,384  (8× fewer)
```

### Which Models Use Which Variant

| Model | Attention | n_heads | n_kv_heads | Why |
|-------|-----------|:-------:|:----------:|-----|
| GPT-2, LLaMA-2-7B | MHA | 32 | 32 | Original design |
| LLaMA-3-8B | GQA | 32 | 8 | 4× KV cache reduction |
| LLaMA-3-70B | GQA | 64 | 8 | 8× KV cache reduction |
| Mistral-7B | GQA | 32 | 8 | Same as LLaMA-3 |
| Gemma-7B | MHA | 16 | 16 | Smaller model, less pressure |
| Falcon-7B | MQA | 71 | 1 | Extreme KV reduction |
| PaLM | MQA | 48 | 1 | Google's original choice |

> **Interview answer**: "GQA was introduced because MHA's KV cache scales as `O(n_heads × seq_len × batch)`. For a 70B model with 64 heads, this becomes prohibitive at long context or large batches. GQA with 8 KV heads reduces this 8×, with minimal quality loss compared to MQA's more aggressive 64× reduction."

---

## Topic 5: Transformer Block

### Feed-Forward Network

Each transformer block has an FFN that applies a nonlinear transformation per-token:

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

The intermediate dimension is typically $4 \times d_{model}$ (3072 for GPT-2 small with $d_{model}=768$).

```python
class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Applied independently to each position (each token).
    This is where the model does most of its "thinking" — 
    storing factual knowledge and applying transformations.
    
    Note: ~⅔ of transformer parameters are in FFN layers.
    """
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # Standard: 4x expansion
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),            # GELU for GPT-style; SwiGLU for LLaMA
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU FFN — used by LLaMA, PaLM, Mistral, most modern LLMs.
    
    SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)
    
    Key difference: uses a gating mechanism (element-wise multiplication)
    The gate controls how much information flows through.
    Empirically outperforms standard GELU FFN.
    """
    def __init__(self, d_model: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * d_model / 3)  # LLaMA convention
        
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))   # Swish activation on gate
        up = self.up_proj(x)
        return self.down_proj(gate * up)   # Gated multiplication
```

### Residual Connections and Layer Norm

```python
class TransformerBlock(nn.Module):
    """
    Complete transformer block.
    
    Two styles:
    - Post-norm (original paper): x = LayerNorm(x + sublayer(x))
    - Pre-norm (modern, LLaMA): x = x + sublayer(LayerNorm(x))  ← more stable
    
    Most modern LLMs use pre-norm.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        pre_norm: bool = True  # True for modern models (LLaMA), False for original
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        
        if self.pre_norm:
            # Pre-norm (LLaMA / modern style):
            # Normalize BEFORE sublayer, add residual after
            attn_out, _ = self.attention(self.ln1(x), mask=mask)
            x = x + self.dropout(attn_out)    # Residual connection
            
            ffn_out = self.ffn(self.ln2(x))
            x = x + self.dropout(ffn_out)      # Residual connection
        else:
            # Post-norm (original paper):
            # Apply sublayer, add residual, THEN normalize
            attn_out, _ = self.attention(x, mask=mask)
            x = self.ln1(x + self.dropout(attn_out))
            
            ffn_out = self.ffn(x)
            x = self.ln2(x + self.dropout(ffn_out))
        
        return x

# Why residual connections are critical (again):
# Without: gradient = product of N layer Jacobians → exponentially small or large
# With:    gradient = gradient + product → gradient of 1 is always available
#          This is the "gradient highway"
```

### Pre-Norm vs. Post-Norm

| | Original Transformer | GPT-2 | Modern LLMs (LLaMA, Mistral) |
|--|:-:|:-:|:-:|
| Style | Post-norm | Post-norm | Pre-norm |
| Training stability | Lower | Moderate | Higher |
| Performance | Good | Better | Best |

---

## Topic 6: Encoder vs. Decoder vs. Encoder-Decoder

### Encoder-Only: BERT

- Processes sequence **bidirectionally** (each token attends to all others)
- No causal masking
- Best for: classification, NER, sentence embeddings
- Examples: BERT, RoBERTa, DeBERTa, sentence-transformers

```python
# BERT's training objective: Masked Language Modeling (MLM)
# Randomly mask 15% of tokens, predict the masked tokens
# This forces bidirectional context learning

# Example:
# Input:  "The [MASK] sat on the mat"
# Target: "cat"
# Model must look at BOTH left and right context to predict "cat"
```

### Decoder-Only: GPT

- Each position attends only to **previous** positions (causal masking)
- Generates text autoregressively (one token at a time)
- Best for: text generation, language modeling, chat, instruction following
- Examples: GPT-2/3/4, LLaMA, Mistral, Falcon, Claude

### Encoder-Decoder: T5, BART

- Encoder processes the input with full (bidirectional) attention
- Decoder generates output attending to encoder output AND previous output tokens
- Best for: translation, summarization, question answering
- Examples: T5, BART, mT5, FLAN-T5

### Causal Masking

```python
def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal (autoregressive) mask.
    
    Each position can only attend to itself and previous positions.
    This is what makes GPT an autoregressive model.
    
    Mask[i, j] = 1 if position i can attend to position j
               = 0 otherwise (will become -inf in attention)
    """
    # Upper triangular matrix with 1s below diagonal
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    # Shape: (seq_len, seq_len)
    # Add batch and head dimensions for broadcasting
    return mask.view(1, 1, seq_len, seq_len)  # (1, 1, seq, seq)

# Visualize causal mask for 6 tokens
mask = create_causal_mask(6, torch.device('cpu'))
print("Causal mask (1 = can attend, 0 = cannot attend):")
print(mask[0, 0].numpy().astype(int))

# Output:
# [[1 0 0 0 0 0]   # Token 0 can only see itself
#  [1 1 0 0 0 0]   # Token 1 can see tokens 0,1
#  [1 1 1 0 0 0]   # Token 2 can see tokens 0,1,2
#  [1 1 1 1 0 0]   # etc.
#  [1 1 1 1 1 0]
#  [1 1 1 1 1 1]]  # Token 5 can see all tokens
```

---

## Topic 7: KV Cache

The KV Cache is critical for inference efficiency. Without understanding it, you cannot reason about LLM serving costs and latency.

**The problem**: During autoregressive generation, we generate one token at a time. For each new token, we compute K and V for the entire sequence — but we already computed K and V for previous tokens!

**The solution**: Cache K and V from previous steps and reuse them.

```python
class CachedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with KV Cache for efficient autoregressive inference.
    
    Without KV cache: O(seq²) computation for each new token
    With KV cache:    O(seq) computation for each new token
    
    Memory cost: O(n_layers * batch_size * seq_len * d_model) for cache
    This is why LLMs have a maximum context length — it's bounded by VRAM.
    """
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_seq_len = max_seq_len
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # KV Cache: pre-allocated for efficiency
        self.cache_k = None
        self.cache_v = None
        self.cache_len = 0
    
    def initialize_cache(self, batch_size: int, device: torch.device):
        """Allocate KV cache at the start of inference."""
        self.cache_k = torch.zeros(
            batch_size, self.n_heads, self.max_seq_len, self.d_k,
            device=device
        )
        self.cache_v = torch.zeros_like(self.cache_k)
        self.cache_len = 0
    
    def forward_with_cache(
        self,
        x: torch.Tensor,   # New token(s): (batch, new_seq=1, d_model)
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Compute Q, K, V for new tokens only
        Q = self._split_heads(self.W_q(x))  # (batch, heads, 1, d_k)
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))
        
        # Append to cache
        pos = self.cache_len
        self.cache_k[:, :, pos:pos+1, :] = K
        self.cache_v[:, :, pos:pos+1, :] = V
        self.cache_len += 1
        
        # Attend over ENTIRE cache (all past + current tokens)
        K_full = self.cache_k[:, :, :self.cache_len, :]  # (batch, heads, seq, d_k)
        V_full = self.cache_v[:, :, :self.cache_len, :]
        
        # Compute attention (no mask needed — attending to all past)
        scores = torch.matmul(Q, K_full.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V_full)
        
        return self.W_o(self._merge_heads(output))
    
    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
    
    def _merge_heads(self, x):
        B, H, T, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

# Memory calculation for KV cache
def calculate_kv_cache_memory(
    n_layers: int, batch_size: int, seq_len: int, d_model: int, 
    bytes_per_param: int = 2  # float16
) -> float:
    """Calculate VRAM needed for KV cache."""
    # 2 (K+V) * n_layers * batch_size * seq_len * d_model
    bytes = 2 * n_layers * batch_size * seq_len * d_model * bytes_per_param
    return bytes / (1024**3)  # Convert to GB

# LLaMA-7B with context=4096, batch=1, float16
mem_gb = calculate_kv_cache_memory(
    n_layers=32, batch_size=1, seq_len=4096, d_model=4096, bytes_per_param=2
)
print(f"KV Cache for LLaMA-7B (ctx=4096, batch=1): {mem_gb:.2f} GB")
# ≈ 2.0 GB — significant portion of available VRAM
```

---

## Topic 8: Full GPT Implementation in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024   # max sequence length
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.1
    bias: bool = True

class GPT(nn.Module):
    """
    Complete GPT implementation.
    Follows Karpathy's nanoGPT — the clearest GPT implementation.
    
    Architecture:
    - Token embedding + positional embedding
    - N transformer blocks (decoder-only, causal masking)
    - Final layer norm
    - Language modeling head (projects to vocabulary)
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),    # Token embeddings
            'wpe': nn.Embedding(config.block_size, config.n_embd),    # Position embeddings
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([
                TransformerBlock(
                    d_model=config.n_embd,
                    n_heads=config.n_head,
                    dropout=config.dropout,
                    pre_norm=True
                )
                for _ in range(config.n_layer)
            ]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        
        # Language model head: projects to vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: share weights between token embedding and lm_head
        # Reduces parameters; standard practice in LLMs
        self.transformer['wte'].weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report parameters
        print(f"GPT parameters: {self.count_params()/1e6:.2f}M")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self,
        idx: torch.Tensor,          # Token indices: (batch, seq_len)
        targets: torch.Tensor = None  # For training: (batch, seq_len)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence too long ({T} > {self.config.block_size})"
        device = idx.device
        
        # Compute token + position embeddings
        tok_emb = self.transformer['wte'](idx)                  # (B, T, n_embd)
        pos_ids = torch.arange(T, device=device)
        pos_emb = self.transformer['wpe'](pos_ids)              # (T, n_embd)
        x = self.transformer['drop'](tok_emb + pos_emb)        # (B, T, n_embd)
        
        # Build causal mask
        mask = create_causal_mask(T, device)  # (1, 1, T, T)
        
        # Process through transformer blocks
        for block in self.transformer['h']:
            x = block(x, mask=mask)
        
        x = self.transformer['ln_f'](x)
        
        # Compute logits
        if targets is not None:
            # Training: compute loss for all positions
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Inference: only need logits for last position
            logits = self.lm_head(x[:, -1:, :])  # (B, 1, vocab_size)
            loss = None
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        
        temperature: controls randomness
          - 0.0: greedy (most likely token always)
          - 1.0: standard sampling
          - >1.0: more random
        top_k: only sample from top-k tokens
        top_p: only sample from tokens with cumulative probability < p
        """
        for _ in range(max_new_tokens):
            # Truncate to block_size if necessary
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]    # (B, vocab_size) — last token only
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, -1:]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_indices_to_remove] = float('-inf')
                logits.scatter_(-1, sorted_indices, sorted_logits)
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat([idx, next_token], dim=1)
        
        return idx


# Train a small GPT on Shakespeare
config = GPTConfig(
    vocab_size=65,      # Character level: 26 lower + 26 upper + punctuation
    block_size=256,
    n_embd=384,
    n_head=6,
    n_layer=6,
    dropout=0.2
)

model = GPT(config)
print(f"Model parameters: {model.count_params() / 1e6:.2f}M")  # ~10M
```

---

## Topic 9: Tokenization in Practice

```python
from transformers import AutoTokenizer
import torch

# Understand tokenization behavior for LLMs
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Special tokens
print(f"BOS: {tokenizer.bos_token}")  # <|endoftext|>
print(f"EOS: {tokenizer.eos_token}")  # <|endoftext|>
print(f"PAD: {tokenizer.pad_token}")  # None (GPT-2 has no pad token!)
# → Need to set pad token for batch training: tokenizer.pad_token = tokenizer.eos_token

# Encoding text
text = "The transformer architecture is revolutionary."
encoding = tokenizer(
    text,
    return_tensors='pt',
    padding=True,          # Pad to same length
    truncation=True,       # Truncate to max_length
    max_length=512,
    return_attention_mask=True
)

print(f"\nInput IDs: {encoding['input_ids']}")
print(f"Shape: {encoding['input_ids'].shape}")
print(f"Tokens: {[tokenizer.decode([t]) for t in encoding['input_ids'][0]]}")
print(f"Attention mask: {encoding['attention_mask']}")
# Attention mask: 1 for real tokens, 0 for padding

# Batch encoding (critical for training)
texts = [
    "Short sentence.",
    "This is a much longer sentence that will need to be padded to the same length as the others.",
    "Medium length sentence here."
]

batch = tokenizer(
    texts,
    padding='longest',    # Pad to longest in batch
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

print(f"\nBatch input_ids shape: {batch['input_ids'].shape}")  # (3, longest_len)
print(f"Attention masks:\n{batch['attention_mask']}")
# Rows with padding have 0s at the end

# Decoding
decoded = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
for original, decoded_text in zip(texts, decoded):
    print(f"  Original: {original[:50]}...")
    print(f"  Decoded:  {decoded_text[:50]}...")
```

---

## Resources

| Rank | Resource | Type | Cost | Why |
|------|----------|------|------|-----|
| 1 | [Andrej Karpathy "Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) | YouTube | Free | **THE most important video in AI.** Watch 3 times. Build GPT from scratch in 2 hours. |
| 2 | [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) | Blog | Free | Best visual explanation of attention. Read BEFORE Karpathy's video. |
| 3 | Attention Is All You Need (Vaswani et al., 2017) | Paper | Free | Read it. Re-read it after Karpathy. Then read it again. |
| 4 | NLP with Transformers (Tunstall, Werra, Wolf) | Book | ~$60 | By Hugging Face team. Best practical book on transformers. |
| 5 | [The Illustrated BERT (Jay Alammar)](https://jalammar.github.io/illustrated-bert/) | Blog | Free | Understand encoder-only models. |
| 6 | [Yannic Kilcher YouTube](https://www.youtube.com/@YannicKilcher) | YouTube | Free | Best paper explanation channel. Watch "Attention is All You Need". |
| 7 | [nanoGPT (Karpathy, GitHub)](https://github.com/karpathy/nanoGPT) | GitHub | Free | The cleanest GPT implementation. Study this code. |

### The Essential Paper

**Attention Is All You Need (Vaswani et al., 2017)**  
[arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

**How to read it**:
1. First pass: Read abstract, conclusion, Figure 1, Figure 2
2. Second pass (after Jay Alammar's blog): Read section 3 (Model Architecture) carefully
3. Third pass (after Karpathy's video): Re-read with implementation in mind

---

## Projects

### Project 9: GPT From Scratch (The Most Important Project)
**Difficulty**: 7/10 | **Time**: 2–3 weeks

Following Karpathy's tutorial exactly:
1. Implement scaled dot-product attention
2. Implement multi-head attention
3. Implement transformer block (attention + FFN + residual + LN)
4. Implement full GPT class
5. Train on Shakespeare (~1MB text file)
6. Generate sample text
7. Visualize attention weights

**Verification**: Your model should achieve perplexity < 2.0 on the training set after enough training.

---

### Project 10: Attention Visualization
**Difficulty**: 5/10 | **Time**: 1 week

Using a pre-trained BERT model from Hugging Face:
1. Extract attention weights for sample sentences
2. Visualize each attention head as a heatmap
3. Identify what each head seems to be attending to
4. Compare across layers (early layers vs. late layers)

```python
from transformers import BertModel, BertTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
model.eval()

text = "The cat sat on the mat because it was tired."
inputs = tokenizer(text, return_tensors='pt')
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

with torch.no_grad():
    outputs = model(**inputs)

# attentions: tuple of (n_layers,) each (batch, n_heads, seq, seq)
attentions = outputs.attentions

# Plot attention for layer 0, head 0
layer, head = 0, 0
attn_matrix = attentions[layer][0, head].numpy()

plt.figure(figsize=(10, 8))
sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap='Blues')
plt.title(f'Layer {layer}, Head {head} Attention')
plt.tight_layout()
plt.show()
```

---

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Not building from scratch | Surface-level understanding; can't debug | Follow Karpathy's video and implement line by line |
| Not understanding the scaling factor | Think it's arbitrary; can't explain to interviewers | Derive the variance argument (see mathematical derivation) |
| Confusing encoder and decoder architectures | Wrong architecture for the task | Remember: BERT=encoder(bidirectional), GPT=decoder(causal) |
| Not understanding KV cache | Can't reason about inference costs or serving | Calculate VRAM requirements for different model sizes |
| Not reading the original paper | Miss important details | Read "Attention Is All You Need" — it's only 11 pages |
| Skipping attention visualization | Miss intuition about what heads learn | Complete Project 10 |
| Using built-in `nn.Transformer` before understanding it | Black box; can't modify | Build from scratch first |

---

## Mastery Checklist

### Theory
- [ ] Can explain Q, K, V with the "database lookup" analogy
- [ ] Can derive WHY we divide by $\sqrt{d_k}$ (variance argument)
- [ ] Can explain why multiple heads are better than one wide head
- [ ] Understands the difference between encoder-only, decoder-only, and encoder-decoder
- [ ] Can explain what causal masking does and why decoder models need it
- [ ] Understands the KV cache and can calculate memory requirements
- [ ] Can explain pre-norm vs. post-norm and why pre-norm is preferred

### Implementation
- [ ] Implemented scaled dot-product attention from scratch in PyTorch
- [ ] Implemented multi-head attention from scratch
- [ ] Implemented causal mask
- [ ] Implemented full transformer block (attention + FFN + residuals + LN)
- [ ] Implemented full GPT and generated text
- [ ] Implemented sinusoidal positional encoding

### Reading
- [ ] Read "Attention Is All You Need" paper
- [ ] Read Jay Alammar's "The Illustrated Transformer"
- [ ] Read Jay Alammar's "The Illustrated GPT-2"
- [ ] Watched Karpathy "Let's build GPT from scratch" video

---

## Moving to Phase 5

**Before proceeding to [Phase 5a: LLMs & Prompt Engineering](./06_Phase5_Part1_LLMs_and_Prompt_Engineering.md), confirm:**

- [ ] Working character-level GPT that generates coherent text
- [ ] Attention visualization project complete
- [ ] Can explain every line of the GPT forward pass
- [ ] Read the original transformer paper

**Why Phase 5 comes next**: You now understand *how* transformers work internally. Phase 5 is about understanding large-scale transformers — how pretraining at scale produces emergent capabilities, and how to effectively use them as a practitioner.

---

## Phase Completion & Readiness Assessment

> This is the most important phase in the entire roadmap. Do not move to Phase 5 until you can implement a transformer from scratch without referring to any code.

---

### 1. Knowledge Checklist

**Attention Mechanism**
- [ ] Scaled dot-product attention formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
- [ ] Why divide by sqrt(d_k): what happens without it (dot products too large → softmax saturation)
- [ ] Q, K, V: what each represents semantically (query=what I want, key=what I have, value=what I give)
- [ ] Causal masking: why it's needed for autoregressive generation
- [ ] Multi-head attention: why multiple heads, how heads are split and merged
- [ ] Cross-attention vs. self-attention

**Positional Encoding**
- [ ] Why transformers need positional encoding (no recurrence)
- [ ] Sinusoidal encoding: formula, why sines and cosines of different frequencies
- [ ] Learned positional embeddings (GPT-2 style)
- [ ] Rotary Position Embeddings (RoPE): why they're better and how they work
- [ ] ALiBi: position bias in attention scores

**Transformer Architecture**
- [ ] Pre-norm vs. post-norm: which is more stable and why
- [ ] Encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5)
- [ ] Feed-forward network in transformer block: shape expansion and contraction
- [ ] SwiGLU activation (used in LLaMA): formula and why it outperforms GELU
- [ ] Layer normalisation placement and its role in stability
- [ ] Residual connections: gradient flow benefit

**Generation**
- [ ] Autoregressive generation: one token at a time
- [ ] Temperature: what it does to the distribution (high=diverse, low=focused)
- [ ] Top-k sampling: what it truncates
- [ ] Top-p (nucleus) sampling: why it adapts better than top-k
- [ ] Greedy decoding vs. beam search: tradeoffs

**KV Cache**
- [ ] What problem KV cache solves (recomputing past tokens)
- [ ] Memory cost of KV cache: calculation for a given model
- [ ] Why context window limits affect KV cache memory

---

### 2. Practical Skills Checklist

- [ ] Implement scaled dot-product attention from scratch (no library calls)
- [ ] Implement multi-head attention with split_heads and merge_heads from scratch
- [ ] Implement sinusoidal positional encoding from scratch
- [ ] Implement a complete transformer block (attention + FFN + residuals + LN) from scratch
- [ ] Implement a full GPT-class model with `generate()` method
- [ ] Train the model on Shakespeare text and generate coherent samples
- [ ] Visualise attention weights using matplotlib heatmaps
- [ ] Implement and verify KV cache speeds up generation (measure tokens/sec)
- [ ] Implement temperature, top-k, and top-p sampling

---

### 3. Coding Challenges

**Challenge A — Attention from Memory**
```python
# Without looking at any code, implement:
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, heads, seq_len, d_k)
    K: (batch, heads, seq_len, d_k)
    V: (batch, heads, seq_len, d_v)
    mask: (batch, 1, 1, seq_len) optional causal mask
    Returns: (batch, heads, seq_len, d_v), attention_weights
    """
    pass

# Verify it matches: F.scaled_dot_product_attention(Q, K, V, is_causal=True)
```

**Challenge B — GPT Block**
```python
# Without looking at any code, implement a full GPT-style transformer block:
# - Pre-norm (LayerNorm before attention and FFN)
# - Causal self-attention with dropout
# - Feed-forward: Linear(d_model, 4*d_model) → GELU → Linear(4*d_model, d_model)
# - Residual connections

# Then build a GPT class:
# - Token embedding + learned positional embedding
# - N transformer blocks
# - Final LayerNorm + linear projection to vocab
# - generate() method with temperature and top-k sampling
```

**Challenge C — Attention Visualisation**
```python
# Load a pre-trained small GPT-2 from HuggingFace
# Extract and plot attention weights for the sentence:
# "The cat sat on the mat because it was tired"
# For each of the 12 heads in layer 0:
# - Plot a 10x10 heatmap of attention weights
# - Label rows and columns with tokens
# Answer: which heads focus on "it" → "cat"? Which focus on syntax?
```

---

### 4. Mini Project

**Karpathy GPT Walkthrough + Extension**: Follow Karpathy's "Let's build GPT from scratch" completely, then extend it:
- Add: learned positional embeddings (instead of sinusoidal)
- Add: KV cache to the generation loop and measure speedup
- Add: top-p sampling to the `generate()` method
- Train on a different dataset (Python code, song lyrics, or news)
- Report: training loss, generation quality, KV cache speedup (tokens/sec with vs. without)

---

### 5. Capstone Project

**nanoGPT Extended**: Full GPT implementation that matches the roadmap's Phase 4b implementation:
- Full GPT-2 small architecture (117M parameter capacity, train a smaller version)
- All of: RoPE, SwiGLU FFN, pre-norm, causal masking, KV cache
- Mixed precision training (BF16)
- Training on TinyStories dataset (1M short children's stories)
- Generate 10 sample stories at three temperatures (0.5, 1.0, 1.5)
- Evaluation: perplexity on held-out test set
- Compare your implementation vs. GPT-2 small loaded from HuggingFace (same architecture)

---

### 6. Interview Questions

**Beginner**

1. **Q: What is self-attention?**
   A: Self-attention allows each token in a sequence to attend to all other tokens. Each token computes a query vector, and all tokens compute key and value vectors. The query is matched against all keys to produce attention weights; the output is a weighted sum of values. This captures relationships between any two positions directly.

2. **Q: Why is the attention score divided by sqrt(d_k)?**
   A: Without scaling, dot products grow large in magnitude as d_k increases (variance of dot product is d_k). Large dot products push softmax into saturation regions where gradients are near zero. Dividing by sqrt(d_k) keeps the variance constant at 1.

3. **Q: What is the difference between encoder-only, decoder-only, and encoder-decoder transformers?**
   A: Encoder-only (BERT): bidirectional attention, sees full context, used for classification/embeddings. Decoder-only (GPT): causal attention, sees only past tokens, used for generation. Encoder-decoder (T5/BART): encoder processes input fully, decoder generates output with cross-attention to encoder.

4. **Q: What is positional encoding and why is it needed?**
   A: Transformers process all tokens in parallel — there is no inherent position information (unlike RNNs). Positional encodings add position information to token embeddings. Sinusoidal: deterministic, allows generalisation to longer sequences. Learned: trained, may not generalise beyond training length.

5. **Q: What is causal masking in transformer attention?**
   A: Causal masking prevents each token from attending to future tokens during training. Implemented by setting future positions to -∞ before softmax, so their weights become 0. Required for language models (you can't see the future when predicting the next token).

6. **Q: What is temperature in text generation?**
   A: Temperature T divides the logits before softmax: p_i = softmax(logits / T). High T (>1) → flatter distribution → more diverse/random. Low T (<1) → sharper distribution → more focused/deterministic. T=0 is greedy (argmax). T=1 is the model's original distribution.

7. **Q: What is multi-head attention and why is it useful?**
   A: Multi-head attention runs H parallel attention operations with different weight matrices, then concatenates and projects. Each head can learn different aspects: syntax, semantics, coreference, etc. Single-head attention has one representation subspace; H heads have H subspaces.

**Intermediate**

8. **Q: What is the KV cache and how does it speed up generation?**
   A: During autoregressive generation, the K and V matrices for previous tokens don't change. KV cache stores them and reuses them for each new token. Without cache: O(n²) computation per step. With cache: O(n) per step — only the new token's K,V is computed.

9. **Q: Explain the difference between pre-norm and post-norm transformers.**
   A: Post-norm (original): LN applied after residual addition → training can be unstable for deep models. Pre-norm (modern default): LN applied before attention/FFN, residual added after → much more stable training, allows larger learning rates. LLaMA, GPT-NeoX use pre-norm (RMSNorm variant).

10. **Q: What is RoPE and why is it preferred over learned positional embeddings?**
    A: Rotary Position Embeddings encode position by rotating Q and K vectors based on position. The dot product Q·K^T naturally encodes relative position distance — the rotation cancels out absolute positions and only relative differences remain. Benefits: generalises to longer sequences than seen during training, efficient implementation.

11. **Q: How does the feed-forward network in a transformer work? What does it compute?**
    A: FFN(x) = W2 · activation(W1 · x). W1 expands to 4× the model dimension (4096 → 16384 in LLaMA-2 7B); W2 contracts back. Attention learns what information to route; FFN stores factual knowledge — researchers have shown specific facts are stored in specific FFN neurons.

12. **Q: What is the memory cost of the KV cache for a 7B parameter model?**
    A: KV cache per token = 2 × n_layers × n_heads × d_head × bytes. For LLaMA-2 7B: 2 × 32 × 32 × 128 × 2 bytes (FP16) = 524KB per token. For 4096 context: ~2GB just for KV cache. This is why long context models are memory-intensive.

13. **Q: What is the difference between top-k and top-p sampling?**
    A: Top-k: keep only the k most likely tokens and renormalise. Problem: k=50 might include very improbable tokens at confident timesteps, or exclude plausible tokens at uncertain ones. Top-p (nucleus): keep the smallest set of tokens whose cumulative probability exceeds p — adapts to the distribution width.

**Advanced**

14. **Q: Derive why attention with Q=K=V is equivalent to a content-based memory lookup.**
    A: Think of K as memory addresses, V as memory contents. Q is the lookup query. The attention score Q·K^T measures similarity between query and each key. Softmax normalises them into weights. The output is a weighted sum of values — a soft memory read. This is the "Neural Turing Machine" intuition.

15. **Q: How does FlashAttention reduce memory from O(N²) to O(N)?**
    A: Standard attention materialises the full N×N attention matrix in HBM memory. FlashAttention computes attention in tiles that fit in SRAM, performing the softmax incrementally (online softmax). Never materialises the full matrix. Memory: O(N) instead of O(N²). Speed: reduces memory bandwidth usage, the real bottleneck.

16. **Q: What is SwiGLU and why does it outperform GELU in transformers?**
    A: SwiGLU(x) = Swish(xW) ⊙ (xV). It's a gated activation: one branch applies Swish, another is a linear gate, they're multiplied element-wise. The gating allows the FFN to selectively activate features. LLaMA reports ~1% perplexity improvement over GELU. The gate gives the FFN more expressive power.

17. **Q: Why does scaling transformers improve performance (scaling laws)?**
    A: Chinchilla scaling laws: loss ∝ 1/N^α + 1/D^β where N=parameters, D=tokens. Both parameters and data are critical. Larger models have more capacity to represent complex patterns. More data provides more diverse training signal. The sweet spot: for compute-optimal training, data should be ~20× parameters in tokens.

18. **Q: What is attention sink and how does it affect generation?**
    A: Research has shown the first token (often a special token like BOS) receives disproportionately high attention in all layers. This "attention sink" acts as a dump for unnecessary attention. StreamingLLM exploits this: keeping only the attention sink + recent tokens allows infinite-length generation with bounded memory.

19. **Q: What is weight tying in language models and why is it used?**
    A: Weight tying shares the embedding matrix (vocab_size × d_model) between input token embeddings and the output projection layer. This reduces parameters by ~10-30% for large vocabulary models (50k tokens). It's justified: the same semantic representation should map input tokens to embeddings and output logits to tokens.

20. **Q: How would you implement rotary embeddings (RoPE) efficiently?**
    A: Apply a rotation matrix to Q and K based on position. In practice: split d_head into pairs, apply cos(mθ) and sin(mθ) rotations where θ_i = 10000^(-2i/d). The relative position between position m and n is encoded in the Q·K dot product because the rotations partially cancel. Efficient: precompute cos/sin tables, apply with complex multiplication trick.

---

### 7. Self-Assessment Quiz

- [ ] Write the scaled dot-product attention formula from memory.
- [ ] Why does causal masking use -infinity before softmax?
- [ ] What is d_k in the attention formula and what does it equal in GPT-2 small?
- [ ] Why do we split into multiple heads instead of one large attention operation?
- [ ] What is the output shape of multi-head attention given (batch=2, seq=10, d_model=512, heads=8)?
- [ ] What is pre-norm vs. post-norm and which is more stable?
- [ ] What does SwiGLU stand for and what is its formula?
- [ ] What is RoPE and what advantage does it have over sinusoidal encoding?
- [ ] What is the memory complexity of standard attention? What does FlashAttention reduce it to?
- [ ] What is the KV cache and what does it trade (memory for speed)?
- [ ] What is temperature=0 equivalent to in generation?
- [ ] What is top-p=0.95 sampling?
- [ ] What is the difference between BERT and GPT architecturally?
- [ ] What is weight tying in language models?
- [ ] Why are residual connections essential in transformers?
- [ ] What is the role of the feed-forward layer vs. the attention layer?
- [ ] What is context length and what limits it practically?
- [ ] What does model.generate() do internally (step by step)?
- [ ] What is the difference between self-attention and cross-attention?
- [ ] What is group query attention (GQA) and why does LLaMA-3 use it?
- [ ] What is the attention score for identical Q and K vectors?
- [ ] What is "head collapse" in multi-head attention?
- [ ] How many parameters does multi-head attention have for d_model=512, heads=8?
- [ ] What is the difference between a language model and a masked language model?
- [ ] Why is layer norm applied to d_model dimension (not batch dimension) in transformers?

**Scoring**: 22–25 ✅ = Ready. 17–21 = Review weak areas. Below 17 = Spend more time on Phase 4b.

---

### 8. Common Mistakes

| Mistake | Why It Happens | How to Avoid |
|---------|---------------|--------------|
| Implementing attention without the sqrt(d_k) scale | Easily forgotten | The scale is in the name "scaled dot-product attention" — always include it |
| Forgetting causal mask for decoder models | Works fine during training, broken at inference | Test generation immediately after implementing; verify it can't see future tokens |
| Mixing up Q, K, V semantics | They look identical in code | Mentally label: Q=what I'm looking for, K=what I offer to be found, V=what I deliver |
| Not using pre-norm | Original paper used post-norm | Modern practice is pre-norm; your model will be harder to train without it |
| Wrong shapes in multi-head attention | Split heads is non-trivial | Print shapes at every step when debugging |
| Using Python loops in attention | Seems natural | The entire attention computation must be vectorised; no Python loops anywhere |
| Training without gradient clipping | Transformer training can have spikes | Always clip to norm 1.0; transformer training is more sensitive than CNN training |
| Not verifying KV cache output matches non-cached output | Implementation bugs | Run both with same inputs and assert outputs are identical |

---

### 9. Readiness Criteria

You are ready for Phase 5 when **all** of the following are true:

- [ ] I implemented scaled dot-product attention from memory in under 10 minutes
- [ ] I implemented a complete GPT class (Project 9) and trained it on Shakespeare
- [ ] My GPT generates coherent text that is recognisably Shakespeare-ish
- [ ] I visualised attention heads and identified at least one interpretable pattern
- [ ] I implemented KV cache and measured a speedup in generation
- [ ] I scored 22/25 or higher on the Self-Assessment Quiz
- [ ] I can answer at least 16/20 Interview Questions correctly
- [ ] I can explain the difference between BERT, GPT, and T5 architecturally

---

### 10. Revision Summary

```
SCALED DOT-PRODUCT ATTENTION
─────────────────────────────────────────────────────
Attention(Q,K,V) = softmax(Q @ Kᵀ / sqrt(d_k)) @ V
Q = query (what I'm looking for)
K = key   (what I offer to be found)
V = value (what I return when found)
Causal mask: set future positions to -inf before softmax

MULTI-HEAD ATTENTION
─────────────────────────────────────────────────────
1. Project Q, K, V with H different weight matrices (W^Q_h, W^K_h, W^V_h)
2. Run attention for each head independently
3. Concatenate head outputs: shape (batch, seq, H*d_v)
4. Project with W^O: shape (batch, seq, d_model)

TRANSFORMER BLOCK (pre-norm)
─────────────────────────────────────────────────────
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
FFN(x) = W2 * SwiGLU(W1 * x, V * x)   [LLaMA style]
      OR W2 * GELU(W1 * x)             [GPT-2 style]

GENERATION
─────────────────────────────────────────────────────
Temperature T: logits → logits/T → softmax
Top-k:         keep k highest-prob tokens, renormalise
Top-p:         keep tokens until cumulative prob > p
KV cache:      store past K,V → O(n) generation vs O(n²)
```

---

### 11. Next Phase Prerequisites

**What Phase 5 (LLMs & Prompt Engineering) requires from Phase 4b:**

| Phase 4b Concept | How Phase 5 Uses It |
|-----------------|---------------------|
| Autoregressive generation | Every LLM API call is this at scale |
| Temperature/top-p sampling | LLM API parameters you'll control daily |
| KV cache mechanics | Understanding context window limits and costs |
| Causal vs. bidirectional attention | Why GPT-style models are used for generation vs. BERT for embeddings |
| Transformer block internals | Understanding why certain prompting techniques work |
| Scaling (more layers/heads = more capable) | Scaling laws, why GPT-4 > GPT-3.5 |
| Weight tying | Understanding embedding space geometry in RAG |

**The critical dependency**: Phase 5 is about *using* LLMs effectively. But effective use requires understanding what happens inside. When you know that temperature controls the softmax distribution and that attention has a context window limit, you'll write better prompts, choose better sampling parameters, and design better systems.

---

*Phase 4b | Part of the [GenAI Engineer Roadmap](./00_README.md)*
