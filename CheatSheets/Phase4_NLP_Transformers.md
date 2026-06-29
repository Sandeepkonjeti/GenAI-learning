# Phase 4 — NLP & Transformers Cheat Sheet

[← CheatSheets Index](./README.md) | [Full Phase Files](../05_Phase4_Part1_NLP_Fundamentals.md)

---

## Attention Mechanism

### Scaled Dot-Product Attention

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- $Q, K, V$: query, key, value matrices — learned projections of input
- $\sqrt{d_k}$: scaling to prevent softmax saturation in high dimensions
- Causal mask: set upper triangle to $-\infty$ before softmax (autoregressive)

```python
import torch, torch.nn.functional as F

def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d_k**0.5  # (B, H, T, T)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return weights @ V  # (B, H, T, d_v)
```

---

## Multi-Head vs Grouped-Query Attention

| Type | KV Heads | Q Heads | KV Cache Size | Used In |
|------|:--------:|:-------:|:-------------:|---------|
| MHA | = Q heads | h | Full | GPT-2, BERT |
| GQA | < Q heads | h | Reduced | **LLaMA-3, Mistral** |
| MQA | 1 | h | Minimum | PaLM, Falcon |

```python
# GQA: n_kv_heads < n_heads
# LLaMA-3-8B: n_heads=32, n_kv_heads=8 → 4× smaller KV cache

# K and V are projected to fewer heads, then expanded for attention
K = K.repeat_interleave(n_heads // n_kv_heads, dim=1)  # (B, n_heads, T, d_k)
V = V.repeat_interleave(n_heads // n_kv_heads, dim=1)
```

---

## Positional Encoding

| Type | Formula | Used In |
|------|---------|---------|
| Sinusoidal | $PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$ | Original Transformer |
| Learned | Embedding table | BERT, GPT-2 |
| **RoPE** | Rotates Q,K by position angle | **LLaMA, Mistral, GPT-NeoX** |
| ALiBi | Adds linear bias to attention scores | Bloom, MPT |

**RoPE key insight**: encodes position via rotation in complex space; allows extrapolation beyond training length via YaRN/LongRoPE.

---

## Transformer Architecture

```
Input Tokens
    ↓ Token Embedding + Positional Encoding
    ↓
[×N Transformer Blocks]
    ├── LayerNorm (pre-norm in modern LLMs)
    ├── Multi-Head Attention (or GQA)
    ├── Residual connection
    ├── LayerNorm
    ├── Feed-Forward Network (Linear → GELU/SiLU → Linear)
    └── Residual connection
    ↓
LayerNorm
    ↓
Linear (vocab projection) → Logits
```

**Pre-norm vs Post-norm**: modern LLMs use pre-norm (LayerNorm before attention/FFN) for more stable training.

**FFN dimension**: typically 4× model dimension. LLaMA uses SwiGLU with $\frac{2}{3} \times 4d$ intermediate.

---

## Tokenization

| Type | Vocab Size | Algorithm | Used In |
|------|:----------:|-----------|---------|
| BPE | ~32K–100K | Byte-pair merges | GPT-2/3/4, LLaMA |
| WordPiece | ~30K | BPE variant (maximize likelihood) | BERT |
| SentencePiece | ~32K | Unigram or BPE | T5, LLaMA-2 |

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokens = tokenizer("Hello world", return_tensors="pt")
# tokens.input_ids shape: (1, sequence_length)
```

---

## Key Numbers

| Model | Params | Layers | d_model | Heads | KV Heads | Context |
|-------|:------:|:------:|:-------:|:-----:|:--------:|:-------:|
| GPT-2 small | 117M | 12 | 768 | 12 | 12 | 1024 |
| LLaMA-3-8B | 8B | 32 | 4096 | 32 | 8 | 8192 |
| LLaMA-3-70B | 70B | 80 | 8192 | 64 | 8 | 8192 |
| Mistral-7B | 7B | 32 | 4096 | 32 | 8 | 32K |

---

## KV Cache

- **What**: Store computed K,V for all previous tokens; don't recompute on each new token
- **Size**: $2 \times \text{layers} \times \text{seq\_len} \times d_{kv} \times \text{bytes\_per\_element}$
- **LLaMA-3-8B** at seq=8192, fp16: ≈ 2 × 32 × 8192 × 128 × 2 ≈ 1.07 GB
- GQA reduces this by $n\_heads / n\_kv\_heads$ (4× for LLaMA-3-8B)

---

## Common Mistakes

- **Forgetting causal mask** for autoregressive generation — future tokens leak
- **Wrong attention shape**: Q is `(B, H, T, d_k)`, not `(B, T, d_model)`
- **Post-norm instead of pre-norm** in training from scratch — less stable
- **Using absolute positions** with long-context models — use RoPE
- **Temperature = 0 ≠ greedy**: temp=0 causes division by zero; implementation clips to very small value or uses `argmax`

---

## Interview Quick-Hits

**Q: Why scale by $\sqrt{d_k}$ in attention?**  
A: For large $d_k$, dot products grow large in magnitude, pushing softmax into regions with tiny gradients (near 0 or 1). Scaling keeps the variance of the dot products around 1.

**Q: What is GQA and why does LLaMA-3 use it?**  
A: Grouped-Query Attention shares one K,V head among multiple Q heads. LLaMA-3-8B uses 8 KV heads for 32 Q heads — 4× smaller KV cache at inference, same quality as full MHA.

**Q: Why do transformers need positional encoding?**  
A: Attention is permutation-invariant — "cat sat mat" and "mat sat cat" produce identical attention inputs without positions. Positional encodings inject order information.

**Q: Explain the residual connection and why it matters.**  
A: Each block computes `x + F(x)`. This creates gradient highways that allow gradients to flow directly to early layers without vanishing. Enables training very deep (100+ layer) networks.

**Q: What is the difference between encoder-only, decoder-only, encoder-decoder?**  
A: Encoder-only (BERT): bidirectional, for representation/classification. Decoder-only (GPT): causal, for generation. Encoder-decoder (T5, BART): encode input, decode output — for seq2seq (translation, summarization).
