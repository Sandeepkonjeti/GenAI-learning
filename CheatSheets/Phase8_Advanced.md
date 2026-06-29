# Phase 8 — Advanced Topics Cheat Sheet

[← CheatSheets Index](./README.md) | [Full Phase File](../09_Phase8_Advanced_Topics.md)

---

## Mixture of Experts (MoE)

```
Standard dense layer:   output = FFN(x)     — ALL parameters active
MoE layer:              output = Σ gᵢ · FFNᵢ(x)   — only top-K experts active

Router: softmax over expert scores → select top-K (usually 2)
Load balancing loss: penalizes routing all tokens to same expert
```

| Model | Total Params | Active Params | Experts |
|-------|:-----------:|:-------------:|:-------:|
| Mistral-8x7B | 47B | ~12.6B per token | 8 (top-2) |
| Mixtral-8x22B | 141B | ~39B per token | 8 (top-2) |
| GPT-4 (alleged) | ~1.8T | ~220B per token | ~16 |

**Why it matters**: MoE models have huge parameter counts (knowledge) but compute of smaller models (speed/cost).

---

## Inference Optimization

### Quantization

| Format | Bits | Memory | Quality Loss | Use Case |
|--------|:----:|:------:|:------------:|---------|
| FP32 | 32 | Baseline | None | Training |
| BF16 | 16 | 0.5× | Negligible | Training/serving |
| FP8 | 8 | 0.25× | Very small | H100 serving |
| GPTQ | 4 | 0.125× | Small | Consumer GPU serving |
| NF4 (QLoRA) | 4 | 0.125× | Small | Fine-tuning |
| GGUF | 2–8 | 0.06–0.25× | Varies | CPU inference (llama.cpp) |

---

## vLLM (Production Inference)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=2,      # split across 2 GPUs
    gpu_memory_utilization=0.9,  # use 90% of GPU memory
    dtype="bfloat16",
)

sampling = SamplingParams(temperature=0.0, max_tokens=512)
outputs = llm.generate(["Explain transformers:"], sampling)
print(outputs[0].outputs[0].text)
```

**PagedAttention**: vLLM's key innovation. Manages KV cache in non-contiguous memory pages (like virtual memory). Enables high batch sizes without memory fragmentation.

**Throughput comparison** (LLaMA-3-8B, A100):
| System | Tokens/sec |
|--------|:----------:|
| Naive HuggingFace | ~200 |
| vLLM | ~1200–2000 |
| vLLM + continuous batching | ~4000+ |

---

## Speculative Decoding

```
Problem: LLM generates 1 token per forward pass → bottleneck
Solution:
  1. Draft model (small, fast) generates K candidate tokens
  2. Target model (large) verifies all K in ONE forward pass
  3. Accept tokens up to first disagreement, reject rest
  4. Net result: ~2–3× throughput on average

Requirements: draft model must have same tokenizer as target
```

---

## Multimodal Quick Reference

| Model | Modalities | Architecture |
|-------|-----------|-------------|
| CLIP | Image + Text | Dual encoder, contrastive |
| LLaVA | Image + Text | CLIP vision encoder + LLM |
| GPT-4V / Claude 3 | Image + Text | End-to-end multimodal |
| Whisper | Audio → Text | Encoder-decoder transformer |

```python
# Use GPT-4o for vision
import base64
from openai import OpenAI

with open("diagram.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = OpenAI().chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Explain this architecture diagram"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
    ]}]
)
```

---

## State Space Models (Mamba)

| Aspect | Transformers | Mamba |
|--------|-------------|-------|
| Attention complexity | O(n²) | O(n) |
| KV cache | Grows with sequence | Fixed size |
| Long context | Struggles at 100K+ | Efficient |
| Short context quality | Better | Slightly worse |
| Status | Dominant | Growing, hybrid models emerging |

---

## Common Mistakes

- **vLLM tensor_parallel_size > GPU count** — crashes on startup
- **Mixed precision errors** — ensure model dtype matches quantization dtype
- **Speculative decoding draft model mismatch** — must share vocabulary
- **MoE load imbalance** — add auxiliary load balancing loss during training
- **Quantization on CPU** — quantization only speeds up GPU inference

---

## Interview Quick-Hits

**Q: What is PagedAttention in vLLM?**  
A: KV cache is managed in fixed-size non-contiguous memory pages (like OS virtual memory). Eliminates memory fragmentation from variable sequence lengths, enabling much higher batch sizes and throughput.

**Q: Why doesn't quantization severely hurt quality?**  
A: LLM weights follow approximately Gaussian distributions with most values small. 4-bit quantization (NF4) is designed around this distribution — normal float 4-bit, not linear. Post-training quantization calibration also compensates.

**Q: Explain MoE in one minute.**  
A: Instead of one FFN per transformer block, you have N "expert" FFNs. A router network picks the top-K for each token. Total parameters = N × FFN size, but compute per token = K × FFN — much cheaper than a dense model with the same parameter count.

**Q: When would you choose vLLM over HuggingFace generate?**  
A: Any production serving scenario. vLLM's continuous batching, PagedAttention, and optimized CUDA kernels give 5–10× better throughput. HuggingFace generate is fine for offline/research use but not for serving real traffic.
