# Phase 7a: Fine-Tuning & LoRA
**Months 14–15 | Difficulty: 8/10 | Label: 🟡 Learn Later**

> **Previous Phase**: [Phase 6 — AI Agents & MCP](./07_Phase6_AI_Agents_and_MCP.md)  
> **Next Phase**: [Phase 7b — RLHF & DPO](./08_Phase7_Part2_RLHF_and_DPO.md)

---

## Table of Contents

- [Phase Overview](#phase-overview)
- [When to Fine-Tune vs. When Not To](#when-to-fine-tune-vs-when-not-to)
- [Types of Fine-Tuning](#types-of-fine-tuning)
- [Dataset Preparation](#dataset-preparation)
  - [Data Quality Over Quantity](#data-quality-over-quantity)
  - [ChatML Format and Conversation Templates](#chatml-format-and-conversation-templates)
  - [Dataset Curation Best Practices](#dataset-curation-best-practices)
- [LoRA: Low-Rank Adaptation](#lora-low-rank-adaptation)
  - [The Mathematical Foundation](#the-mathematical-foundation)
  - [Connection to SVD](#connection-to-svd)
  - [LoRA Hyperparameters Explained](#lora-hyperparameters-explained)
  - [Implementing LoRA from Scratch](#implementing-lora-from-scratch)
- [QLoRA: Quantized LoRA](#qlora-quantized-lora)
  - [4-bit NF4 Quantization](#4-bit-nf4-quantization)
  - [Paged Optimizers](#paged-optimizers)
  - [VRAM Requirements](#vram-requirements)
- [Fine-Tuning with Unsloth and Hugging Face TRL](#fine-tuning-with-unsloth-and-hugging-face-trl)
  - [Full Fine-Tuning Pipeline](#full-fine-tuning-pipeline)
- [Inference with Fine-Tuned Models](#inference-with-fine-tuned-models)
- [Resources](#resources)
- [Projects](#projects)
- [Common Mistakes](#common-mistakes)
- [Mastery Checklist](#mastery-checklist)

---

## Phase Overview

| Attribute | Details |
|-----------|---------|
| **Duration** | Months 14–15 (4 weeks) |
| **Daily Time** | 1.5–2 hours |
| **Difficulty** | 8/10 |
| **Label** | 🟡 Learn Later |
| **Prerequisites** | Phase 5a (LLMs, instruction tuning), Phase 3b (PyTorch training loop) |
| **Outcome** | Can fine-tune open-source LLMs with LoRA/QLoRA on a consumer GPU or Databricks cluster |

---

## When to Fine-Tune vs. When Not To

This decision is critical. Fine-tuning is expensive and often unnecessary.

```
Decision tree:
                    ┌─────────────────────────────────────┐
                    │ Can prompting solve your problem?   │
                    └─────────────────┬───────────────────┘
                                      │
                          YES ────────┴──── NO
                           │                │
              ┌────────────▼───┐    ┌───────▼─────────────────────┐
              │ Use prompting  │    │ Is it a knowledge problem?  │
              │ (simpler, fast)│    │ (needs private data/facts)  │
              └────────────────┘    └───────┬─────────────────────┘
                                            │
                                YES ────────┴──── NO
                                 │                 │
                    ┌────────────▼──┐    ┌─────────▼─────────────────┐
                    │  Use RAG      │    │ Is it a style/behavior    │
                    │  (no training)│    │ problem? (tone, format,   │
                    └───────────────┘    │ domain-specific reasoning)│
                                         └─────────┬─────────────────┘
                                                   │
                                       YES ────────┴──── NO
                                        │                │
                           ┌────────────▼──┐    ┌────────▼────────────┐
                           │  Fine-tune    │    │ Reconsider: is it   │
                           │  with LoRA    │    │ achievable at all?  │
                           └───────────────┘    └─────────────────────┘
```

**Fine-tuning IS the right choice when**:
- You need a specific response style consistently (medical, legal, code)
- You need to teach the model proprietary terminology/domain knowledge
- Latency requires a smaller, specialized model
- You need behavior that prompting can't achieve reliably
- Cost efficiency at scale (fine-tuned 7B >> expensive GPT-4 for narrow tasks)

**Fine-tuning is NOT the right choice when**:
- The base model already does the task well with prompting
- You have less than 1000 high-quality examples
- The task requires up-to-date world knowledge (use RAG)
- You don't have evaluation infrastructure

---

## Types of Fine-Tuning

| Type | What It Adapts | Data Needed | Cost | When |
|------|---------------|:-----------:|:----:|------|
| **Full fine-tuning** | All weights | 10K–100K | Highest | Research, full control |
| **Instruction tuning** | Behavior/following | 1K–50K | High | Base → assistant |
| **Domain adaptation** | Knowledge/style | 1K–100K | High | Medical, legal, code |
| **PEFT (LoRA/QLoRA)** | Small adapters only | 500–10K | Low | Most practical use cases |
| **Prefix tuning** | Soft prompt tokens | 500–5K | Very low | Simple behavior changes |

---

## Dataset Preparation

### Data Quality Over Quantity

> "100 perfect examples beat 10,000 noisy ones."

Checklist for fine-tuning data:
- [ ] Every example is the exact behavior you want the model to learn
- [ ] No contradictory examples (different answers to the same question)
- [ ] Diversity: covers the full range of inputs the model will see
- [ ] Consistent formatting: same template throughout
- [ ] Correct answers: every example verified by a human or strong model

---

### ChatML Format and Conversation Templates

```python
# The standard fine-tuning data format
from datasets import Dataset

# Single-turn instruction format
single_turn_example = {
    "messages": [
        {
            "role": "system",
            "content": "You are a Databricks expert who provides concise, accurate answers about Spark and Delta Lake."
        },
        {
            "role": "user",
            "content": "What is the difference between cache() and persist() in PySpark?"
        },
        {
            "role": "assistant",
            "content": """Both `cache()` and `persist()` materialize a DataFrame to avoid recomputation, but differ in storage level:

- **`cache()`**: Always uses `MEMORY_AND_DISK` (or `MEMORY_AND_DISK_DESER` in Spark 3.x). Simple shorthand.
- **`persist(storageLevel)`**: Lets you choose the storage level explicitly:
  - `MEMORY_ONLY`: Fast, but spills data if RAM is full
  - `MEMORY_AND_DISK`: Fallback to disk if RAM is full (same as cache)
  - `DISK_ONLY`: Useful for very large DataFrames
  - `OFF_HEAP`: Stores in off-heap memory (avoids GC pressure)

Use `persist()` when you need control over storage; otherwise `cache()` is fine for most cases."""
        }
    ]
}

# Multi-turn conversation format
multi_turn_example = {
    "messages": [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a PySpark function to deduplicate a DataFrame."},
        {"role": "assistant", "content": "Here's a PySpark deduplication function:\n\n```python\ndef deduplicate_df(df, subset=None, keep='first'):\n    if subset:\n        return df.dropDuplicates(subset)\n    return df.dropDuplicates()\n```"},
        {"role": "user", "content": "Can you add window function support for keeping the latest record?"},
        {"role": "assistant", "content": "Here's the enhanced version with window functions:\n\n```python\nfrom pyspark.sql import functions as F\nfrom pyspark.sql.window import Window\n\ndef deduplicate_keep_latest(df, id_cols, order_col):\n    window = Window.partitionBy(id_cols).orderBy(F.col(order_col).desc())\n    return (\n        df\n        .withColumn('_rank', F.row_number().over(window))\n        .filter(F.col('_rank') == 1)\n        .drop('_rank')\n    )\n```"}
    ]
}

# Convert to HuggingFace Dataset
data = [single_turn_example, multi_turn_example]  # + more examples
dataset = Dataset.from_list(data)
```

---

### Dataset Curation Best Practices

```python
from transformers import AutoTokenizer

def analyze_dataset(dataset, tokenizer_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    """Analyze dataset quality before fine-tuning."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    token_counts = []
    for example in dataset:
        # Format as the model expects
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        tokens = tokenizer(text, return_tensors="pt")["input_ids"].shape[1]
        token_counts.append(tokens)
    
    import numpy as np
    token_array = np.array(token_counts)
    
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Token count stats:")
    print(f"  Min: {token_array.min()}")
    print(f"  Max: {token_array.max()}")
    print(f"  Mean: {token_array.mean():.0f}")
    print(f"  P95: {np.percentile(token_array, 95):.0f}")
    print(f"  Exceeds 4096 tokens: {(token_array > 4096).sum()} examples")
    
    # Flag potential issues
    short = (token_array < 50).sum()
    if short > 0:
        print(f"  Warning: {short} very short examples (< 50 tokens) — possibly corrupt")

# Recommended dataset sizes by task
dataset_sizes = {
    "Style adaptation":          "200–1000 examples",
    "Domain Q&A":                "500–5000 examples",
    "Code generation (specific)":"1000–10000 examples",
    "General instruction":       "5000–50000 examples",
    "Full chat assistant":       "50000+ examples",
}
```

---

## LoRA: Low-Rank Adaptation

### The Mathematical Foundation

Fine-tuning updates every weight in the model:

$$W_{new} = W_{pretrained} + \Delta W$$

**The problem**: For a 7B parameter model, $\Delta W$ has 7 billion parameters. That's:
- 7B × 4 bytes (float32) = **28 GB** just for the weight updates
- Plus optimizer states: Adam stores 2 copies → **84 GB** for optimizer alone
- Impossible on a single consumer GPU

**LoRA's insight**: The weight updates $\Delta W$ have a low **intrinsic rank**. You don't need a full $d \times d$ matrix — a low-rank approximation works almost as well.

$$\Delta W = BA \text{ where } B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d, k)$$

**Parameter reduction**:
- Full $\Delta W$: $d \times k$ parameters (e.g., $4096 \times 4096 = 16.7M$)
- LoRA $\Delta W = BA$: $d \times r + r \times k = r(d + k)$ parameters (e.g., $r=16$: $16 \times 8192 = 131K$)
- **Reduction factor**: $\frac{d \times k}{r(d+k)} \approx \frac{4096}{16 \times 2} = 128\times$ fewer parameters!

---

### Connection to SVD

LoRA's low-rank decomposition is closely related to Singular Value Decomposition (SVD). If you studied Phase 1 (Mathematics), you'll recognize this.

```python
import torch
import numpy as np

# Recall SVD: any matrix W can be written as W = UΣV^T
# where U, V are orthogonal and Σ is diagonal with singular values

# For LoRA, we approximate ΔW ≈ BA where:
# B ≈ U[:, :r] × sqrt(Σ[:r,:r])  (left singular vectors)
# A ≈ sqrt(Σ[:r,:r]) × V^T[:r,:] (right singular vectors)

# The rank r controls how much of the "important" information we keep
# The top-r singular values capture the most variance in ΔW

# Why does this work for fine-tuning?
# Research (Aghajanyan et al. 2020, "Intrinsic Dimensionality Explains...") shows
# that language model fine-tuning updates reside in a very low-dimensional subspace
# Even rank 4-16 captures the essential changes for most tasks!

def demonstrate_low_rank_approximation():
    # Create a random matrix simulating a weight update
    d, k = 768, 768
    W = torch.randn(d, k)
    
    # SVD decomposition
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    
    # Reconstruct with different ranks
    for r in [4, 8, 16, 32, 64]:
        W_approx = U[:, :r] @ torch.diag(S[:r]) @ Vt[:r, :]
        
        # Frobenius norm reconstruction error
        error = torch.norm(W - W_approx) / torch.norm(W)
        variance_captured = (S[:r].sum() / S.sum()).item()
        
        print(f"r={r:3d}: error={error:.4f}, variance_captured={variance_captured:.4f}, "
              f"params={r*(d+k):,} vs {d*k:,}")

demonstrate_low_rank_approximation()
# Output shows that even r=16 captures >95% of variance for typical weight matrices
```

---

### LoRA Hyperparameters Explained

| Hyperparameter | Typical Values | What It Controls |
|---------------|:--------------:|-----------------|
| `r` (rank) | 4, 8, 16, 32, 64 | Capacity of the adapter; higher = more expressive but more params |
| `alpha` | 8, 16, 32 (≈ 2r) | Scaling factor: effective scale = alpha/r |
| `dropout` | 0.0–0.1 | Regularization on LoRA layers |
| `target_modules` | See below | Which layers get LoRA adapters |

**Target modules** (what to adapt per architecture):

| Model | target_modules |
|-------|---------------|
| LLaMA, Mistral | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| GPT-2 | `c_attn, c_proj, c_fc` |
| Falcon | `query_key_value, dense, dense_h_to_4h, dense_4h_to_h` |
| Conservative (fewer params) | `q_proj, v_proj` only |

---

### Implementing LoRA from Scratch

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    During training:
      output = x @ (W + BA * scale)
      where scale = alpha / r
    
    During inference (after merging):
      output = x @ W_merged  (W_merged = W + BA * scale)
    
    During inference (unmerged):
      output = x @ W + (x @ A.T) @ B.T * scale
      (two small matmuls instead of one large one)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        self.r = r
        self.scale = alpha / r
        
        # Frozen original weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        
        # LoRA trainable adapters
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))  # Zero init!
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B is initialized to zeros → no change at the start of training
        # This means ΔW = BA = 0 at initialization → same as pre-trained model
        # Training gradually learns non-zero ΔW
        
        # Freeze base weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base weight computation (frozen)
        result = nn.functional.linear(x, self.weight, self.bias)
        
        # LoRA computation (trainable)
        lora_update = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        result = result + lora_update * self.scale
        
        return result
    
    def merge_weights(self) -> None:
        """
        Merge LoRA weights into base weights for efficient inference.
        After merging, this layer behaves exactly like a regular Linear layer.
        """
        with torch.no_grad():
            delta_W = self.lora_B @ self.lora_A  # (out, r) @ (r, in) = (out, in)
            self.weight.data += delta_W * self.scale
        
        # Zero out LoRA weights (they're now baked in)
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()

# Demonstrate: apply LoRA to a simple network
class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Linear(d_model, d_model * 4)
    
    def replace_with_lora(self, r: int = 8, alpha: float = 16.0):
        """Replace target layers with LoRA versions."""
        # Replace q_proj and v_proj with LoRA versions
        self.q_proj = LoRALinear(512, 512, r=r, alpha=alpha)
        self.v_proj = LoRALinear(512, 512, r=r, alpha=alpha)
        # ffn is NOT replaced (conservative LoRA)

def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total

block = SimpleTransformerBlock()
trainable, total = count_trainable_params(block)
print(f"Without LoRA: {trainable:,}/{total:,} trainable ({trainable/total:.1%})")

block.replace_with_lora(r=8)
trainable, total = count_trainable_params(block)
print(f"With LoRA:    {trainable:,}/{total:,} trainable ({trainable/total:.1%})")
```

---

## QLoRA: Quantized LoRA

### 4-bit NF4 Quantization

QLoRA (Dettmers et al., 2023) combines LoRA with 4-bit quantization of the base model weights.

```python
# QLoRA reduces base model memory by ~75% while maintaining quality

# Standard approach (no quantization):
# 7B params × 4 bytes (float32) = 28 GB VRAM

# QLoRA approach:
# 7B params × 0.5 bytes (4-bit NF4) = 3.5 GB for base weights
# + LoRA adapters in float16: ~0.2 GB
# Total: ~4 GB — fits on a consumer GPU!

# NF4 (Normal Float 4-bit):
# Standard 4-bit quantization maps weights to 16 fixed values
# NF4 maps weights to 16 values chosen by quantiles of a standard normal distribution
# Why normal? Pre-trained weights are approximately normally distributed
# Result: better representation of the actual weight distribution

# Double quantization:
# Quantize the quantization constants themselves (saves another ~0.5 GB per 7B model)

from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 4-bit quantization
    bnb_4bit_quant_type="nf4",           # NF4 (better than fp4 for LLMs)
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bfloat16 (faster than float32)
    bnb_4bit_use_double_quant=True,      # Double quantization (save memory)
)
```

### VRAM Requirements

| Configuration | 7B Model | 13B Model | 70B Model |
|--------------|:--------:|:---------:|:---------:|
| Full FP32 | 28 GB | 52 GB | 280 GB |
| Full BF16 | 14 GB | 26 GB | 140 GB |
| LoRA (BF16 base) | 14 GB | 26 GB | 140 GB |
| QLoRA (4-bit base) | **5–6 GB** | **9–10 GB** | **48–50 GB** |
| Inference only (4-bit) | **3–4 GB** | **7–8 GB** | **35–40 GB** |

---

## Fine-Tuning with Unsloth and Hugging Face TRL

### Full Fine-Tuning Pipeline

```python
# pip install unsloth transformers trl peft bitsandbytes accelerate

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from datasets import load_dataset, Dataset

# ── Step 1: Load Model with Unsloth ──
# Unsloth provides 2x faster training and 60% memory reduction through custom CUDA kernels
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    dtype=None,          # Auto-detect: bfloat16 if supported
    load_in_4bit=True,   # QLoRA
)

# ── Step 2: Add LoRA adapters ──
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                   # Rank
    target_modules=[        # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,          # Scaling: alpha/r = 1.0 (no additional scaling)
    lora_dropout=0.0,       # Unsloth recommends 0.0 for optimal performance
    bias="none",            # Don't add bias to LoRA layers
    use_gradient_checkpointing="unsloth",  # Saves memory at cost of speed
    random_state=42,
)

# Show parameter count
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

# ── Step 3: Prepare Dataset ──
def format_for_training(examples: dict) -> dict:
    """Format messages into training text."""
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

# Load your dataset (replace with your actual data)
raw_dataset = Dataset.from_list([
    {
        "messages": [
            {"role": "system", "content": "You are a PySpark expert."},
            {"role": "user", "content": "How do I read a Delta table in PySpark?"},
            {"role": "assistant", "content": "Use spark.read.format('delta').load('/path/to/table') or spark.table('catalog.schema.table')."}
        ]
    }
    # ... add more examples
])

train_dataset = raw_dataset.map(format_for_training, batched=True)

# ── Step 4: Configure Trainer ──
training_args = SFTConfig(
    output_dir="./llama3-databricks-expert",
    
    # Training
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,    # Effective batch size = 2 × 4 = 8
    
    # Optimization
    learning_rate=2e-4,               # Higher than full fine-tuning (LoRA uses smaller updates)
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    
    # Memory
    fp16=False,
    bf16=True,                        # bfloat16 for modern GPUs (A100, H100)
    optim="adamw_8bit",               # 8-bit Adam optimizer (saves 6x memory)
    gradient_checkpointing=True,
    
    # Logging
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
    
    # Data
    max_seq_length=2048,
    dataset_text_field="text",
    packing=False,                    # True packs multiple examples per sequence (faster)
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset.select(range(min(50, len(train_dataset)))),
)

# ── Step 5: Train ──
trainer_stats = trainer.train()

print(f"\nTraining complete!")
print(f"  Duration: {trainer_stats.metrics['train_runtime']:.0f}s")
print(f"  Final loss: {trainer_stats.metrics['train_loss']:.4f}")

# ── Step 6: Save ──
# Option A: Save LoRA adapters only (small, ~100MB)
model.save_pretrained("./lora_adapters")
tokenizer.save_pretrained("./lora_adapters")

# Option B: Save merged model (full model with weights merged)
model.save_pretrained_merged(
    "./merged_model",
    tokenizer,
    save_method="merged_16bit"   # or "merged_4bit" to keep quantization
)

# Option C: Push to HuggingFace Hub
# model.push_to_hub("your-username/llama3-databricks-expert")
```

---

## Distributed and Multi-GPU Training

Single-GPU fine-tuning (QLoRA on 24GB) covers most use cases. When you need to scale up, here are the patterns — especially relevant with your Databricks background.

### Data Parallelism (the simplest approach)

Each GPU gets the full model and a different batch. Gradients are aggregated across GPUs after each step.

```python
# PyTorch DistributedDataParallel (DDP)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def train_ddp(rank: int, world_size: int, model, dataset):
    """Function to run on each GPU process."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Move model to this GPU
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # DistributedSampler ensures each GPU sees different data
    from torch.utils.data import DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, sampler=sampler, batch_size=8)
    
    # Training loop (same as single GPU — DDP handles gradient sync)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    for batch in loader:
        loss = model(**batch).loss
        loss.backward()        # Automatically syncs gradients across GPUs
        optimizer.step()
        optimizer.zero_grad()
    
    dist.destroy_process_group()

# Launch on 4 GPUs:
# mp.spawn(train_ddp, args=(4, model, dataset), nprocs=4)
```

### DeepSpeed ZeRO — For Larger Models

ZeRO (Zero Redundancy Optimizer) shards optimizer states, gradients, and parameters across GPUs:

| ZeRO Stage | What's Sharded | VRAM Reduction | Speed Impact |
|:----------:|---------------|:--------------:|:------------:|
| Stage 1 | Optimizer states | ~4× | Minimal |
| Stage 2 | + Gradients | ~8× | Small |
| Stage 3 | + Parameters | ~64× | Significant |
| ZeRO-Infinity | + CPU/NVMe offload | Unlimited | Large |

```python
# DeepSpeed integration with HuggingFace Trainer
# Create ds_config.json:
ds_config = {
    "zero_optimization": {
        "stage": 2,                  # Start with stage 2; use 3 for largest models
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "fp16": {"enabled": True},
    "gradient_accumulation_steps": 8,
    "train_micro_batch_size_per_gpu": 4
}

# Then in TrainingArguments:
from transformers import TrainingArguments
args = TrainingArguments(
    deepspeed="ds_config.json",    # Just add this line
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    fp16=True,
    # ... other args
)
# Launch: deepspeed --num_gpus=4 train.py
```

### Databricks: TorchDistributor

With your Databricks background, you can scale to multi-node training without managing infrastructure:

```python
from pyspark.ml.torch.distributor import TorchDistributor

def train_on_cluster(rank: int, num_proc: int):
    """This function runs on each GPU worker in the Databricks cluster."""
    import torch.distributed as dist
    dist.init_process_group("nccl")
    
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            output_dir="/dbfs/mnt/models/finetuned",  # DBFS mount
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
        ),
        train_dataset=dataset,
    )
    trainer.train()

# Launch on 4 GPUs (1 node) or 8 GPUs (2 nodes):
distributor = TorchDistributor(
    num_processes=4,
    local_mode=False,    # False = distributed across cluster nodes
    use_gpu=True
)
distributor.run(train_on_cluster)
```

**When to use each approach**:

| Scenario | VRAM Available | Approach |
|----------|:--------------:|----------|
| 1 × 24GB GPU | 24 GB | QLoRA (single GPU, this is fine for 7B models) |
| 4 × 24GB GPUs (1 node) | 96 GB | DDP with LoRA, or DeepSpeed ZeRO-2 |
| 8 × 80GB GPUs (2 nodes) | 640 GB | DDP or DeepSpeed ZeRO-2, can full-fine-tune 70B |
| Large Databricks cluster | Unlimited | TorchDistributor + ZeRO-3 or Mosaic AI Training |

---

## Inference with Fine-Tuned Models

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ── Option A: Load LoRA adapters on top of base model ──
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./lora_adapters")
tokenizer = AutoTokenizer.from_pretrained("./lora_adapters")

# ── Option B: Load merged model (faster inference) ──
model = AutoModelForCausalLM.from_pretrained(
    "./merged_model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate
messages = [
    {"role": "system", "content": "You are a PySpark expert."},
    {"role": "user", "content": "What's the best way to handle skewed data in PySpark joins?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

---

## Resources

| Rank | Resource | Type | Cost | Why |
|------|----------|------|------|-----|
| 1 | [Unsloth Documentation](https://docs.unsloth.ai/) | Docs | Free | The fastest way to fine-tune on limited hardware |
| 2 | [HuggingFace TRL Documentation](https://huggingface.co/docs/trl) | Docs | Free | Standard library for SFT, RLHF, DPO |
| 3 | [QLoRA Paper (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) | Paper | Free | Understanding the math behind QLoRA |
| 4 | [LoRA Paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) | Paper | Free | Original LoRA paper — well written, short |
| 5 | [Sebastian Raschka Fine-Tuning Guide](https://github.com/rasbt/LLM-finetuning-scripts) | GitHub | Free | Practical, well-commented fine-tuning scripts |
| 6 | [Maxime Labonne LLM Course](https://github.com/mlabonne/llm-course) | GitHub | Free | Best practical LLM fine-tuning tutorial |

---

## Projects

### Project 18: Domain Expert Fine-Tune
**Difficulty**: 7/10 | **Time**: 2–3 weeks

Fine-tune LLaMA 3.1 8B to be an expert on your codebase domain (Databricks, Kafka, PySpark):

1. **Collect data**: Mine your codebase, documentation, Stack Overflow threads
2. **Clean and format**: Convert to instruction-response pairs in ChatML format
3. **Quality filter**: Manually review 100 random examples; discard poor quality
4. **Fine-tune**: Use QLoRA with the pipeline above on a free Colab GPU
5. **Evaluate**: Compare to base model on 30 test questions
6. **Share**: Push adapters to HuggingFace Hub

**Target**: Fine-tuned model should outperform GPT-4o-mini on your domain-specific questions.

---

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Fine-tuning when prompting would work | Wasted compute and time | Always try prompting first |
| Using wrong chat template | Model learns bad format; poor outputs | Use tokenizer.apply_chat_template() |
| Too high learning rate | Training divergence, model collapse | Start with 1e-4 to 2e-4 for LoRA |
| Not evaluating after training | Don't know if model improved | Always evaluate on held-out test set |
| Training on all tokens including the prompt | Model learns to repeat input | Mask prompt tokens; only train on completion |
| Too little data | Overfitting; model memorizes examples | Minimum 200 examples; prefer 1K+ |
| Not saving adapters separately | Lose efficiency of LoRA | Always save adapters; merge only for inference |

---

## Mastery Checklist

- [ ] Can explain why fine-tuning is NOT always the answer (decision tree)
- [ ] Understands LoRA mathematically (low-rank decomposition, connection to SVD)
- [ ] Can explain r, alpha, and target_modules hyperparameters
- [ ] Can explain QLoRA's NF4 quantization and why it works
- [ ] Built a working fine-tuning pipeline from scratch
- [ ] Evaluated fine-tuned model vs. base model systematically
- [ ] Read the LoRA and QLoRA papers

---

## Moving to Phase 7b

**Before proceeding to [Phase 7b: RLHF & DPO](./08_Phase7_Part2_RLHF_and_DPO.md), confirm:**

- [ ] Fine-tuned at least one model with LoRA/QLoRA
- [ ] Can explain how the training loss should behave during fine-tuning
- [ ] Understand the difference between base models, instruction-tuned models, and your fine-tuned model

**Why Phase 7b comes next**: SFT (supervised fine-tuning) teaches models what to do. RLHF and DPO teach them to do it in the way humans prefer — which is a fundamentally different and harder problem.

---

## Phase Completion & Readiness Assessment

> Complete this assessment before Phase 7b. Fine-tuning is a sought-after skill — most engineers use libraries without understanding the math. You should understand both.

---

### 1. Knowledge Checklist

**When to Fine-Tune**
- [ ] Fine-tuning vs. RAG vs. prompt engineering — decision framework
- [ ] Tasks that genuinely benefit from fine-tuning: style, format, domain vocabulary
- [ ] Tasks better served by RAG: factual recall, recent information
- [ ] Dataset size requirements: minimum for SFT, what happens with too little data

**LoRA Mathematics**
- [ ] Low-rank decomposition: ΔW = BA where B ∈ R^{d×r}, A ∈ R^{r×k}
- [ ] Why low-rank is justified: intrinsic dimensionality hypothesis
- [ ] SVD connection: best rank-r approximation via SVD
- [ ] Parameter reduction: r(d+k) vs. d×k (why rank=16 is ~128× fewer parameters)
- [ ] LoRA hyperparameters: r (rank), alpha (scaling), dropout, target_modules

**QLoRA**
- [ ] 4-bit NF4 quantisation: what Normal Float quantisation is
- [ ] BitsAndBytes: how it enables LoRA on consumer hardware
- [ ] VRAM requirements: full FP32 vs. BF16 vs. LoRA vs. QLoRA comparison
- [ ] Double quantisation: what it does to reduce VRAM further

**Training Pipeline**
- [ ] ChatML format: system/user/assistant message structure
- [ ] Data quality principles: more important than quantity
- [ ] SFTTrainer from TRL: key configuration parameters
- [ ] Training metrics: training loss, validation loss, perplexity
- [ ] Inference with adapters: `PeftModel.from_pretrained`, `.merge_and_unload()`

---

### 2. Practical Skills Checklist

- [ ] Implement LoRALinear from scratch (forward pass, weight merging)
- [ ] Set up QLoRA training with BitsAndBytesConfig from memory
- [ ] Configure SFTTrainer with appropriate hyperparameters
- [ ] Collect, clean, and format a 500+ example instruction dataset
- [ ] Run a fine-tuning job on free Google Colab GPU (T4 or A100)
- [ ] Push trained adapters to HuggingFace Hub
- [ ] Load and run inference with `PeftModel.from_pretrained`
- [ ] Compare fine-tuned model vs. base model on 20 evaluation questions

---

### 3. Coding Challenges

**Challenge A — LoRALinear from Scratch**
```python
# Implement a LoRA-augmented linear layer that:
# 1. Keeps the original weight W_0 frozen
# 2. Adds trainable matrices A (r×k) and B (d×r)
# 3. Forward: y = x @ (W_0 + B @ A).T * (alpha/r)
# 4. Implements merge_weights() that adds BA into W_0 and removes A, B
# 5. Verifies: merged output == unmerged output (within floating point tolerance)
# Test on a simple linear regression: only A and B should have gradients
```

**Challenge B — Hyperparameter Ablation**
```python
# Train 4 QLoRA checkpoints on the same dataset with different rank values:
# ranks = [4, 8, 16, 32]
# Keep alpha = rank (alpha/r = 1 for all)
# Evaluate each on 50 test questions using LLM-as-judge (1-5 scale)
# Plot: quality score vs. rank vs. trainable parameter count
# Answer: is there a clear winner? What is the diminishing returns point?
```

**Challenge C — Dataset Quality Filter**
```python
# Write a function that scores each instruction-response pair in your dataset:
# Criteria:
#   1. Response length: reject if < 50 or > 2000 tokens
#   2. Code quality: if response contains code, verify it's syntactically valid Python/SQL
#   3. Relevance: embed question and answer; reject if cosine sim < 0.3
#   4. Diversity: flag near-duplicates (cosine sim > 0.95 with existing examples)
# Output: cleaned dataset + quality report (% rejected per criterion)
```

---

### 4. Mini Project

**Databricks Domain Expert Fine-Tune** (Project 18 from the roadmap):
- Collect 1500+ QA pairs from: Databricks docs, Stack Overflow, your own experience
- Format in ChatML with system prompt: "You are a Databricks and PySpark expert..."
- Run quality filter (Challenge C) before training
- Fine-tune LLaMA-3.1 8B with QLoRA (r=16, alpha=16) on Colab A100
- Track training with W&B: loss curve, eval loss, sample generations per 100 steps
- Evaluate: 30 domain-specific questions vs. base model and GPT-4o-mini
- Push to HuggingFace Hub: `your-username/databricks-llama-3.1-8b-lora`

---

### 5. Capstone Project

**Fine-Tuning Benchmark Study**:
- Compare 5 approaches on the same task (Databricks QA):
  1. Base LLaMA-3.1 8B (zero-shot)
  2. Base LLaMA-3.1 8B (few-shot, 5 examples)
  3. LLaMA-3.1 8B SFT LoRA (r=8)
  4. LLaMA-3.1 8B SFT LoRA (r=32)
  5. GPT-4o-mini (baseline)
- Evaluation: 50 test questions, LLM-as-judge (score 1-5), human evaluation (10%)
- Report: performance vs. cost vs. latency table
- Conclusion: when does fine-tuning beat prompting? At what dataset size?

---

### 6. Interview Questions

**Beginner**

1. **Q: What is fine-tuning and when should you use it instead of prompting?**
   A: Fine-tuning updates model weights on a task-specific dataset. Use when: (1) consistent output format is required across thousands of calls; (2) domain vocabulary is absent from the base model; (3) style/tone requirements are hard to achieve via prompting; (4) latency matters and system prompts are long. Don't fine-tune if RAG or better prompting can solve the problem.

2. **Q: What is LoRA?**
   A: Low-Rank Adaptation. Instead of updating all weights W, LoRA adds a low-rank update: W' = W + (B @ A) × (alpha/r). Only B and A are trained (rank r matrices). For a 4096×4096 weight with r=16: LoRA trains 4096×16 + 16×4096 = 131k parameters instead of 16.8M.

3. **Q: What is QLoRA?**
   A: LoRA applied to a model that's been quantised to 4-bit (NF4) precision. The base model weights are stored in 4-bit, the LoRA adapters are in BF16. This allows fine-tuning a 7B model on a single 16GB GPU that would otherwise require ~28GB.

4. **Q: What is ChatML format?**
   A: The message format used to structure training data: `<|system|>...<|user|>...<|assistant|>...`. Each example has a system instruction, a user query, and the expected assistant response. This format is used for SFT training of instruction-following models.

5. **Q: What is an adapter in the context of LoRA?**
   A: The adapter is the collection of small matrices (A and B for each targeted layer) that are added to the frozen base model. Adapters are saved separately from the base model (~10-50MB vs. ~14GB for 7B base). Multiple adapters can be loaded onto the same base model for different tasks.

6. **Q: What is the difference between SFT and full fine-tuning?**
   A: Full fine-tuning updates all model weights (requires same VRAM as training from scratch). SFT (Supervised Fine-Tuning) with LoRA only updates the adapter matrices. SFT with LoRA uses ~10× less VRAM and is regularised by the frozen base weights.

7. **Q: How do you decide what rank r to use in LoRA?**
   A: Start with r=16. Higher rank captures more complex adaptations but risks overfitting with small datasets. Rule of thumb: r=8 for small datasets (<500 examples), r=16 for medium (500-5000), r=32-64 for large. Run the ablation (Challenge B) to find the sweet spot for your specific task.

**Intermediate**

8. **Q: Why is low-rank a reasonable assumption for fine-tuning?**
   A: The intrinsic dimensionality hypothesis: task-specific adaptations lie in a low-dimensional subspace of the full parameter space. Empirically, even very small rank (r=4) achieves most of the benefit of full fine-tuning for many tasks. The pre-trained model already has rich representations; fine-tuning is a small adjustment.

9. **Q: What is target_modules in LoRA and how do you choose which layers to adapt?**
   A: target_modules specifies which layer types receive LoRA adapters. Common: `["q_proj", "v_proj"]` (attention query and value projections). More modules = more parameters but better adaptation: also add `k_proj`, `o_proj` for better results, or include FFN layers (`gate_proj`, `up_proj`, `down_proj`) for factual adaptation.

10. **Q: What is the difference between alpha and r in LoRA, and how do they affect training?**
    A: r is the rank (dimensionality). alpha is the scaling factor: the effective update is (alpha/r) × BA. If alpha=r (e.g., both=16), scaling=1. If alpha=2r (alpha=32, r=16), updates are doubled. Setting alpha=r is the most common default. alpha effectively controls the learning rate for the LoRA update.

11. **Q: How does NF4 quantisation work in QLoRA?**
    A: Normal Float 4-bit: quantises weights using a data type optimised for neural network weight distributions (near-normal). Instead of uniform spacing (standard INT4), NF4 has more levels near zero where most weights cluster, and fewer at extremes. Reduces model to ~2 bits/weight effective with minimal accuracy loss.

12. **Q: What is catastrophic forgetting in fine-tuning and how does LoRA help?**
    A: Full fine-tuning can overwrite general capabilities (catastrophic forgetting) when the task dataset is small. LoRA avoids this: the base weights are frozen, so general knowledge is preserved. Only the adapter learns the new task. After training, you can also switch tasks by swapping adapters.

13. **Q: What data quality principles matter most for SFT?**
    A: (1) 1000 high-quality examples > 100,000 low-quality ones (Lima paper). (2) Diversity: avoid near-duplicates. (3) Correctness: wrong answers corrupt the model faster than they help it. (4) Format consistency: inconsistent formatting leads to inconsistent outputs. (5) Coverage: ensure all task variations are represented.

**Advanced**

14. **Q: Derive the parameter count reduction from LoRA for LLaMA-2 7B.**
    A: LLaMA-2 7B has 32 transformer layers. Each layer has: q_proj (4096×4096), k_proj, v_proj, o_proj, and FFN layers. With r=16, targeting q+v: params per layer = 2 × (4096×16 + 16×4096) = 262k. Total LoRA params = 32 × 262k = 8.4M. Total model params = 7B. LoRA fraction = 8.4M/7B = 0.12% of parameters.

15. **Q: What is the merge_and_unload operation and when should you use it?**
    A: merge_and_unload() adds the LoRA update (BA) directly into the base weights (W' = W + B@A) and removes the adapter modules. Result: a merged model that runs at full speed (no overhead) but can no longer be swapped. Use when: deploying for inference (removes adapter overhead); sharing the full model.

16. **Q: How would you fine-tune a model when you only have 100 examples?**
    A: (1) Use few-shot prompting first — 100 examples may be sufficient for prompting. (2) If fine-tuning: use very low rank (r=4), strong regularisation (LoRA dropout=0.1, weight_decay=0.01), small learning rate (1e-4), only 1-2 epochs. (3) Data augmentation: use GPT-4 to generate variations. (4) Start from an instruction-tuned model, not a base model.

17. **Q: What is the packing strategy in SFTTrainer and why does it improve GPU utilisation?**
    A: Packing fills each batch sequence to the maximum length by concatenating multiple training examples (with EOS tokens between them). Without packing, short examples waste tokens with padding. With packing, GPU utilisation is much higher. TrainingArguments: `packing=True` in SFTTrainer.

18. **Q: How do you prevent overfitting during fine-tuning?**
    A: (1) Validation loss monitoring — stop when val loss increases. (2) Low learning rate (1e-4 to 3e-5). (3) LoRA dropout (0.05-0.1). (4) Small number of epochs (1-3 for SFT). (5) Low rank r. (6) Diverse dataset with minimal near-duplicates. (7) Perplexity on held-out examples.

19. **Q: What is Flash Attention 2 and why is it important for fine-tuning?**
    A: FlashAttention 2 computes attention without materialising the full N×N matrix (tiles in SRAM). For fine-tuning: (1) allows much longer sequences (crucial for long document fine-tuning); (2) faster training (IO-bound reduction); (3) constant memory growth with sequence length. Always enable with `use_flash_attention_2=True`.

20. **Q: How would you create a high-quality instruction dataset using GPT-4?**
    A: (1) Write 50 diverse seed instructions manually. (2) Use GPT-4 to generate 1000+ variations with `temperature=1.0` for diversity. (3) Use Self-Instruct or Alpaca-style pipeline. (4) Run quality filter: length, format, semantic deduplication, code validation. (5) Manually review 10% for quality. (6) Use GPT-4 as judge to score all examples (reject below 3/5).

---

### 7. Self-Assessment Quiz

- [ ] Write the LoRA forward pass formula.
- [ ] What is the parameter count for LoRA with r=16, applied to a 4096×4096 weight?
- [ ] What is the difference between r and alpha in LoRA?
- [ ] What is target_modules and what do you typically target?
- [ ] What is NF4 quantisation?
- [ ] What is the difference between SFT and full fine-tuning?
- [ ] What is catastrophic forgetting and how does LoRA help?
- [ ] What is the ChatML format?
- [ ] What does merge_and_unload() do?
- [ ] What VRAM does QLoRA need for a 7B model?
- [ ] Name 3 data quality principles for SFT.
- [ ] What is packing in SFTTrainer?
- [ ] What is perplexity and is lower better or worse?
- [ ] How do you push LoRA adapters to HuggingFace Hub?
- [ ] What is BitsAndBytesConfig?
- [ ] What is the intrinsic dimensionality hypothesis?
- [ ] What does LoRA dropout do?
- [ ] What is the difference between LoRA and prefix tuning?
- [ ] When would you fine-tune vs. use RAG?
- [ ] What is a checkpoint in fine-tuning training?
- [ ] What does Flash Attention 2 do to memory usage?
- [ ] How do you detect if your model has overfit to the fine-tuning data?
- [ ] What is self-instruct and how is it used for data generation?
- [ ] What is LLaMA-3.1 8B and why is it a good starting point?
- [ ] What is the difference between unsloth and vanilla HuggingFace training?

**Scoring**: 22–25 ✅ = Ready. 17–21 = Review weak areas. Below 17 = Spend more time on Phase 7a.

---

### 8. Common Mistakes

| Mistake | Why It Happens | How to Avoid |
|---------|---------------|--------------|
| Fine-tuning when prompting would work | Excitement about fine-tuning | Always try prompting first; fine-tune only when prompting demonstrably fails |
| Not validating dataset before training | Eager to start | Run quality filter; a few bad examples can corrupt training more than thousands of good ones |
| Using too high a rank | More is better instinct | r=16 is sufficient for most tasks; higher ranks risk overfitting on small datasets |
| Not tracking validation loss | Only track training loss | Training loss always decreases; validation loss tells you if you're overfitting |
| Training too many epochs | More training = better | For instruction tuning, 1-3 epochs is usually optimal; more causes overfitting |
| Forgetting to set pad_token_id | Error on the first batch | Always check: `tokenizer.pad_token is not None`; set to `eos_token` if None |
| Not testing the merged model before deploying | Merge seems straightforward | Always run inference on the merged model; floating-point precision can cause tiny differences |

---

### 9. Readiness Criteria

You are ready for Phase 7b when **all** of the following are true:

- [ ] I implemented LoRALinear from scratch with merge_weights() (Challenge A)
- [ ] I collected and quality-filtered a 500+ example dataset (Challenge C)
- [ ] I fine-tuned LLaMA-3.1 8B with QLoRA on Colab
- [ ] My fine-tuned model outperforms the base model on domain questions
- [ ] I pushed adapters to HuggingFace Hub
- [ ] I scored 22/25 or higher on the Self-Assessment Quiz
- [ ] I can answer at least 16/20 Interview Questions correctly
- [ ] I can explain the LoRA math from memory (ΔW = BA × alpha/r)

---

### 10. Revision Summary

```
LORA MATHEMATICS
─────────────────────────────────────────────────────
W' = W_frozen + (B @ A) * (alpha / r)
B: d×r  (initialised to zeros)
A: r×k  (initialised from Gaussian)
Params saved: r(d+k) vs. d×k  →  ~100× fewer for r=16

QLORA STACK
─────────────────────────────────────────────────────
Base model:  4-bit NF4 (BitsAndBytes) — frozen
Adapters:    BF16 LoRA matrices — trainable
VRAM 7B:     ~6-8GB (vs. ~28GB for full BF16)

TRAINING PIPELINE
─────────────────────────────────────────────────────
1. Format data as ChatML (system/user/assistant)
2. Quality filter: length, dedup, correctness
3. BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
4. LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj","v_proj"])
5. SFTTrainer(model, train_data, packing=True, max_seq_length=2048)
6. Evaluate: perplexity, LLM-as-judge on held-out set
7. model.merge_and_unload() → push to HuggingFace Hub

DECISION FRAMEWORK
─────────────────────────────────────────────────────
Use RAG if:        needs up-to-date or private knowledge
Use prompting if:  task is achievable with good examples
Use fine-tuning if: consistent style/format, domain vocabulary,
                    latency-critical, or prompting repeatedly fails
```

---

### 11. Next Phase Prerequisites

**What Phase 7b (RLHF & DPO) requires from Phase 7a:**

| Phase 7a Concept | How Phase 7b Uses It |
|-----------------|----------------------|
| SFT pipeline | RLHF/DPO starts from an SFT-trained model (your Project 18 output) |
| LoRA adapters | DPO training uses the same QLoRA setup |
| Dataset format | Preference dataset extends ChatML: adds chosen/rejected fields |
| Model evaluation | Comparing SFT vs. DPO requires the same evaluation framework |
| TRL library | DPOTrainer is from the same library as SFTTrainer |
| VRAM budgeting | DPO requires 2 model copies; QLoRA enables this on consumer hardware |

**The critical dependency**: DPO starts from your SFT-trained model. The quality of your Phase 7a fine-tune directly determines the ceiling of your DPO-aligned model. A well-trained SFT model + DPO is powerful; a poorly trained SFT model can't be rescued by DPO.

---

*Phase 7a | Part of the [GenAI Engineer Roadmap](./00_README.md)*
