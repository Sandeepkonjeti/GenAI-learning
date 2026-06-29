# Phase 8: Advanced Topics
**Months 16–17 | Difficulty: 9/10 | Label: 🟡 Learn Later**

> **Previous Phase**: [Phase 7b — RLHF & DPO](./08_Phase7_Part2_RLHF_and_DPO.md)  
> **Next Phase**: [Phase 9 — Production AI Engineering](./10_Phase9_Production_AI_Engineering.md)

---

## Table of Contents

- [Phase Overview](#phase-overview)
- [Topic 1: Multimodal AI](#topic-1-multimodal-ai)
  - [How Vision Transformers Work](#how-vision-transformers-work)
  - [CLIP: Connecting Text and Images](#clip-connecting-text-and-images)
  - [Multimodal LLMs: LLaVA Architecture](#multimodal-llms-llava-architecture)
  - [Whisper: Speech Recognition](#whisper-speech-recognition)
- [Topic 2: Diffusion Models](#topic-2-diffusion-models)
  - [The Forward and Reverse Process](#the-forward-and-reverse-process)
  - [Training the Denoiser (U-Net)](#training-the-denoiser-u-net)
  - [Classifier-Free Guidance](#classifier-free-guidance)
  - [Latent Diffusion Models (Stable Diffusion)](#latent-diffusion-models-stable-diffusion)
- [Topic 3: GPU Fundamentals for AI](#topic-3-gpu-fundamentals-for-ai)
  - [GPU Architecture and Memory Hierarchy](#gpu-architecture-and-memory-hierarchy)
  - [The Roofline Model](#the-roofline-model)
  - [FlashAttention](#flashattention)
  - [GPU Profiling with nvidia-smi and PyTorch Profiler](#gpu-profiling)
- [Topic 4: Inference Optimization](#topic-4-inference-optimization)
  - [Quantization: GPTQ and AWQ](#quantization-gptq-and-awq)
  - [Speculative Decoding](#speculative-decoding)
  - [PagedAttention and vLLM](#pagedattention-and-vllm)
  - [Continuous Batching](#continuous-batching)
  - [torch.compile](#torchcompile)
- [Resources](#resources)
- [Projects](#projects)
- [Common Mistakes](#common-mistakes)
- [Mastery Checklist](#mastery-checklist)

---

## Phase Overview

| Attribute | Details |
|-----------|---------|
| **Duration** | Months 16–17 (4 weeks) |
| **Daily Time** | 1–2 hours |
| **Difficulty** | 9/10 |
| **Label** | 🟡 Learn Later |
| **Prerequisites** | All previous phases; PyTorch; transformer architecture |
| **Outcome** | Understand multimodal AI; can deploy optimized LLM inference; can read GPU metrics |

**Focus recommendation**: If time is limited, prioritize topics 3 (GPU Fundamentals) and 4 (Inference Optimization). These have the highest practical value in production settings.

---

## Topic 1: Multimodal AI

### How Vision Transformers Work

ViT (Vision Transformer) applies the transformer architecture directly to images by treating image patches as tokens.

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Convert an image into a sequence of patch embeddings.
    
    For a 224×224 image with patch_size=16:
    - Number of patches = (224/16)^2 = 196
    - Each patch = 16×16×3 = 768 flattened pixels
    - Linear projection to d_model
    """
    def __init__(self, image_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, d_model: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        
        # Single conv layer extracts and flattens patches simultaneously
        self.projection = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, height, width)
        x = self.projection(x)  # (batch, d_model, n_patches_h, n_patches_w)
        x = x.flatten(2)        # (batch, d_model, n_patches)
        x = x.transpose(1, 2)  # (batch, n_patches, d_model)
        return x

class VisionTransformer(nn.Module):
    """
    A minimal ViT implementation.
    Key difference from text transformer: 2D spatial structure of input.
    """
    def __init__(self, image_size=224, patch_size=16, d_model=768, n_heads=12, 
                 n_layers=12, n_classes=1000):
        super().__init__()
        n_patches = (image_size // patch_size) ** 2
        
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, d_model)
        
        # CLS token: a learnable token prepended to the patch sequence
        # Its representation after all transformer layers is used for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Positional embeddings for all patches + CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        
        # Standard transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, pre_norm=True)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Patch embeddings
        x = self.patch_embed(x)  # (B, n_patches, d_model)
        
        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, n_patches+1, d_model)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks (no causal mask — full bidirectional attention)
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0, :]  # (B, d_model)
        return self.head(cls_output)

# ViT variants:
vit_variants = {
    "ViT-S/16": {"d_model": 384, "n_heads": 6, "n_layers": 12, "patch": 16},
    "ViT-B/16": {"d_model": 768, "n_heads": 12, "n_layers": 12, "patch": 16},
    "ViT-L/16": {"d_model": 1024, "n_heads": 16, "n_layers": 24, "patch": 16},
    "ViT-H/14": {"d_model": 1280, "n_heads": 16, "n_layers": 32, "patch": 14},
}
```

---

### CLIP: Connecting Text and Images

```python
# CLIP (Contrastive Language-Image Pretraining, OpenAI, 2021)
# Trains an image encoder and text encoder jointly on 400M (image, caption) pairs
# Objective: make (image, its_caption) embeddings close; make mismatched pairs far apart

# The contrastive loss:
# For a batch of N (image, text) pairs:
# - N correct pairs (on diagonal)
# - N^2 - N incorrect pairs (off diagonal)
# Maximize cosine similarity for correct pairs, minimize for incorrect pairs

import torch.nn.functional as F

def clip_loss(image_features: torch.Tensor, text_features: torch.Tensor, 
              temperature: float = 0.07) -> torch.Tensor:
    """
    CLIP contrastive loss.
    
    image_features: (batch, d)  — L2-normalized image embeddings
    text_features:  (batch, d)  — L2-normalized text embeddings
    """
    # Compute similarity matrix: (batch, batch)
    similarity = (image_features @ text_features.T) / temperature
    
    # Labels: the i-th image matches the i-th text
    labels = torch.arange(len(similarity), device=similarity.device)
    
    # Symmetric cross-entropy loss
    loss_img = F.cross_entropy(similarity, labels)            # Image → Text
    loss_txt = F.cross_entropy(similarity.T, labels)          # Text → Image
    
    return (loss_img + loss_txt) / 2

# Using CLIP for zero-shot image classification:
from transformers import CLIPModel, CLIPProcessor
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
from PIL import Image
import requests

# Load an image
image = Image.open(requests.get("https://example.com/cat.jpg", stream=True).raw)

# Zero-shot classification: no training required!
class_names = ["a cat", "a dog", "a bird", "a car"]
inputs = processor(
    text=class_names,
    images=image,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # (1, n_classes)
    probs = logits_per_image.softmax(dim=1)

for cls, prob in zip(class_names, probs[0]):
    print(f"{cls}: {prob:.3f}")
```

---

### Multimodal LLMs: LLaVA Architecture

```python
# LLaVA (Large Language and Vision Assistant) architecture:
# 
# Image → CLIP Vision Encoder → Vision Features → Linear Projection → 
#                                                                      ↘
#                                                                   LLM (LLaMA)  → Response
#                                                                      ↗
# Text  → Tokenizer → Text Embeddings ──────────────────────────────
#
# Key insight: image patches are treated as "visual tokens" — 
# they're just added to the token sequence before the text

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch

# Load LLaVA
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

image = Image.open("chart.png")
prompt = "[INST] <image>\nWhat trends do you see in this chart? [/INST]"

inputs = processor(prompt, image, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=200)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

---

### Whisper: Speech Recognition

```python
# Whisper (OpenAI, 2022): Transformer for speech recognition
# Architecture: Encoder-Decoder
# Input: Log-Mel spectrogram (80-channel, 30-second windows)
# Output: Text transcription

import whisper
import numpy as np

model = whisper.load_model("base")  # tiny, base, small, medium, large

# Transcribe audio
result = model.transcribe("audio.mp3")
print(result["text"])
print(result["language"])  # Auto-detected language

# For streaming transcription:
import torch
import librosa

def transcribe_stream(audio_path: str, chunk_duration: float = 5.0) -> str:
    """Transcribe long audio in chunks."""
    audio, sr = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz
    
    chunks = []
    chunk_size = int(chunk_duration * sr)
    
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start:start + chunk_size]
        
        # Pad to 30 seconds (Whisper requirement)
        if len(chunk) < sr * 30:
            chunk = np.pad(chunk, (0, sr * 30 - len(chunk)))
        
        mel = whisper.log_mel_spectrogram(torch.tensor(chunk))
        result = model.decode(mel, whisper.DecodingOptions(language="en"))
        chunks.append(result.text)
    
    return " ".join(chunks)
```

---

## Topic 2: Diffusion Models

### The Forward and Reverse Process

```python
import torch
import torch.nn as nn
import numpy as np

# Diffusion models learn to reverse a noising process:
# 
# Forward process (noising, fixed):
# x_0 (clean image) → x_1 (slightly noisy) → x_2 → ... → x_T (pure noise)
# x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε,  ε ~ N(0, I)
#
# Reverse process (denoising, learned):
# x_T (pure noise) → ... → x_1 → x_0 (clean image)
# p_θ(x_{t-1}|x_t) = learned by neural network (U-Net)
#
# Training objective:
# Predict the noise ε that was added at timestep t
# L = E[||ε - ε_θ(x_t, t)||²]  (noise prediction MSE loss)

class NoiseScheduler:
    """
    Linear noise schedule for DDPM.
    Defines how much noise to add at each timestep.
    """
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = num_timesteps
        
        # β_t: noise schedule (how much noise to add at step t)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # α_t = 1 - β_t
        self.alphas = 1 - self.betas
        
        # ᾱ_t = cumulative product of α_t: allows direct jump from x_0 to x_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Useful pre-computations
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    
    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to x_0 at timestep t.
        Returns noisy image x_t and the noise ε that was added.
        """
        noise = torch.randn_like(x_0)
        
        # q(x_t | x_0) = N(sqrt(ᾱ_t) * x_0, (1-ᾱ_t) * I)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise
    
    def reverse_step(self, x_t: torch.Tensor, predicted_noise: torch.Tensor, t: int) -> torch.Tensor:
        """One denoising step (DDPM sampler)."""
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        
        # Predict x_0 from x_t and predicted noise
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        
        # Compute mean of p(x_{t-1}|x_t)
        if t > 0:
            noise = torch.randn_like(x_t)
            alpha_bar_prev = self.alphas_cumprod[t - 1]
        else:
            noise = torch.zeros_like(x_t)
            alpha_bar_prev = torch.ones(1)
        
        mean = (
            (torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)) * x_0_pred +
            (torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)) * x_t
        )
        
        variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        x_prev = mean + torch.sqrt(variance) * noise
        
        return x_prev
```

---

### Classifier-Free Guidance

```python
# Classifier-free guidance (CFG) makes generated images better match text prompts
# 
# Key idea: run the denoiser twice per step:
# 1. Conditioned on text prompt: ε_θ(x_t, t, prompt)
# 2. Unconditioned (empty prompt): ε_θ(x_t, t, "")
# 3. Interpolate: guided_noise = uncond + guidance_scale * (cond - uncond)
# 
# guidance_scale = 7.5 is common (1 = no guidance, 15 = very strong)
# Higher guidance = more prompt adherence but less diversity

def cfg_step(
    denoiser,
    x_t: torch.Tensor,
    t: int,
    text_embedding: torch.Tensor,
    empty_embedding: torch.Tensor,
    guidance_scale: float = 7.5
) -> torch.Tensor:
    """One classifier-free guidance denoising step."""
    
    # Batch both to run in a single forward pass (2x efficient)
    batch = torch.cat([x_t, x_t], dim=0)
    embeddings = torch.cat([empty_embedding, text_embedding], dim=0)
    timesteps = torch.tensor([t, t])
    
    # Single forward pass for both conditioned and unconditioned
    noise_pred = denoiser(batch, timesteps, embeddings)
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    
    # Guidance: push in the direction of the conditioned prediction
    guided_noise = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    
    return guided_noise
```

---

### Latent Diffusion Models (Stable Diffusion)

```python
# Stable Diffusion solves the computational problem of pixel-space diffusion
# Training on 512×512 images at the pixel level is extremely expensive
#
# Key innovation: run diffusion in a compressed LATENT space
#   1. Encode image to latents: x_0 (512×512×3) → z_0 (64×64×4)  [8x compression]
#   2. Run diffusion in latent space (much cheaper)
#   3. Decode latents back to pixels: z_0 (64×64×4) → x_0 (512×512×3)
#
# Components:
# - VAE (Variational Autoencoder): encode/decode between pixel and latent space
# - U-Net: the denoiser, operates in latent space
# - CLIP Text Encoder: encodes text prompts to condition the U-Net
# - Noise Scheduler: DDPM or DDIM

from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Generate image
image = pipe(
    prompt="A photorealistic painting of a sunset over the Databricks logo, digital art",
    negative_prompt="blurry, distorted, low quality",
    num_inference_steps=50,  # DDIM steps (20-50 typical)
    guidance_scale=7.5,
    height=512,
    width=512,
).images[0]

image.save("generated.png")

# SDXL (higher quality, 1024×1024):
from diffusers import StableDiffusionXLPipeline
pipe_xl = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
```

---

## Topic 3: GPU Fundamentals for AI

### GPU Architecture and Memory Hierarchy

```
GPU Memory Hierarchy (A100 80GB example):

┌─────────────────────────────────────────┐
│            L2 Cache (40 MB)             │  Fast, shared across SMs
├─────────────────────────────────────────┤
│          HBM2e (High Bandwidth Memory)  │  80 GB, ~2 TB/s bandwidth
│          (Global Memory)               │
└─────────────────────────────────────────┘
           ↑
           │ PCIe/NVLink to CPU
           ↓
┌─────────────────────────────────────────┐
│              CPU RAM                    │  Much slower than HBM
└─────────────────────────────────────────┘
```

```python
import torch

# Check GPU memory usage
def gpu_memory_report():
    if not torch.cuda.is_available():
        print("No GPU available")
        return
    
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        allocated = torch.cuda.memory_allocated(i) / 1e9
        
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total:     {total:.1f} GB")
        print(f"  Reserved:  {reserved:.1f} GB")
        print(f"  Allocated: {allocated:.1f} GB")
        print(f"  Free:      {total - reserved:.1f} GB")

# Memory breakdown for fine-tuning a 7B model (float32):
memory_breakdown = {
    "Model weights (BF16)":        "14 GB",
    "Gradients (BF16)":            "14 GB",
    "Optimizer states (Adam BF16)":"28 GB",  # 2x model size for momentum/variance
    "Activations (gradient ckpt)": "2-4 GB",
    "Total":                       "~58-60 GB",
    "Required GPU":                "A100 80GB or 2x A40"
}

# With QLoRA:
qlora_memory = {
    "Model weights (4-bit)":       "3.5 GB",
    "LoRA adapters (BF16)":        "0.2 GB",
    "Optimizer states (8-bit Adam)":"0.4 GB",
    "Activations":                  "1-2 GB",
    "Total":                        "~5-6 GB",
    "Required GPU":                 "RTX 3090/4090 (24 GB)"
}
```

---

### The Roofline Model

```python
# The Roofline Model: understand if an operation is memory-bound or compute-bound
# 
# Arithmetic Intensity (AI) = FLOPs / bytes_accessed
# 
# If AI < machine's arithmetic intensity threshold:
#   → MEMORY BOUND: system is waiting for data, not compute
#   → Optimization: reduce memory accesses (fused kernels, tiling)
# 
# If AI > machine's arithmetic intensity threshold:
#   → COMPUTE BOUND: system is utilizing compute fully
#   → Optimization: reduce FLOPs, use better algorithms

# A100 80GB specs:
A100_SPECS = {
    "peak_fp16_tflops": 312,      # 312 TFLOPS for tensor cores
    "memory_bandwidth_tbs": 2.0,   # 2 TB/s HBM bandwidth
    "arithmetic_intensity_ridge": 312 / 2.0  # 156 FLOPs/byte
}

# Transformer attention: is it memory or compute bound?
def analyze_attention_intensity(seq_len: int, d_model: int, batch_size: int = 1) -> dict:
    """Analyze arithmetic intensity of attention computation."""
    
    n_heads = d_model // 64
    d_k = 64
    
    # FLOPs for QK^T matmul: 2 * batch * n_heads * seq * seq * d_k
    qkt_flops = 2 * batch_size * n_heads * seq_len * seq_len * d_k
    
    # Bytes accessed: Q, K, V, output
    bytes_accessed = (
        4 * batch_size * n_heads * seq_len * d_k  # Q, K, V, output
        * 2  # float16
    )
    
    ai = qkt_flops / bytes_accessed
    bound = "memory-bound" if ai < A100_SPECS["arithmetic_intensity_ridge"] else "compute-bound"
    
    return {
        "seq_len": seq_len,
        "flops": f"{qkt_flops/1e9:.1f}G",
        "bytes": f"{bytes_accessed/1e9:.2f}GB",
        "arithmetic_intensity": f"{ai:.1f} FLOPs/byte",
        "bottleneck": bound
    }

# Short sequences are memory-bound (common in generation)
# Long sequences are compute-bound (common in prefill)
for seq in [128, 512, 2048, 8192]:
    print(analyze_attention_intensity(seq, 4096))
```

---

### FlashAttention

```python
# FlashAttention (Dao et al., 2022) makes attention memory-efficient
# 
# Standard attention:
# 1. Compute S = QK^T           → requires O(seq²) memory to store
# 2. Compute P = softmax(S)     → requires O(seq²) memory
# 3. Compute output = PV        → final output
# 
# Problem: for seq=4096, we need 4096^2 = 16M floats per head ≈ 128 MB per layer!
# 
# FlashAttention:
# - Compute attention in TILES that fit in SRAM (on-chip memory, ~20 MB on A100)
# - Use online softmax normalization to handle tiles without seeing all at once
# - Never materialize the full attention matrix
# - Result: O(seq) memory, same O(seq²) compute, but IO-optimal
# 
# ~2-4x faster than standard attention on A100
# ~5-8x lower memory than standard attention

# Using FlashAttention in PyTorch:
import torch
import torch.nn.functional as F

# PyTorch 2.0+ has FlashAttention built in via scaled_dot_product_attention
def flash_attention(Q, K, V, mask=None, dropout_p=0.0):
    """
    Use PyTorch's built-in FlashAttention (or fallback to standard).
    Automatically selects the best kernel (FlashAttention, memory-efficient, or math).
    """
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,        # Use FlashAttention if available
        enable_math=True,         # Fallback to standard math
        enable_mem_efficient=True # Memory-efficient attention as fallback
    ):
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=dropout_p)

# Verify it works:
batch, heads, seq, d_k = 2, 12, 1024, 64
Q = torch.randn(batch, heads, seq, d_k, device="cuda", dtype=torch.float16)
K = torch.randn(batch, heads, seq, d_k, device="cuda", dtype=torch.float16)
V = torch.randn(batch, heads, seq, d_k, device="cuda", dtype=torch.float16)

with torch.cuda.amp.autocast():
    output = flash_attention(Q, K, V)
print(f"Output shape: {output.shape}")  # (2, 12, 1024, 64)
```

---

### GPU Profiling

```python
# nvidia-smi: quick GPU status
# nvidia-smi → see all GPUs, VRAM usage, power, temperature
# nvidia-smi dmon → continuous monitoring
# nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 1

# PyTorch built-in profiler
from torch.profiler import profile, record_function, ProfilerActivity

def profile_model(model, inputs):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("model_inference"):
            output = model(inputs)
    
    # Print top operators by GPU time
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15
    ))
    
    # Export for visualization in TensorBoard or chrome://tracing
    prof.export_chrome_trace("trace.json")
    
    return prof
```

---

## Topic 4: Inference Optimization

### Quantization: GPTQ and AWQ

```python
# Post-training quantization: reduce precision AFTER training
# 
# INT8 quantization: 2x memory, ~2x faster
# INT4 quantization: 4x memory, 3-4x faster
# 
# Challenge: standard rounding to INT4 causes large accuracy loss
# 
# GPTQ (Frantar et al., 2022):
# - Uses second-order information (Hessian) to minimize quantization error
# - Quantizes layer by layer; each weight is rounded to minimize output error
# 
# AWQ (Lin et al., 2023):
# - Identifies "salient" weights (small %, but critical to accuracy)
# - Keeps salient weights at higher precision or scales them carefully
# - Faster than GPTQ, similar quality

# Loading GPTQ quantized models:
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Load a pre-quantized GPTQ model (from HuggingFace Hub)
model_name = "TheBloke/Llama-2-7B-GPTQ"  # 4-bit GPTQ quantized
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoGPTQForCausalLM.from_quantized(
    model_name,
    use_safetensors=True,
    device="cuda:0",
    inject_fused_attention=True  # Use faster fused kernels
)

# AWQ models:
from awq import AutoAWQForCausalLM

model_name = "TheBloke/Llama-2-7B-AWQ"
model = AutoAWQForCausalLM.from_quantized(
    model_name,
    fuse_layers=True,
    trust_remote_code=False
)
```

---

### Speculative Decoding

```python
# Speculative Decoding (Leviathan et al., 2023; Chen et al., 2023)
# 
# Key insight: decoding is memory-bandwidth-limited, not compute-limited
# GPUs can compute more tokens per second than one — but autoregressive decoding forces one at a time
# 
# Speculative decoding exploits this:
# 1. DRAFT: small fast model (e.g., LLaMA-7B) generates k=5 draft tokens autoregressively
# 2. VERIFY: large model (e.g., LLaMA-70B) processes all k tokens in PARALLEL (one forward pass)
# 3. ACCEPT: if draft tokens match large model's distribution → keep them (free tokens!)
#            if not → reject at the divergence point, resample from large model
# 
# Result: ~2-3x speedup for 70B model with no quality loss
# (draft tokens are accepted ~80% of the time in practice)

# The acceptance criterion (stochastic):
def speculative_accept(draft_prob: float, target_prob: float) -> tuple[bool, float]:
    """
    Determine whether to accept a draft token.
    
    Accept probability = min(1, target_prob / draft_prob)
    
    This ensures the final distribution matches the target model exactly.
    """
    accept_prob = min(1.0, target_prob / draft_prob)
    accepted = torch.rand(1).item() < accept_prob
    
    # If rejected, resample from corrected distribution
    correction_prob = max(0, target_prob - draft_prob * accept_prob)
    
    return accepted, correction_prob

# Libraries that implement speculative decoding:
# - Hugging Face generate() with assistant_model parameter
# - vLLM speculative decoding
# - llama.cpp speculative decoding

# Example with HF:
draft_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
target_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B")

output = target_model.generate(
    inputs,
    assistant_model=draft_model,  # HF's built-in speculative decoding
    max_new_tokens=200,
)
```

---

### PagedAttention and vLLM

```python
# vLLM (Kwon et al., 2023) — the state-of-the-art LLM serving engine
# 
# Key innovation: PagedAttention
# Problem: KV cache has variable size per request; traditional allocation is wasteful
# Solution: Manage KV cache like OS virtual memory pages
#   - Fixed-size "pages" allocated dynamically
#   - No internal fragmentation (waste within allocated memory)
#   - Result: 60-90% more requests served per GPU vs. HuggingFace

# Using vLLM for high-throughput serving:
from vllm import LLM, SamplingParams

# Initialize engine (loads model and allocates KV cache)
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,    # Number of GPUs (for tensor parallelism)
    gpu_memory_utilization=0.9, # Use 90% of VRAM for KV cache
    max_num_batched_tokens=4096,# Max tokens per batch
    quantization="awq",        # Optional: use quantized model
)

# Sampling parameters
params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Batch inference (much faster than single requests)
prompts = [
    "What is machine learning?",
    "Explain the transformer architecture.",
    "What is RAG?"
]

outputs = llm.generate(prompts, params)
for output in outputs:
    print(f"Prompt: {output.prompt[:50]}...")
    print(f"Output: {output.outputs[0].text[:100]}...\n")

# vLLM OpenAI-compatible server:
# python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct
# Then use standard OpenAI client against http://localhost:8000
```

---

### torch.compile

```python
# torch.compile (PyTorch 2.0+): JIT compilation for 1.5-3x speedup
# 
# Uses TorchInductor backend to:
# - Fuse operations (avoid materializing intermediate tensors)
# - Vectorize computations
# - Compile to efficient GPU kernels
# 
# Modes:
# "default"    - balanced optimization
# "reduce-overhead" - minimize Python overhead (good for inference)
# "max-autotune"    - maximum performance (slow compile, fast runtime)

import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

# Compile the model
compiled_model = torch.compile(model, mode="reduce-overhead")

# First call: slow (compilation happening)
# Subsequent calls: fast (using compiled kernels)
inputs = tokenizer("Hello world", return_tensors="pt")

import time

# Warm up
with torch.no_grad():
    _ = compiled_model(**inputs)

# Measure
n_runs = 100
start = time.time()
with torch.no_grad():
    for _ in range(n_runs):
        _ = compiled_model(**inputs)
compiled_time = (time.time() - start) / n_runs

# Compare to uncompiled
start = time.time()
with torch.no_grad():
    for _ in range(n_runs):
        _ = model(**inputs)
baseline_time = (time.time() - start) / n_runs

print(f"Baseline:  {baseline_time*1000:.1f} ms/run")
print(f"Compiled:  {compiled_time*1000:.1f} ms/run")
print(f"Speedup:   {baseline_time/compiled_time:.2f}x")
```

---

## Resources

| Rank | Resource | Type | Cost | Why |
|------|----------|------|------|-----|
| 1 | [Illustrated Stable Diffusion (Jay Alammar)](https://jalammar.github.io/illustrated-stable-diffusion/) | Blog | Free | Best visual explanation of diffusion models |
| 2 | [FlashAttention Paper (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) | Paper | Free | Essential for understanding attention optimization |
| 3 | [vLLM Paper (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180) | Paper | Free | PagedAttention and LLM serving |
| 4 | [An Introduction to Vision Transformers](https://arxiv.org/abs/2010.11929) | Paper | Free | ViT original paper (well-written) |
| 5 | [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) | Docs | Free | Learn to profile GPU operations |
| 6 | [Lilian Weng "What are Diffusion Models?"](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | Blog | Free | Mathematical deep dive into diffusion |

---

## Projects

### Project 20: Multimodal Document QA
**Difficulty**: 8/10 | **Time**: 2 weeks

Build a system that:
1. Accepts PDFs with images and charts
2. Uses LLaVA to extract information from charts/diagrams
3. Combines visual and text information in a RAG pipeline
4. Answers questions that require both text and image understanding

---

### Project 21: Optimized LLM Inference Server
**Difficulty**: 8/10 | **Time**: 1 week

Build a production inference server:
1. Deploy LLaMA 3.1 8B with vLLM
2. Implement 4-bit AWQ quantization
3. Benchmark: tokens/second, latency p50/p95/p99, VRAM usage
4. Compare: HuggingFace vs. vLLM, FP16 vs. AWQ
5. Serve via OpenAI-compatible API

---

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Not profiling before optimizing | Optimizing the wrong thing | Always profile first; find the actual bottleneck |
| Using fp32 in inference | 2x slower, 2x more memory | Always use fp16 or bfloat16 for inference |
| Not using FlashAttention | OOM for long contexts | Use torch.sdp with FlashAttention enabled |
| Loading model to CPU then moving | Very slow; double memory | Use device_map="auto" or specify device at load time |
| Running vLLM with single requests | Much slower than batching | vLLM shines with concurrent requests; use async API |

---

## Mastery Checklist

### Multimodal
- [ ] Understand ViT patch embedding (images as tokens)
- [ ] Understand CLIP contrastive training
- [ ] Used LLaVA to answer questions about images
- [ ] Used Whisper to transcribe audio

### Diffusion
- [ ] Understand the forward noising process and reverse denoising
- [ ] Know what classifier-free guidance does and how to tune it
- [ ] Generated images with Stable Diffusion (local or API)

### GPU Fundamentals
- [ ] Can read nvidia-smi output meaningfully
- [ ] Understand the difference between memory-bound and compute-bound operations
- [ ] Know what FlashAttention optimizes (IO operations, not FLOPs)
- [ ] Profiled a model with PyTorch profiler

### Inference Optimization
- [ ] Understand GPTQ and AWQ (what they optimize, how they differ)
- [ ] Understand speculative decoding mechanism
- [ ] Deployed a model with vLLM
- [ ] Used torch.compile and measured speedup

---

## Moving to Phase 9

**Before proceeding to [Phase 9: Production AI Engineering](./10_Phase9_Production_AI_Engineering.md), confirm:**

- [ ] Comfortable with GPU memory management
- [ ] Can deploy an LLM with vLLM
- [ ] Understand the quantization tradeoffs

**Why Phase 9 comes next**: Individual model knowledge is necessary but not sufficient for production. Phase 9 teaches you how to build complete AI systems — with monitoring, evaluation, MLOps, security, and reliability.

---

## Phase Completion & Readiness Assessment

> Complete this assessment before Phase 9. Advanced topics separate competent AI engineers from experts — GPU optimisation and multimodal skills are increasingly required.

---

### 1. Knowledge Checklist

**Multimodal Models**
- [ ] Vision Transformer (ViT): patch embeddings, CLS token, positional encoding
- [ ] CLIP: contrastive pretraining, image-text alignment, zero-shot classification
- [ ] LLaVA: how vision encoder connects to LLM via projection layer
- [ ] Whisper: encoder-decoder for speech, spectrogram input, CTC vs. attention
- [ ] Why multimodal models are harder to align than text-only models

**Diffusion Models**
- [ ] Forward diffusion process: add noise progressively (Markov chain)
- [ ] Reverse diffusion: learn to denoise
- [ ] DDPM: denoising diffusion probabilistic model
- [ ] U-Net architecture: encoder, bottleneck, decoder, skip connections
- [ ] Guidance: classifier-free guidance (CFG) — the scale parameter
- [ ] Latent diffusion: why working in latent space is efficient
- [ ] Stable Diffusion: VAE + CLIP text encoder + U-Net + scheduler

**GPU Optimisation**
- [ ] GPU memory hierarchy: HBM, L2 cache, SRAM (shared memory), registers
- [ ] Memory bandwidth vs. compute-bound operations
- [ ] Roofline model: how to classify operations
- [ ] FlashAttention: tile-based computation, no O(N²) materialisation
- [ ] Quantisation: FP32 → BF16 → INT8 → GPTQ → AWQ (quality vs. speed)
- [ ] Speculative decoding: draft model + target model, acceptance criterion
- [ ] Continuous batching in vLLM: PagedAttention concept
- [ ] `torch.compile`: what it optimises (kernel fusion, memory layout)

---

### 2. Practical Skills Checklist

- [ ] Load CLIP and classify images zero-shot (without fine-tuning)
- [ ] Use LLaVA to answer questions about images
- [ ] Generate images with Stable Diffusion (vary CFG scale, steps)
- [ ] Profile a model with PyTorch profiler and identify the bottleneck
- [ ] Quantise a model with AWQ and compare quality vs. original
- [ ] Deploy a model with vLLM and benchmark throughput vs. HuggingFace
- [ ] Apply `torch.compile` and measure speedup
- [ ] Implement speculative decoding (draft + verify loop)

---

### 3. Coding Challenges

**Challenge A — CLIP Zero-Shot Classifier**
```python
# Build a zero-shot image classifier using CLIP:
# 1. Define 10 classes with text descriptions (not just labels)
# 2. Encode all text descriptions with CLIP text encoder
# 3. For each test image: encode with CLIP image encoder
# 4. Score = cosine similarity(image_embedding, text_embeddings)
# 5. Predict = argmax(scores)
# Test on: CIFAR-10 (1000 test images)
# Compare: CLIP zero-shot vs. ResNet-18 trained on CIFAR-10 full training set
# Report: accuracy, worst-class accuracy, confusion matrix
```

**Challenge B — vLLM Benchmark**
```python
# Deploy LLaMA-3.1 8B with both HuggingFace and vLLM:
# Benchmark at batch sizes: 1, 4, 8, 16, 32 (or as many as fit in VRAM)
# Metrics per config: tokens/sec, TTFT (p50, p95), TBT (p50, p95), VRAM
# Prompts: use a fixed set of 100 prompts at ~200 token context
# Plot: throughput (tokens/sec) vs. batch size for both backends
# Conclusion: at what batch size does vLLM's advantage kick in?
```

**Challenge C — Quantisation Quality Study**
```python
# Quantise LLaMA-3.1 8B to 4 precision levels:
# - BF16 (baseline)
# - INT8 (bitsandbytes)
# - 4-bit GPTQ
# - 4-bit AWQ
# Evaluate each on:
# 1. Perplexity (WikiText-2 test set)
# 2. 20 domain-specific questions (score 1-5 LLM-as-judge)
# 3. Speed: tokens/sec and latency p50
# 4. VRAM usage
# Table: precision × (perplexity, quality, speed, VRAM)
```

---

### 4. Mini Project

**Multimodal Document QA** (Project 20 from the roadmap):
- Process PDFs that contain charts, tables, and images
- Extract: text pages (PyMuPDF), images (PDFs → images)
- For images: use LLaVA to generate text descriptions
- Store all chunks (text + image descriptions) in a vector DB
- RAG pipeline: retrieve both text and image description chunks
- Test: 15 questions that require reading a chart or diagram to answer correctly

---

### 5. Capstone Project

**Inference Optimisation Study** (Project 21 + 27 combined):
- Base: LLaMA-3.1 8B on a single GPU
- Test 6 configurations: BF16 baseline, BF16+FA2, INT8, AWQ, AWQ+vLLM, AWQ+vLLM+compile
- Report: tokens/sec, p50 latency, VRAM, perplexity (quality loss)
- Roofline analysis: which operations are memory-bound vs. compute-bound?
- Recommendation: which configuration to use for each scenario:
  - Max throughput (batch serving)
  - Min latency (interactive)
  - Limited VRAM (consumer GPU)
  - Quality-critical (medical/legal)

---

### 6. Interview Questions

**Beginner**

1. **Q: What is CLIP and how does zero-shot classification work?**
   A: CLIP trains a vision encoder and text encoder jointly using contrastive loss on 400M image-text pairs. At inference: encode image and all class text descriptions; predict the class whose text embedding is most similar to the image embedding. Zero-shot: no examples of the target classes are needed during inference.

2. **Q: What is a diffusion model and how does it generate images?**
   A: Diffusion models learn to reverse a noise-adding process. Forward: add Gaussian noise to an image over T steps until it's pure noise. Reverse (learned): remove noise step by step to generate a coherent image from pure noise. Conditioned with text embeddings (Stable Diffusion) to guide the denoising towards the desired content.

3. **Q: What is classifier-free guidance (CFG) in Stable Diffusion?**
   A: CFG scales the influence of the text conditioning. At each denoising step: output = unconditioned + scale × (conditioned - unconditioned). Scale=7.5 means 7.5× amplification of the text signal. Higher scale: more faithful to prompt but less diverse. Lower scale: more creative but may ignore the prompt.

4. **Q: What is vLLM and why does it improve throughput?**
   A: vLLM uses PagedAttention: manages KV cache memory like OS virtual memory (in non-contiguous "pages"). This eliminates memory fragmentation that wastes GPU memory. Combined with continuous batching (starting new requests as others finish), vLLM achieves 3-10× higher throughput than HuggingFace.

5. **Q: What is quantisation and what are the tradeoffs?**
   A: Quantisation reduces the precision of weights/activations: FP32 (32-bit) → FP16/BF16 (16-bit) → INT8 → INT4. Less precision = less VRAM + faster computation + possible quality degradation. INT4 uses 4× less VRAM than FP16 with typically <1% quality loss for GPTQ/AWQ on 7B+ models.

6. **Q: What is FlashAttention and why is it important?**
   A: FlashAttention computes attention in tiles that fit in the GPU's SRAM (fast memory) rather than HBM (slow memory). Never materialises the full N×N attention matrix. Result: memory = O(N) instead of O(N²), and much faster due to reduced memory bandwidth. Essential for long-context models.

7. **Q: What is speculative decoding?**
   A: A small "draft" model generates k token candidates quickly. The large "target" model verifies all k tokens in parallel (one forward pass). Accepted tokens are kept; rejected tokens are regenerated. Speedup: the verification pass is much faster than k sequential target model passes.

**Intermediate**

8. **Q: How does LLaVA connect a vision encoder to an LLM?**
   A: LLaVA uses CLIP's vision encoder to produce image patch embeddings, then projects them into the LLM's embedding space using a simple linear projection (or MLP). The projected image tokens are concatenated with the text token embeddings and fed into the LLM's transformer layers. This enables the LLM to "see".

9. **Q: Explain the roofline model for GPU performance analysis.**
   A: The roofline model plots achievable FLOPS vs. arithmetic intensity (FLOPS/byte). The roofline has two regimes: (1) memory-bound: limited by bandwidth, performance proportional to arithmetic intensity; (2) compute-bound: limited by peak FLOPS, performance flat. Operations below the roof are inefficient. FlashAttention moves attention from memory-bound to the optimal ridge point.

10. **Q: What is the difference between GPTQ and AWQ for LLM quantisation?**
    A: GPTQ: post-training quantisation that minimises quantisation error layer by layer using Hessian information. Requires calibration data (~100 examples). AWQ: Activation-aware Weight Quantisation; identifies and protects the most important weights (those that activate heavily). AWQ typically slightly outperforms GPTQ and is faster to run.

11. **Q: How does speculative decoding maintain output quality?**
    A: The acceptance criterion: draft token x_i is accepted if `random() < min(1, p_target(x_i)/p_draft(x_i))`. This is based on rejection sampling theory — the distribution of accepted tokens matches the target model's distribution exactly. No approximation, no quality loss. The draft model is used only for speed; quality is fully determined by the target model.

12. **Q: What is PagedAttention in vLLM?**
    A: KV cache memory is divided into non-contiguous fixed-size "blocks" (like OS virtual memory pages). Logical KV blocks are mapped to physical blocks on-demand. Benefits: (1) no pre-allocation of maximum sequence length; (2) memory sharing for parallel sampling; (3) eliminates fragmentation. Enables significantly higher batch sizes.

13. **Q: How does Whisper handle different languages and tasks?**
    A: Whisper is trained on 680k hours of audio with special tokens encoding language and task. Input: log-mel spectrogram. The decoder receives: [LANGUAGE_TOKEN, TASK_TOKEN] which conditions generation. Tasks: transcription, translation, language detection, timestamp prediction — all handled by the same model with different conditioning tokens.

**Advanced**

14. **Q: Why does latent diffusion (Stable Diffusion) work in latent space?**
    A: Pixel-space diffusion is computationally expensive (512×512×3 = 786k dimensions). A VAE compresses images to a smaller latent space (64×64×4 = 16k dimensions — 49× smaller). Running diffusion in this latent space is ~50× cheaper. The VAE decoder converts latents back to pixels only at the end.

15. **Q: How does `torch.compile` improve performance?**
    A: torch.compile uses TorchDynamo to trace the computation graph and TorchInductor to generate optimised kernels. Key optimisations: (1) kernel fusion: multiple ops merged into one GPU kernel (reduces kernel launch overhead); (2) memory layout optimisation; (3) loop unrolling. Typical speedup: 10-50% for training, 20-100% for inference.

16. **Q: What is the accuracy-latency tradeoff in speculative decoding and how do you choose the draft model?**
    A: Acceptance rate depends on how well the draft model approximates the target. Ideal draft: same family as target (small LLaMA as draft for large LLaMA). If acceptance rate is high, you approach k× speedup. If acceptance rate is low, speculative decoding is slower than no speculative decoding (overhead of verification). Typical speedup: 2-3× for text generation.

17. **Q: Explain the mathematics of DDPM forward and reverse processes.**
    A: Forward: q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I). Can sample x_t from x_0 directly: q(x_t|x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I) where ᾱ_t = Π β_{s≤t}. Reverse: p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ). The model learns to predict ε (the noise) or x_0 directly.

18. **Q: What is continuous batching and how does it differ from static batching?**
    A: Static batching: wait until batch is full, process together, release all together. Wastes time when some requests finish early — GPU sits idle. Continuous batching: as requests finish, immediately add new requests to the batch. Keeps GPU utilisation near 100%. vLLM, TGI use continuous batching.

19. **Q: How do you profile GPU memory usage during LLM inference?**
    A: (1) `torch.cuda.memory_allocated()` / `torch.cuda.max_memory_allocated()` for peak usage. (2) PyTorch memory snapshot: `torch.cuda.memory._record_memory_history()` → visualise allocations. (3) `nvidia-smi` for total VRAM. (4) vLLM logs detailed memory allocation. (5) Profile with `nsys profile` for kernel-level analysis.

20. **Q: What is the difference between FP16 and BF16, and which is preferred for LLM training?**
    A: FP16: 5 exponent bits, 10 mantissa bits (range: ~65k, precision: 10 bits). BF16: 8 exponent bits, 7 mantissa bits (range: same as FP32, precision: 7 bits). BF16 is preferred for LLM training: its larger dynamic range prevents overflow on gradient accumulation. FP16 can overflow with large learning rates. Modern A100/H100 GPUs support BF16 natively.

---

### 7. Self-Assessment Quiz

- [ ] What is a Vision Transformer (ViT)?
- [ ] How does CLIP training work?
- [ ] What is CFG scale in Stable Diffusion?
- [ ] What is the difference between diffusion and autoregressive generation?
- [ ] What is FlashAttention and what memory complexity does it achieve?
- [ ] What is PagedAttention?
- [ ] What is torch.compile and what does it optimise?
- [ ] What is the difference between GPTQ and AWQ?
- [ ] How does speculative decoding guarantee output quality?
- [ ] What is the roofline model?
- [ ] What is memory bandwidth and why does it matter for LLM inference?
- [ ] What is the difference between memory-bound and compute-bound operations?
- [ ] What is the acceptance criterion in speculative decoding?
- [ ] What is the forward process in diffusion?
- [ ] What is a U-Net and why is it used in diffusion models?
- [ ] What is continuous batching and why does it improve GPU utilisation?
- [ ] What is latent diffusion and what is the role of the VAE?
- [ ] How does LLaVA connect the vision encoder to the LLM?
- [ ] What is LLM throughput vs. latency and when does each matter?
- [ ] What is a spectrogram input for Whisper?
- [ ] What is the difference between BF16 and FP16?
- [ ] What is INT4 quantisation and what quality loss is typical?
- [ ] What is continuous batching in vLLM?
- [ ] Name 3 techniques to reduce inference latency.
- [ ] What is GGUF format and what is it used for?

**Scoring**: 22–25 ✅ = Ready. 17–21 = Review weak areas. Below 17 = Spend more time on Phase 8.

---

### 8. Common Mistakes

| Mistake | Why It Happens | How to Avoid |
|---------|---------------|--------------|
| Using high CFG without increasing steps | High CFG looks great at first | High CFG amplifies artifacts; need more denoising steps to correct |
| Not verifying speculative decoding output matches target output | "Looks about right" | Assert: output of speculative should be identical to target; test with same seed |
| Benchmarking with batch=1 only | Simple to implement | Always benchmark at multiple batch sizes; vLLM advantage only shows at batch>1 |
| Loading quantised model on CPU before moving to GPU | Logical but wasteful | Use `device_map="cuda:0"` directly; never load to CPU then move |
| Not measuring quality after quantisation | Speed improvement is obvious | Always check perplexity and domain-specific quality after quantising |
| Comparing throughput without fixing output length | Variable lengths skew results | Fix max_new_tokens when benchmarking |
| Using CLIP for fine-grained tasks without fine-tuning | CLIP is powerful zero-shot | CLIP may struggle with domain-specific fine-grained distinctions; fine-tune for production |

---

### 9. Readiness Criteria

You are ready for Phase 9 when **all** of the following are true:

- [ ] I deployed LLaMA-3.1 8B with vLLM and benchmarked throughput vs. HuggingFace
- [ ] I quantised a model to INT4 with AWQ and measured quality loss
- [ ] I used CLIP for zero-shot image classification (Challenge A)
- [ ] I profiled a model with PyTorch profiler and identified the bottleneck
- [ ] I scored 22/25 or higher on the Self-Assessment Quiz
- [ ] I can answer at least 16/20 Interview Questions correctly
- [ ] I understand the difference between memory-bound and compute-bound operations

---

### 10. Revision Summary

```
MULTIMODAL
─────────────────────────────────────────────────────
ViT:     image → patches → linear embed → transformer → [CLS] embedding
CLIP:    image + text encoders, trained with contrastive loss (similar=close, dissimilar=far)
LLaVA:  CLIP image encoder → projection → LLM token stream
Stable Diffusion: text → CLIP → condition U-Net → latent space → VAE decode → image

INFERENCE OPTIMISATION
─────────────────────────────────────────────────────
FlashAttention:   O(N) memory; tiles in SRAM; never materialises N×N matrix
Quantisation:     FP32 → BF16 → INT8 → AWQ/GPTQ INT4 (less VRAM, faster)
vLLM:             PagedAttention (no fragmentation) + continuous batching
Speculative:      draft k tokens → verify in parallel → accept/reject
torch.compile:    kernel fusion → reduced overhead → 10-50% speedup

ROOFLINE MODEL
─────────────────────────────────────────────────────
Arithmetic intensity = FLOPS / bytes read
Memory-bound:   intensity < ridge point → limited by bandwidth
Compute-bound:  intensity > ridge point → limited by FLOPS
FlashAttention: moves attention from memory-bound toward ridge point
```

---

### 11. Next Phase Prerequisites

**What Phase 9 (Production AI Engineering) requires from Phase 8:**

| Phase 8 Concept | How Phase 9 Uses It |
|----------------|---------------------|
| vLLM deployment | Production serving is built on vLLM |
| Quantisation | Production systems balance quality vs. cost — quantisation is key |
| Inference profiling | Production monitoring includes GPU utilisation and latency |
| Multimodal (LLaVA) | Production multimodal systems built on these models |
| Throughput vs. latency | SLA definitions require understanding these tradeoffs |
| Benchmarking methodology | Production evaluation uses the same methodology |

**The critical dependency**: Phase 9 production engineering requires you to make concrete technical decisions: which model to serve, what precision, what serving framework, what hardware. Phase 8 gives you the knowledge to make these decisions based on measured data rather than guesswork.

---

*Phase 8 | Part of the [GenAI Engineer Roadmap](./00_README.md)*
