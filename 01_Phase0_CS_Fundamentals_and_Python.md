# Phase 0: CS Fundamentals & Python for AI
**Months 1–2 | Difficulty: 3/10 | Label: 🔴 Must Learn**

> **Previous Phase**: None — this is the starting point.  
> **Next Phase**: [Phase 1 — Mathematics for ML](./02_Phase1_Mathematics_for_ML.md)

---

## Table of Contents

- [Phase Overview](#phase-overview)
- [Topic 1: Targeted CS Fundamentals](#topic-1-targeted-cs-fundamentals)
  - [Why This Exists](#why-cs-fundamentals-exist)
  - [What You Actually Need](#what-you-actually-need)
  - [Linux & WSL2 Setup](#linux--wsl2-setup-windows)
  - [Docker Basics](#docker-basics)
  - [Resources](#cs-fundamentals-resources)
  - [Mastery Checklist](#cs-fundamentals-mastery-checklist)
- [Topic 2: Python for AI](#topic-2-python-for-ai)
  - [Why This Exists](#why-python-for-ai-exists)
  - [NumPy Deep Dive](#numpy-deep-dive)
  - [Advanced Python Patterns](#advanced-python-patterns)
  - [Async Python](#async-python)
  - [Resources](#python-resources)
  - [Practice Exercises](#practice-exercises)
  - [Mastery Checklist](#python-mastery-checklist)
- [Phase 0 Projects](#phase-0-projects)
- [Common Mistakes](#common-mistakes)
- [Week-by-Week Plan](#week-by-week-plan)
- [Interview Importance](#interview-importance)

---

## Phase Overview

| Attribute | Details |
|-----------|---------|
| **Duration** | Months 1–2 (8 weeks) |
| **Daily Time** | 1–2 hours |
| **Difficulty** | 3/10 |
| **Label** | 🔴 Must Learn |
| **Prerequisites** | Basic Python, basic programming experience |
| **Outcome** | Environment ready; NumPy fluent; Python at 8/10; async patterns understood |

### What This Phase Produces

By the end of Phase 0, you will:
- Have a professional AI development environment on Windows (WSL2 + conda/uv + Docker)
- Write NumPy-vectorized code that matches what you see in research papers
- Understand Python patterns used throughout Hugging Face, PyTorch, and LangChain codebases
- Be able to write async Python for concurrent API calls (critical for agents)

---

## Topic 1: Targeted CS Fundamentals

### Why CS Fundamentals Exist

AI code runs on top of computer science infrastructure. When you debug a PyTorch OOM error, you need to understand memory. When you optimize a RAG pipeline, you need to understand data structure tradeoffs. When you read open-source AI repos, you need Git and Linux fluency.

**Important**: You already have a strong CS background from data engineering. This phase fills *specific gaps* relevant to AI, not a complete CS curriculum review.

### What You Actually Need

| Topic | Relevance to AI | Your Current Level | Action |
|-------|:-:|:-:|--------|
| Big-O complexity | Understanding batch processing | Likely good | Quick review only |
| Hash maps, queues | LRU caches in inference, KV cache | Likely good | Quick review only |
| Recursion | Tree-of-thought agents, recursive chunking | Moderate | Practice |
| Git (advanced) | Open source contributions, CI/CD | Need to improve | Study |
| Linux/WSL2 | CUDA, AI tooling runs on Linux | Windows-based | Must set up |
| Virtual environments | Dependency isolation for AI projects | Know basics | Upgrade to `uv` |
| Docker | Containerizing AI apps for production | Know basics | AI-specific patterns |

---

### Linux & WSL2 Setup (Windows)

This is a critical setup step. Most AI tooling — CUDA, llama.cpp, vLLM, many datasets — works significantly better or exclusively on Linux. WSL2 gives you a real Linux kernel on Windows.

```powershell
# Step 1: Enable WSL2 in PowerShell (as Administrator)
wsl --install

# Step 2: Install Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# Step 3: Verify installation
wsl --list --verbose
```

Inside WSL2 (Ubuntu), set up your AI development environment:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essentials
sudo apt install -y git curl wget build-essential python3-pip python3-venv

# Install Miniconda (recommended for AI/ML work)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Or install uv (faster alternative, increasingly popular in AI community)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify Python
python3 --version  # Should be 3.10 or 3.11
```

Create a standard project structure:

```bash
# Standard AI project template
mkdir my-ai-project
cd my-ai-project

# Using uv (recommended — installs in seconds, not minutes)
uv init
uv add torch numpy pandas matplotlib scikit-learn jupyter

# Or using conda
conda create -n ai-env python=3.11
conda activate ai-env
pip install torch numpy pandas matplotlib scikit-learn jupyter
```

**Essential `.env` pattern for AI projects:**

```bash
# .env file (never commit this to git)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HUGGINGFACE_TOKEN=hf_...
PINECONE_API_KEY=...

# In Python:
# from dotenv import load_dotenv
# import os
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
```

---

### Docker Basics

Docker is non-negotiable for production AI. Every model serving setup uses containers.

```dockerfile
# Example: Dockerfile for a Python AI application
FROM python:3.11-slim

WORKDIR /app

# Copy dependency file first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t my-ai-app .
docker run -p 8000:8000 --env-file .env my-ai-app

# Docker Compose for multi-service AI apps (e.g., app + vector database)
docker compose up -d
```

---

### CS Fundamentals Resources

| Rank | Resource | Type | Cost | Why |
|------|----------|------|------|-----|
| 1 | [Missing Semester of CS — MIT](https://missing.csail.mit.edu/) | Course | Free | Shell, git, vim, debugging, profiling — tools pros use daily. 12 lectures, each ~1 hour. |
| 2 | [Pro Git Book](https://git-scm.com/book/en/v2) | Book | Free online | Comprehensive, official. Read chapters 1–5 and chapter 10. |
| 3 | [Docker Get Started](https://docs.docker.com/get-started/) | Docs | Free | Official Docker tutorial series |
| 4 | [WSL2 Documentation](https://learn.microsoft.com/en-us/windows/wsl/) | Docs | Free | Microsoft's official WSL2 guide |

**Best YouTube**: [Fireship's Docker in 100 seconds](https://www.youtube.com/watch?v=Gjnup-PuquQ) for a quick mental model, then the official docs.

---

### CS Fundamentals Mastery Checklist

- [ ] WSL2 installed and configured with Ubuntu 22.04
- [ ] Python 3.11 environment running inside WSL2 with conda or uv
- [ ] Can navigate Linux filesystem, use pipes, write basic bash scripts
- [ ] Git: can branch, merge, rebase, resolve conflicts, write clean commit messages
- [ ] Understands `.env` files and never commits secrets to git
- [ ] Can write a Dockerfile for a Python application
- [ ] Can use `docker compose` for multi-container setups
- [ ] Has completed at least 3 lectures from MIT Missing Semester

---

## Topic 2: Python for AI

### Why Python for AI Exists

Your Python is at 6/10. That means you write Python that **works**. AI requires Python that **performs** and is **idiomatic**. The gap matters because:

1. **NumPy vectorization** is the mental model for all tensor operations in PyTorch. If you think in for-loops, you'll struggle to read AI code.
2. **Generators and lazy evaluation** are used everywhere in data loading pipelines. Misunderstanding these causes OOM errors on large datasets.
3. **`asyncio`** is the foundation of modern AI API clients, agents, and serving frameworks.
4. **Type hints and dataclasses** appear throughout LangChain, Hugging Face, and research codebases.

---

### NumPy Deep Dive

NumPy is the single most important library for understanding AI mathematics. Every tensor operation in PyTorch mirrors NumPy. Master this first.

#### Creating Arrays

```python
import numpy as np

# 1D vector
v = np.array([1.0, 2.0, 3.0])

# 2D matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Shapes — always check these
print(v.shape)   # (3,)
print(A.shape)   # (2, 3)

# Important creation functions
zeros = np.zeros((3, 4))
ones = np.ones((2, 2))
identity = np.eye(4)       # Used in attention mechanisms
random = np.random.randn(3, 3)  # Standard normal — used to initialize weights
```

#### Broadcasting — The Most Important NumPy Concept

Broadcasting is how NumPy (and PyTorch) applies operations between arrays of different shapes. Understanding this is *required* to read transformer code.

```python
# Rule: dimensions are compatible if they are equal or one of them is 1
# NumPy aligns shapes from the RIGHT

# Example 1: Adding a vector to each row of a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape: (2, 3)
b = np.array([10, 20, 30]) # shape:    (3,)

result = A + b              # b is broadcast to (2, 3)
# [[11, 22, 33],
#  [14, 25, 36]]

# Example 2: This is how layer normalization bias works
# shape (batch, seq, d_model) + shape (d_model,) -> broadcast to (batch, seq, d_model)
batch_output = np.random.randn(8, 512, 768)  # 8 samples, 512 tokens, 768 dimensions
bias = np.random.randn(768)
output = batch_output + bias  # bias broadcasts correctly

# Example 3: Outer product via broadcasting (used in attention)
q = np.array([1, 2, 3]).reshape(3, 1)  # column vector
k = np.array([4, 5, 6]).reshape(1, 3)  # row vector
outer = q * k   # shape (3, 3) — this is broadcasting!
```

#### Views vs. Copies — Critical for Performance

```python
a = np.array([1, 2, 3, 4, 5])

# View — shares memory with original
b = a[1:4]   # slice creates a VIEW
b[0] = 100   # modifies a too!
print(a)     # [1, 100, 3, 4, 5]

# Copy — independent
c = a[1:4].copy()
c[0] = 999   # does NOT modify a
print(a)     # unchanged

# Check: use b.base to see if it's a view
print(b.base is a)  # True — b is a view of a

# In AI: PyTorch tensors work the same way
# .view() and .reshape() may return views or copies
# Always be aware of this in training loops to avoid accidental gradient issues
```

#### Einstein Summation — `einsum`

`einsum` is used throughout transformer code for tensor contractions. Learn to read it.

```python
# einsum notation: "ij,jk->ik" means matrix multiply A[i,j] * B[j,k] = C[i,k]

A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# Matrix multiplication
C = np.einsum('ij,jk->ik', A, B)
assert C.shape == (3, 5)

# Batched matrix multiply (used in multi-head attention)
# batch_size=8, seq_len=512, d_model=64
Q = np.random.randn(8, 512, 64)
K = np.random.randn(8, 512, 64)

# Compute attention scores: Q @ K.T for each item in batch
scores = np.einsum('bqd,bkd->bqk', Q, K)  # shape: (8, 512, 512)
# b=batch, q=query_pos, k=key_pos, d=dimension

# Outer product
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5])
outer = np.einsum('i,j->ij', v1, v2)  # shape: (3, 2)
```

#### Vectorization — Replace Every For-Loop

```python
# WRONG — Python for-loop over array elements
def sigmoid_slow(x):
    result = []
    for val in x:
        result.append(1 / (1 + np.exp(-val)))
    return np.array(result)

# RIGHT — vectorized
def sigmoid_fast(x):
    return 1 / (1 + np.exp(-x))

# Performance test
import time
x = np.random.randn(1_000_000)

start = time.time()
sigmoid_slow(x)
print(f"Slow: {time.time() - start:.3f}s")  # ~2-3 seconds

start = time.time()
sigmoid_fast(x)
print(f"Fast: {time.time() - start:.3f}s")  # ~0.01 seconds

# Real-world: ReLU activation
def relu(x):
    return np.maximum(0, x)  # vectorized, no loop needed

# Softmax (temperature-controlled — used in LLM sampling)
def softmax(x, temperature=1.0):
    x = x / temperature
    x = x - np.max(x, axis=-1, keepdims=True)  # numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

---

### Advanced Python Patterns

These patterns appear constantly in AI codebases. You must recognize and write them fluently.

#### Dataclasses — Used in AI Configs

```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    """Configuration for an LLM — pattern used throughout Hugging Face."""
    vocab_size: int = 50257
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    block_size: int = 1024
    dropout: float = 0.1
    bias: bool = True
    
    # Mutable defaults must use field(default_factory=...)
    layer_sizes: List[int] = field(default_factory=lambda: [768, 3072, 768])
    
    def __post_init__(self):
        """Validation — runs automatically after __init__"""
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"

# Usage
config = ModelConfig(n_embd=1024, n_head=16)
print(config)  # Nice repr automatically
```

#### Context Managers — Used in PyTorch Training

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(name: str):
    """Time a block of code — used to benchmark training steps."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name}: {elapsed:.4f}s")

# Usage
with timer("Matrix multiply"):
    result = np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)

# PyTorch uses context managers extensively:
# with torch.no_grad():     # disable gradient tracking during inference
# with torch.autocast():    # mixed precision training
# with model.eval():        # not exactly, but similar pattern
```

#### Generators — Used in Data Loading

```python
from typing import Generator, Iterator
import json

def stream_large_file(filepath: str, chunk_size: int = 1000) -> Generator[list, None, None]:
    """
    Stream large JSONL training data without loading into memory.
    This pattern is used in every production ML data pipeline.
    """
    buffer = []
    with open(filepath, 'r') as f:
        for line in f:  # File iteration is lazy — reads one line at a time
            buffer.append(json.loads(line))
            if len(buffer) >= chunk_size:
                yield buffer
                buffer = []  # Release memory
    if buffer:  # Don't forget the last partial batch
        yield buffer

# Usage — processes 100GB file with constant memory
for batch in stream_large_file("huge_training_data.jsonl"):
    # process batch
    pass

# Generator expression (even more concise)
def tokenize_dataset(texts: list[str], tokenizer) -> Iterator[list[int]]:
    return (tokenizer.encode(text) for text in texts)  # lazy
```

#### Decorators — Used in AI Frameworks

```python
import functools
import time
from typing import Callable, Any

def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Retry decorator — used for LLM API calls that may rate-limit.
    This exact pattern appears in production AI systems.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)  # preserves function metadata
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay * (2 ** attempt))  # exponential backoff
        return wrapper
    return decorator

@retry(max_attempts=3, delay=1.0)
def call_llm_api(prompt: str) -> str:
    """This will automatically retry up to 3 times on failure."""
    # ... API call here
    pass

def cache_embeddings(func: Callable) -> Callable:
    """Simple LRU cache for expensive embedding computations."""
    cache = {}
    @functools.wraps(func)
    def wrapper(text: str) -> list[float]:
        if text not in cache:
            cache[text] = func(text)
        return cache[text]
    return wrapper
```

#### Type Hints — Standard in Modern AI Code

```python
from typing import Optional, Union, Tuple, Dict, List, Any
import numpy as np

# Type hints make AI code readable and catch bugs early
def compute_attention(
    query: np.ndarray,          # shape: (batch, seq, d_k)
    key: np.ndarray,            # shape: (batch, seq, d_k)
    value: np.ndarray,          # shape: (batch, seq, d_v)
    mask: Optional[np.ndarray] = None,  # shape: (batch, seq, seq)
    dropout_prob: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: (output, attention_weights)
    output shape: (batch, seq, d_v)
    attention_weights shape: (batch, seq, seq)
    """
    d_k = query.shape[-1]
    scores = np.einsum('bqd,bkd->bqk', query, key) / np.sqrt(d_k)
    
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    weights = softmax(scores)
    output = np.einsum('bqk,bkd->bqd', weights, value)
    return output, weights
```

---

### Async Python

Modern AI applications are async by default. OpenAI, Anthropic, and every major AI provider offers async clients. LangChain, FastAPI, and agent frameworks are all async.

```python
import asyncio
import aiohttp
from typing import List

# Synchronous — slow, makes API calls one at a time
def embed_texts_sync(texts: List[str]) -> List[List[float]]:
    results = []
    for text in texts:
        result = call_embedding_api(text)  # waits for each
        results.append(result)
    return results

# Async — fast, all API calls run concurrently
async def embed_texts_async(texts: List[str]) -> List[List[float]]:
    async with aiohttp.ClientSession() as session:
        tasks = [call_embedding_api_async(session, text) for text in texts]
        results = await asyncio.gather(*tasks)  # all run concurrently
    return results

# Real-world example: concurrent LLM calls with rate limiting
import asyncio
from asyncio import Semaphore

async def process_documents_with_llm(
    documents: List[str],
    max_concurrent: int = 5  # rate limit: max 5 concurrent calls
) -> List[str]:
    semaphore = Semaphore(max_concurrent)
    
    async def process_one(doc: str) -> str:
        async with semaphore:  # ensures max_concurrent limit
            # simulate LLM API call
            await asyncio.sleep(0.1)
            return f"processed: {doc[:50]}"
    
    tasks = [process_one(doc) for doc in documents]
    return await asyncio.gather(*tasks)

# Run async code
if __name__ == "__main__":
    documents = [f"Document {i}" for i in range(100)]
    results = asyncio.run(process_documents_with_llm(documents))
```

#### Async Generator — Used in Streaming LLM Responses

```python
import asyncio
from typing import AsyncGenerator

async def stream_llm_response(prompt: str) -> AsyncGenerator[str, None]:
    """
    Stream tokens from an LLM as they're generated.
    This is how ChatGPT's streaming interface works.
    """
    # Simulate streaming — in reality, use openai.AsyncOpenAI
    tokens = ["Hello", " world", "!", " How", " can", " I", " help", "?"]
    for token in tokens:
        await asyncio.sleep(0.05)  # simulate network delay
        yield token

async def main():
    print("Streaming response: ", end="", flush=True)
    async for token in stream_llm_response("Hello"):
        print(token, end="", flush=True)
    print()  # newline at end

asyncio.run(main())
```

---

### Python Resources

| Rank | Resource | Type | Cost | Why Better |
|------|----------|------|------|-----------|
| 1 | [NumPy 100 Exercises (GitHub)](https://github.com/rougier/numpy-100) | Exercises | Free | 100 progressive exercises; essential muscle memory |
| 2 | [Arjan Codes YouTube](https://www.youtube.com/@ArjanCodes) | YouTube | Free | Best modern Python patterns; clean code, type hints, design patterns |
| 3 | Fluent Python, 2nd ed. (Luciano Ramalho) | Book | ~$50 | The definitive advanced Python book; covers everything |
| 4 | [Python for Data Analysis (McKinney)](https://wesmckinney.com/book/) | Book | Free online | By Pandas creator; definitive NumPy/Pandas reference |
| 5 | [Real Python](https://realpython.com/) | Tutorials | Free | High-quality, practical tutorials |
| 6 | [Python asyncio documentation](https://docs.python.org/3/library/asyncio.html) | Docs | Free | Official; most authoritative |

**Best YouTube Playlist**: [Arjan Codes — Python Tips](https://www.youtube.com/playlist?list=PLC0nd42SBTaMpVAAHCAifm5gN2zLk2MBo) for modern patterns

---

### Practice Exercises

Work through these in order. Do not skip or look up solutions immediately.

#### Exercise Set 1: NumPy Fundamentals (Week 1)

```python
# Exercise 1: Implement these without using the built-in functions
# then verify your answer against the built-in

import numpy as np

def dot_product_manual(v1: np.ndarray, v2: np.ndarray) -> float:
    """Implement dot product using only element-wise multiply and sum."""
    # Your code here
    pass

def matrix_multiply_manual(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Implement matrix multiply using only for-loops and +, *, sum."""
    # Your code here (understand the algorithm first)
    pass

def normalize_rows(A: np.ndarray) -> np.ndarray:
    """Divide each row by its L2 norm. No loops. Broadcasting required."""
    # Your code here
    pass

# Exercise 2: Broadcasting practice
# Given a 3x4 matrix and a 1x4 row vector, subtract the row vector from each row
# WITHOUT a for-loop

# Exercise 3: Implement softmax that works on any axis
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax. Must work on batches."""
    # Subtract max for numerical stability (why? think about e^(large number))
    pass

# Exercise 4: Implement layer normalization
def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize x to have mean=0 and std=1 along the last axis.
    shape (batch, seq, d_model) -> same shape
    No loops.
    """
    pass
```

#### Exercise Set 2: Advanced Python (Week 2)

```python
# Exercise 1: Build a simple LRU cache from scratch (no functools.lru_cache)
# This teaches you about OrderedDict and is a real interview question

from collections import OrderedDict

class LRUCache:
    """
    Least Recently Used cache — used in KV cache implementations.
    Used to cache expensive embedding computations.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: str) -> any:
        # Return value if exists, else -1
        # Move to end (most recently used) on access
        pass
    
    def put(self, key: str, value: any) -> None:
        # Add/update value
        # Evict least recently used if over capacity
        pass

# Exercise 2: Implement a simple tokenizer using a generator
def tokenize_stream(text_stream, vocab: dict) -> ...:
    """Convert a stream of text to token IDs. Memory efficient."""
    pass

# Exercise 3: Build a retry mechanism with exponential backoff
# (You'll use this for every LLM API call in production)
```

#### Exercise Set 3: NumPy AI Applications (Weeks 3–4)

```python
import numpy as np

# Exercise 1: Implement PCA from scratch
# This is the most important NumPy exercise — uses everything you learned
def pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Principal Component Analysis using eigendecomposition.
    Do NOT use sklearn.decomposition.PCA
    
    Steps:
    1. Center the data (subtract mean)
    2. Compute covariance matrix
    3. Compute eigenvalues and eigenvectors
    4. Sort by eigenvalue (descending)
    5. Project data onto top n_components eigenvectors
    
    Returns: (projected_data, explained_variance_ratio)
    """
    pass

# Exercise 2: Implement cosine similarity
def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between each row of A and each row of B.
    shape A: (n, d), shape B: (m, d)
    Returns: (n, m) similarity matrix
    This is the core operation in semantic search.
    No loops.
    """
    pass

# Exercise 3: Implement simple attention mechanism
def scaled_dot_product_attention(
    Q: np.ndarray,  # (batch, seq, d_k)
    K: np.ndarray,  # (batch, seq, d_k)
    V: np.ndarray,  # (batch, seq, d_v)
    mask: np.ndarray = None  # (batch, seq, seq)
) -> np.ndarray:
    """
    This is the core of transformer attention.
    Implement it using NumPy only.
    Formula: softmax(QK^T / sqrt(d_k)) * V
    """
    pass
```

**Solution validation**:

```python
# After implementing, validate against known values:
# PCA
X = np.random.randn(100, 10)
projected, ratios = pca(X, n_components=3)
assert projected.shape == (100, 3)
assert np.allclose(ratios.sum(), 1.0, atol=0.01)  # ratios sum ~1

# Cosine similarity
A = np.array([[1, 0], [0, 1]])
B = np.array([[1, 0], [0, 1], [1, 1]])
sims = cosine_similarity(A, B)
assert sims.shape == (2, 3)
assert np.isclose(sims[0, 0], 1.0)  # identical vectors
assert np.isclose(sims[0, 1], 0.0)  # orthogonal vectors
```

---

### Python Mastery Checklist

#### NumPy
- [ ] Can explain NumPy broadcasting rules from memory
- [ ] Writes vectorized code — no Python for-loops over array elements
- [ ] Understands the difference between a NumPy view and a copy
- [ ] Can use `einsum` for batched tensor operations
- [ ] Has completed 50+ exercises from the NumPy 100 exercises repo
- [ ] Implemented softmax, sigmoid, relu in vectorized form
- [ ] Implemented PCA from scratch (eigendecomposition, no sklearn)
- [ ] Implemented cosine similarity matrix (vectorized, no loops)

#### Advanced Python
- [ ] Writes dataclasses for configuration objects
- [ ] Uses context managers (`with` statement) correctly
- [ ] Understands generators and can use them for memory-efficient data loading
- [ ] Can write a decorator with arguments (e.g., `@retry(max_attempts=3)`)
- [ ] Uses type hints throughout all code
- [ ] Can read and write async Python (`asyncio`, `await`, `async for`)
- [ ] Understands `functools.partial`, `functools.lru_cache`
- [ ] Uses `collections.defaultdict`, `collections.Counter`, `collections.OrderedDict`

#### Environment
- [ ] WSL2 set up and working on Windows
- [ ] Comfortable in bash: navigation, piping, file permissions
- [ ] Python project template set up (venv/conda, `.env`, `requirements.txt`)
- [ ] Docker: can build and run a Python application in a container
- [ ] Git: can branch, merge, rebase, write clean commits

---

## Phase 0 Projects

### Mini Project 1: Vectorized Data Analysis Pipeline
**Time**: 3–4 hours | **Difficulty**: 2/10

Build a data analysis pipeline using only NumPy (no pandas) that:
1. Loads a CSV file line by line (generator, not all at once)
2. Cleans and normalizes numerical columns (vectorized)
3. Computes correlation matrix
4. Implements and applies PCA for dimensionality reduction
5. Outputs top-N most similar rows to a query row (cosine similarity)

**Skills learned**: Generators, vectorization, linear algebra in code, I/O efficiency

### Mini Project 2: Async API Client
**Time**: 2–3 hours | **Difficulty**: 3/10

Build an async client that:
1. Accepts a list of texts
2. Makes concurrent API calls to any public API (e.g., Wikipedia summary API)
3. Implements retry with exponential backoff
4. Rate-limits to N concurrent requests
5. Caches results using a simple LRU cache

**Skills learned**: `asyncio`, `aiohttp`, decorators, rate limiting (critical for LLM API usage)

### Mini Project 3: Implement Attention from Scratch
**Time**: 4–5 hours | **Difficulty**: 4/10

Implement the full scaled dot-product attention mechanism using only NumPy:
1. Query, Key, Value projections
2. Scaled dot-product scores
3. Causal masking (for autoregressive models)
4. Softmax normalization
5. Weighted sum of values
6. Visualize the attention weights as a heatmap

**Skills learned**: NumPy mastery, attention mechanism intuition, matplotlib visualization

---

## Common Mistakes

| Mistake | Why It Happens | How to Avoid |
|---------|---------------|-------------|
| Skipping NumPy for sklearn immediately | Impatience | NumPy is the foundation; without it, PyTorch is confusing |
| Ignoring broadcasting | Seems complex | Spend 2 hours on it now; saves 20 hours of debugging later |
| Not setting up WSL2 | Seems optional | CUDA, vLLM, llama.cpp require Linux. Set up WSL2 in Week 1 |
| Writing loops over arrays | Old habits | Code review every piece of code: does it have a loop over array elements? |
| Skipping async | "I'll learn it later" | Every AI framework is async. Learn it now or relearn everything later |
| Committing API keys | Carelessness | Immediately set up `.gitignore` with `.env`; use pre-commit hooks |
| Not understanding views vs copies | Hidden bug source | Test this explicitly; it causes silent bugs in training loops |

---

## Week-by-Week Plan

| Week | Daily Focus (1–2 hrs) | Weekend Project |
|------|----------------------|-----------------|
| **Week 1** | WSL2 setup (Day 1), Git (Day 2), Bash (Day 3), NumPy arrays + indexing (Days 4–5) | NumPy 100 exercises (1–30) |
| **Week 2** | NumPy broadcasting (Days 1–2), Views vs copies (Day 3), Vectorization (Days 4–5) | Implement PCA from scratch |
| **Week 3** | einsum (Day 1), Dataclasses + type hints (Days 2–3), Generators (Days 4–5) | Mini Project 1: Vectorized pipeline |
| **Week 4** | Decorators (Days 1–2), Context managers (Day 3), asyncio basics (Days 4–5) | Mini Project 2: Async API client |
| **Week 5** | asyncio advanced (Days 1–2), Docker basics (Days 3–4), Review week | Mini Project 3: Attention from NumPy |
| **Week 6** | Profiling Python code (Day 1), Memory management (Day 2), functools (Days 3–4), Review | NumPy 100 exercises (50–100) |
| **Week 7** | Polish all three projects, document them, push to GitHub | Start Phase 1: watch 3Blue1Brown |
| **Week 8** | Buffer / catch-up week | Review checklist, ensure all items ticked |

---

## Interview Importance

| Topic | Interview Frequency | Note |
|-------|:--:|-----|
| NumPy vectorization | 🔴 High | Asked in take-home ML coding tests |
| Broadcasting | 🟡 Medium | "Why does this code fail?" debugging questions |
| Python generators | 🔴 High | Data pipeline design questions |
| Async Python | 🟡 Medium | System design questions for AI services |
| Docker | 🔴 High | "How would you deploy this model?" |
| Git | 🟡 Medium | Collaboration workflow questions |

---

## Moving to Phase 1

**Before proceeding to [Phase 1: Mathematics for ML](./02_Phase1_Mathematics_for_ML.md), confirm:**

- [ ] Implemented scaled dot-product attention using only NumPy
- [ ] Completed 50+ NumPy exercises
- [ ] Implemented PCA from scratch (not using sklearn)
- [ ] Wrote async Python that makes concurrent API calls
- [ ] All three mini projects pushed to GitHub
- [ ] WSL2 environment stable and comfortable

**Why Phase 1 comes next**: NumPy fluency unlocks mathematical implementation. You'll be implementing linear algebra algorithms (dot products, matrix decompositions) in code throughout Phase 1. Phase 0 ensures that code is second nature.

---

## Phase Completion & Readiness Assessment

> Complete this assessment **before** moving to Phase 1. Do not progress until you meet the Readiness Criteria.

---

### 1. Knowledge Checklist

- [ ] Python data types: list, tuple, dict, set — know when to use each
- [ ] List and dict comprehensions, generator expressions
- [ ] Functions: `*args`, `**kwargs`, default arguments, closures
- [ ] Classes: `__init__`, `__repr__`, `__len__`, `__getitem__`, inheritance, `super()`
- [ ] Decorators: how they work, how to write one from scratch
- [ ] Context managers: `with` statement, `__enter__`/`__exit__`, `contextlib`
- [ ] Generators: `yield`, `yield from`, lazy evaluation vs. eager evaluation
- [ ] Iterators: `__iter__`, `__next__`, difference from iterables
- [ ] `asyncio`: `async def`, `await`, `asyncio.gather`, event loop
- [ ] Type hints: `List[int]`, `Optional[str]`, `Union`, `Callable`, generics
- [ ] `dataclasses` and `Pydantic` for structured data
- [ ] NumPy: array creation, shape, dtype, strides
- [ ] NumPy: broadcasting rules (when shapes are compatible)
- [ ] NumPy: vectorized operations vs. Python loops (speed difference)
- [ ] NumPy: fancy indexing, boolean masking, `np.where`
- [ ] Pandas: `DataFrame` vs `Series`, `groupby`, `merge`, `pivot_table`, `apply`
- [ ] Docker: image, container, `Dockerfile`, `docker-compose.yml`
- [ ] Git: `commit`, `branch`, `merge`, `rebase`, `cherry-pick`, `stash`
- [ ] GitHub Actions: YAML syntax, triggers, jobs, steps

---

### 2. Practical Skills Checklist

- [ ] Write a decorator that times a function — without looking it up
- [ ] Write a generator that yields batches of a list (batch size configurable)
- [ ] Write an async function that calls an API concurrently using `asyncio.gather`
- [ ] Implement softmax using only NumPy (numerically stable version)
- [ ] Implement batch normalization using only NumPy
- [ ] Write a `Dataset` class with `__len__` and `__getitem__` methods
- [ ] Use `np.einsum` to perform matrix multiplication
- [ ] Profile a function using `cProfile` and identify the bottleneck
- [ ] Write a `Dockerfile` that packages a Python script and runs on port 8000
- [ ] Set up a GitHub Actions workflow that runs `pytest` on every push

---

### 3. Coding Challenges

**Challenge A — NumPy Vectorization**
```python
# Implement the following using ONLY NumPy (no Python loops):
# 1. Compute pairwise cosine similarity between all rows of a matrix X (shape: N x D)
# 2. Implement softmax over the last axis (numerically stable)
# 3. Implement one-hot encoding for a 1D array of integer labels
# 4. Compute the running mean and variance of a 1D array (batch normalization input)
```

**Challenge B — Python Internals**
```python
# Without using functools.lru_cache, implement your own @memoize decorator
# that caches function results in a dictionary.
# It must handle: positional args, keyword args, and unhashable args (raise TypeError).
# Test it on a recursive Fibonacci function.
```

**Challenge C — Async API Caller**
```python
# Write an async function that:
# 1. Takes a list of URLs
# 2. Fetches all of them concurrently using aiohttp
# 3. Returns a list of (url, status_code, response_time_ms) tuples
# 4. Has a configurable timeout per request
# 5. Gracefully handles connection errors without crashing
```

---

### 4. Mini Project

**Data Pipeline Package**: Build a Python package called `dataprep` that:
- Has a `Dataset` class that loads CSV or JSON files lazily (using generators)
- Has a `Preprocessor` class that normalizes numeric columns and encodes categoricals
- Supports method chaining: `Dataset("data.csv").normalize().encode().to_numpy()`
- Is fully type-hinted
- Has a `Dockerfile` and runs in a container
- Has a GitHub Actions workflow that runs unit tests

**Deliverable**: Working package in a GitHub repo with README, tests, and Docker setup.

---

### 5. Capstone Project

Combine Projects 1 and 2 from the roadmap into a unified **"AI Dev Toolkit"** repository:
- Environment setup with reproducible `requirements.txt` and `environment.yml`
- NumPy benchmark notebook: loops vs. NumPy vs. PyTorch (all operations)
- Softmax, batch norm, and cross-entropy implemented three ways: pure Python, NumPy, PyTorch
- Performance comparison charts
- Full CI/CD with GitHub Actions

---

### 6. Interview Questions

**Beginner**

1. **Q: What is the difference between a list and a generator in Python?**
   A: A list stores all items in memory at once. A generator produces items lazily on demand using `yield`, consuming O(1) memory regardless of size. Use generators for large datasets that don't fit in memory.

2. **Q: What does `*args` and `**kwargs` do in a function signature?**
   A: `*args` collects positional arguments into a tuple; `**kwargs` collects keyword arguments into a dict. They allow functions to accept variable numbers of arguments.

3. **Q: What is a decorator?**
   A: A decorator is a function that takes a function and returns a modified function. It adds behaviour (logging, timing, caching) without modifying the original function's code.

4. **Q: What is the GIL in Python and why does it matter for ML?**
   A: The Global Interpreter Lock prevents multiple threads from executing Python bytecode simultaneously. For CPU-bound ML workloads this limits parallelism — which is why NumPy/PyTorch bypass the GIL by running C/CUDA code.

5. **Q: What is broadcasting in NumPy?**
   A: Broadcasting is NumPy's mechanism for performing operations on arrays of different shapes by implicitly expanding the smaller array. Example: adding shape (3,) to shape (4,3) broadcasts the row vector across all 4 rows.

6. **Q: What is the difference between `deepcopy` and shallow copy?**
   A: A shallow copy creates a new object but shares references to nested objects. A deep copy creates a fully independent copy of all nested objects. For NumPy arrays, `arr.copy()` is a deep copy; `arr.view()` shares memory.

7. **Q: What does `async`/`await` do?**
   A: `async def` defines a coroutine. `await` suspends the coroutine and yields control to the event loop while waiting for an I/O operation, allowing other coroutines to run. This enables concurrency without threads.

**Intermediate**

8. **Q: Explain how NumPy achieves its speedup over Python loops.**
   A: NumPy operations are executed in compiled C code with contiguous memory layouts and SIMD CPU instructions. Python loops have interpreter overhead and object boxing costs per iteration. NumPy eliminates both.

9. **Q: What is the difference between `np.dot`, `np.matmul`, and `@` operator?**
   A: For 2D arrays they're equivalent. For N-D arrays: `np.dot` sums over the last axis of the first and second-to-last of the second; `np.matmul`/`@` broadcasts over batch dimensions. Prefer `@` for clarity.

10. **Q: How do context managers work internally?**
    A: They implement `__enter__` (setup) and `__exit__` (teardown) methods. `with` calls `__enter__` on entry and `__exit__` on exit (even on exception). `contextlib.contextmanager` wraps a generator function into a context manager.

11. **Q: What is the difference between `__iter__` and `__next__`?**
    A: `__iter__` returns the iterator object (usually `self`). `__next__` returns the next value and raises `StopIteration` when exhausted. An *iterable* has `__iter__`; an *iterator* has both.

12. **Q: How does Docker layering work?**
    A: Each `Dockerfile` instruction creates an immutable layer. Layers are cached and shared across images. Place infrequently-changing instructions (OS packages) before frequently-changing ones (app code) to maximize cache hits.

13. **Q: What is the difference between `git merge` and `git rebase`?**
    A: `merge` creates a merge commit preserving both histories. `rebase` replays commits from one branch onto another, creating linear history. Use rebase for feature branches before merging to keep history clean; never rebase public/shared branches.

**Advanced**

14. **Q: How would you implement a memory-efficient data loader for a 100GB dataset?**
    A: Use a generator that reads and yields batches on demand. For random access, memory-map the file with `np.memmap`. For shuffling, pre-compute a shuffled index array and read in that order. Use `asyncio` or threading for prefetching.

15. **Q: Explain NumPy strides and when they matter.**
    A: Strides define how many bytes to step in memory for each dimension. Transposing an array doesn't copy data — it just changes strides. Operations on non-contiguous arrays (after transpose) are slower; call `.contiguous()` (PyTorch) or `np.ascontiguousarray()` before intensive operations.

16. **Q: What is Python's descriptor protocol?**
    A: Descriptors implement `__get__`, `__set__`, `__delete__`. Properties, classmethods, staticmethods, and PyTorch's `nn.Parameter` all use descriptors. When you access `obj.attr`, Python checks if the class has a descriptor for `attr` before looking in `obj.__dict__`.

17. **Q: How would you design a configuration system for an ML experiment?**
    A: Use Pydantic BaseModel for type-validated config; load from YAML with OmegaConf or Hydra; support environment variable overrides; use `dataclasses` for nested configs; serialize configs as JSON alongside saved models for reproducibility.

18. **Q: What is the event loop in asyncio and how does it schedule tasks?**
    A: The event loop is a single-threaded scheduler. It maintains a queue of callbacks and futures. When `await` suspends a coroutine, the loop runs other ready coroutines. I/O completion (via OS callbacks) reschedules waiting coroutines. `asyncio.gather` runs multiple coroutines "concurrently" on this single thread.

19. **Q: How would you profile and optimize a slow Python data pipeline?**
    A: (1) Profile with `cProfile`/`line_profiler` to find hotspots. (2) Vectorize loops with NumPy. (3) Use `multiprocessing.Pool` for CPU-bound tasks. (4) Use `asyncio` for I/O-bound tasks. (5) Consider Cython or Numba for remaining bottlenecks.

20. **Q: Explain how `functools.partial` and `functools.wraps` work and why they're used.**
    A: `partial` creates a new function with some arguments pre-filled (currying). `wraps` copies metadata (`__name__`, `__doc__`) from the wrapped function to the decorator's wrapper — without it, all decorated functions would show the wrapper's name in tracebacks.

---

### 7. Self-Assessment Quiz

Answer each question without looking at notes. Mark ✅ if you can answer confidently.

1. [ ] What is the time complexity of list lookup by index vs. dict lookup by key?
2. [ ] Write the broadcasting rule: can shapes (3, 1) and (1, 4) be broadcast together? What is the output shape?
3. [ ] What does `yield from` do differently from `yield`?
4. [ ] What is the difference between `@staticmethod` and `@classmethod`?
5. [ ] Write the numerically stable softmax formula and explain why stability matters.
6. [ ] What does `np.einsum('ij,jk->ik', A, B)` compute?
7. [ ] How do you prevent a Docker container from running as root?
8. [ ] What is `git stash` and when would you use it?
9. [ ] What is the difference between `==` and `is` in Python?
10. [ ] What does `__slots__` do and when would you use it?
11. [ ] What is a closure and give an example where it's useful in ML code?
12. [ ] What does `np.newaxis` do? Give an example.
13. [ ] What is the output of `np.array([1,2,3]) * np.array([1,2,3]).reshape(3,1)`?
14. [ ] How does Python's garbage collector work? What is reference counting?
15. [ ] What is the difference between `multiprocessing` and `threading` in Python?
16. [ ] What is `__all__` in a Python module?
17. [ ] What is the difference between `COPY` and `ADD` in a Dockerfile?
18. [ ] What does `asyncio.create_task` do vs. `await coroutine()`?
19. [ ] Explain Python's method resolution order (MRO) with `super()`.
20. [ ] What is a `NamedTuple` and when would you prefer it over a `dataclass`?
21. [ ] What does `np.view` vs `np.copy` return, and how can you tell which one you have?
22. [ ] What is the purpose of `__repr__` vs `__str__`?
23. [ ] Write the signature of a function that accepts an arbitrary number of keyword arguments and prints them as `key=value`.
24. [ ] What is `functools.lru_cache` and when is caching a bad idea?
25. [ ] What does `enumerate(iterable, start=1)` do?

**Scoring**: 23–25 ✅ = Ready. 18–22 = Review weak areas. Below 18 = Spend more time on Phase 0.

---

### 8. Common Mistakes

| Mistake | Why It Happens | How to Avoid |
|---------|---------------|--------------|
| Using Python loops for array math | Instinct from non-NumPy background | Default to NumPy for all numeric operations; reach for loops only when vectorization is impossible |
| Broadcasting confusion | Don't read error messages carefully | When you see shape mismatch errors, print shapes before and after every operation |
| Modifying a list while iterating | Easy to do accidentally | Use list comprehensions or iterate over a copy: `for x in list(lst):` |
| Shallow copying mutable defaults | `def f(x=[]):` — this list is shared across calls | Always use `None` as default and create inside the function body |
| Ignoring async event loop blocking | Calling blocking code inside async functions | Never use `time.sleep()` in async code; use `asyncio.sleep()` |
| Over-using Pandas `apply` | Feels intuitive | `apply` is a Python loop under the hood; always try vectorized Pandas operations first |
| Forgetting `np.float64` vs Python `float` | NumPy types have different behaviour in some edge cases | When in doubt, check `type()` and `dtype` |
| Not using `.copy()` after slicing NumPy | NumPy slices are views; mutating them mutates the original | After slicing, call `.copy()` if you need independence |
| Leaving secrets in Docker images | Not thinking about security | Use `ARG` for build-time vars, environment variables for secrets, never `ENV SECRET=abc` in Dockerfile |
| Committing to `main` directly | Feature development habit | Always branch; use `git flow` or trunk-based development with short-lived branches |

---

### 9. Readiness Criteria

You are ready for Phase 1 when **all** of the following are true:

- [ ] I can explain every item in the Knowledge Checklist without notes
- [ ] I completed Coding Challenge A (NumPy vectorization) without looking up syntax
- [ ] I completed Coding Challenge B (memoize decorator) from memory
- [ ] I built the Mini Project (data pipeline package) without copying code
- [ ] I scored 23/25 or higher on the Self-Assessment Quiz
- [ ] I can answer at least 16/20 Interview Questions correctly
- [ ] I have Projects 1 and 2 committed to GitHub with a README

**Stop and revisit if**: You look up basic NumPy syntax more than once per session, or you can't explain broadcasting without looking it up.

---

### 10. Revision Summary

```
PYTHON ESSENTIALS
─────────────────────────────────────────────────────
Generators:    yield keyword, lazy, O(1) memory
Decorators:    @wrapper — function that wraps a function
Context mgr:   with — guarantees __exit__ even on exception  
Async:         async/await — concurrent I/O on single thread

NUMPY CORE
─────────────────────────────────────────────────────
Vectorization: C loops + SIMD → 100-1000x faster than Python
Broadcasting:  align shapes right → expand dims of size 1
Softmax:       exp(x - max(x)) / sum(exp(x - max(x)))
Batch Norm:    (x - mean) / sqrt(var + eps)  then  γx + β
einsum:        'ij,jk->ik' = matmul; 'bii->bi' = batch trace

TOOLCHAIN
─────────────────────────────────────────────────────
Docker:        FROM → RUN → COPY → CMD  (layers are cached)
Git:           commit early, branch always, rebase before merge
CI/CD:         push → GitHub Actions → pytest → merge if green
```

---

### 11. Next Phase Prerequisites

**What Phase 1 (Mathematics for ML) requires from Phase 0:**

| Phase 0 Skill | Used In Phase 1 |
|--------------|-----------------|
| NumPy array operations | Implementing matrix operations, dot products, SVD |
| Broadcasting | Vectorizing gradient computations |
| Python OOP | Building `LinearAlgebra`, `GradientDescent` classes |
| Type hints + dataclasses | Clean implementation of math algorithms |
| Generators | Stochastic gradient descent mini-batch samplers |

**The critical dependency**: Every Phase 1 concept (eigenvalues, SVD, gradients, KL divergence) will be *implemented in NumPy*. If you're still thinking in Python loops, Phase 1 implementations will be 10–100x slower and the code will be unreadable. Phase 0 fluency is the difference between struggling and flowing through Phase 1.

---

*Phase 0 | Part of the [GenAI Engineer Roadmap](./00_README.md)*
