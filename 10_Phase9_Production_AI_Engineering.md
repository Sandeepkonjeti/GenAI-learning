# Phase 9: Production AI Engineering
**Month 17–18 | Difficulty: 8/10 | Label: 🔴 Must Learn**

> **Previous Phase**: [Phase 8 — Advanced Topics](./09_Phase8_Advanced_Topics.md)  
> **Next Phase**: [Phase 10 — Research & Open Source](./11_Phase10_Research_and_OpenSource.md)

---

## Table of Contents

- [Phase Overview](#phase-overview)
- [Topic 1: AI System Design](#topic-1-ai-system-design)
  - [RAG vs. Fine-Tuning Decision Framework](#rag-vs-fine-tuning-decision-framework)
  - [Latency, Quality, and Cost Tradeoffs](#latency-quality-and-cost-tradeoffs)
  - [Caching Strategies](#caching-strategies)
  - [Non-Determinism in Production](#non-determinism-in-production)
- [Topic 2: MLOps for AI Systems](#topic-2-mlops-for-ai-systems)
  - [Experiment Tracking with Weights & Biases](#experiment-tracking-with-weights--biases)
  - [Data Versioning with DVC](#data-versioning-with-dvc)
  - [CI/CD for ML](#cicd-for-ml)
  - [Model Registry and Serving](#model-registry-and-serving)
  - [Drift Monitoring](#drift-monitoring)
- [Topic 3: LLMOps](#topic-3-llmops)
  - [Prompt Versioning and Management](#prompt-versioning-and-management)
  - [LangSmith: Tracing LLM Applications](#langsmith-tracing-llm-applications)
  - [Guardrails: Input/Output Safety](#guardrails-inputoutput-safety)
  - [Cost Tracking and Optimization](#cost-tracking-and-optimization)
  - [LLM-as-Judge Evaluation](#llm-as-judge-evaluation)
- [Topic 4: AI Security (OWASP LLM Top 10)](#topic-4-ai-security)
- [Topic 5: AI Evaluation Frameworks](#topic-5-ai-evaluation-frameworks)
- [Resources](#resources)
- [Projects](#projects)
- [Common Mistakes](#common-mistakes)
- [Mastery Checklist](#mastery-checklist)

---

## Phase Overview

| Attribute | Details |
|-----------|---------|
| **Duration** | Months 17–18 (4 weeks) |
| **Daily Time** | 1–2 hours |
| **Difficulty** | 8/10 |
| **Label** | 🔴 Must Learn |
| **Prerequisites** | All prior phases; basic software engineering (CI/CD, Docker) |
| **Outcome** | Can build, monitor, evaluate, and secure production AI systems |

---

## Topic 1: AI System Design

### RAG vs. Fine-Tuning Decision Framework

```python
# The most common architectural decision in AI engineering

def choose_architecture(requirements: dict) -> str:
    """
    Decision framework for RAG vs. fine-tuning.
    Based on: knowledge needs, behavior needs, latency, cost.
    """
    knowledge_gap = requirements.get("private_knowledge", False)
    behavior_gap = requirements.get("style_or_reasoning_change", False)
    freshness = requirements.get("real_time_knowledge", False)
    scale = requirements.get("high_volume", False)
    latency = requirements.get("low_latency_ms", 5000)
    budget = requirements.get("monthly_llm_budget_usd", 1000)
    
    # Decision tree
    if freshness:
        return "RAG with live data source (knowledge changes frequently)"
    
    if knowledge_gap and not behavior_gap:
        return "RAG (all you need is grounding in private data)"
    
    if behavior_gap and not knowledge_gap:
        if budget < 100:
            return "Fine-tune small model (cost constraint)"
        return "Prompt engineering / DPO (behavior adaptation without full fine-tune)"
    
    if knowledge_gap and behavior_gap:
        return "RAG + Fine-tuning (both knowledge grounding AND behavior change needed)"
    
    if scale and latency < 200:
        return "Fine-tune and self-host for cost efficiency and latency"
    
    return "Prompt engineering with strong base model (simplest solution first)"

# Architecture comparison table
architectures = {
    "Prompt engineering only": {
        "Knowledge accuracy": "Limited to training cutoff",
        "Response style": "Generic (prompt-guided)",
        "Latency": "1-2s",
        "Cost": "$$",
        "Maintenance": "Low",
        "When": "Prototyping, well-defined tasks"
    },
    "RAG": {
        "Knowledge accuracy": "High (grounded in docs)",
        "Response style": "Generic",
        "Latency": "2-5s",
        "Cost": "$$+",
        "Maintenance": "Medium (keep docs fresh)",
        "When": "Enterprise Q&A, support bots"
    },
    "Fine-tuning only": {
        "Knowledge accuracy": "Limited to training data",
        "Response style": "Specialized",
        "Latency": "0.5-2s",
        "Cost": "$$$+upfront, $ inference",
        "Maintenance": "High (retrain for new knowledge)",
        "When": "Domain-specific style, high volume"
    },
    "RAG + Fine-tuning": {
        "Knowledge accuracy": "Highest",
        "Response style": "Specialized",
        "Latency": "2-4s",
        "Cost": "$$$",
        "Maintenance": "High",
        "When": "Production enterprise AI"
    },
}
```

---

### Latency, Quality, and Cost Tradeoffs

```python
# The Iron Triangle of LLM systems: you can optimize 2 of 3
# Fast + Cheap = Low Quality
# Fast + High Quality = Expensive
# High Quality + Cheap = Slow

# Concrete latency targets:
latency_targets = {
    "Chat UI (user-facing)":      "< 500ms time-to-first-token",
    "Document QA (user-facing)":  "< 1s time-to-first-token",
    "Batch processing":           "No latency constraint",
    "Agent reasoning step":       "< 2s per step (< 30s total)",
    "Autocomplete":               "< 100ms",
}

# Latency optimization strategies:
latency_strategies = {
    "Streaming":              "Return tokens as they're generated (perceived latency drops dramatically)",
    "Smaller model":          "7B vs 70B is ~5x faster with some quality loss",
    "Quantization":           "4-bit AWQ ~2x faster than BF16 with minimal quality loss",
    "Caching":                "Cache common queries/embeddings (fastest is no computation)",
    "vLLM":                   "2-4x throughput vs. HuggingFace for concurrent requests",
    "Prompt compression":     "Reduce input token count with LLMLingua or similar",
    "Speculative decoding":   "2-3x faster generation for large models",
}

# Cost estimation
def estimate_monthly_cost(
    n_requests_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    model: str = "gpt-4o-mini"
) -> float:
    """Estimate monthly LLM API cost."""
    prices = {
        "gpt-4o":        {"input": 2.50/1e6, "output": 10.00/1e6},
        "gpt-4o-mini":   {"input": 0.15/1e6, "output": 0.60/1e6},
        "claude-3-haiku":{"input": 0.25/1e6, "output": 1.25/1e6},
    }
    
    p = prices.get(model, prices["gpt-4o-mini"])
    daily_cost = n_requests_per_day * (
        avg_input_tokens * p["input"] +
        avg_output_tokens * p["output"]
    )
    monthly_cost = daily_cost * 30
    
    print(f"Model: {model}")
    print(f"Daily requests: {n_requests_per_day:,}")
    print(f"Avg tokens: {avg_input_tokens} in / {avg_output_tokens} out")
    print(f"Daily cost: ${daily_cost:.2f}")
    print(f"Monthly cost: ${monthly_cost:.2f}")
    return monthly_cost
```

---

### FastAPI Deployment for AI Systems

Every production AI system is a REST API. Here is the complete pattern — streaming endpoint, auth middleware, health check, Docker deployment:

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import asyncio, json, os, time
from slowapi import Limiter
from slowapi.util import get_remote_address

app = FastAPI(title="RAG API", version="1.0.0")
aclient = AsyncOpenAI()
limiter = Limiter(key_func=get_remote_address)

# ── Request/Response models ──
class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    max_tokens: int = 512

class HealthResponse(BaseModel):
    status: str
    latency_ms: float

# ── API key auth middleware ──
VALID_API_KEYS = set(os.getenv("API_KEYS", "").split(","))

def verify_api_key(x_api_key: str = Header(default=None)):
    if not x_api_key or x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# ── Streaming chat endpoint ──
async def generate_stream(query: str, max_tokens: int):
    """Async generator yielding SSE-formatted chunks."""
    start = time.time()
    try:
        stream = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}],
            max_tokens=max_tokens,
            stream=True
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield f"data: {json.dumps({'token': delta.content, 'ttft_ms': round((time.time()-start)*1000, 1)})}\n\n"
        yield f"data: {json.dumps({'done': True, 'total_ms': round((time.time()-start)*1000, 1)})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/v1/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(request: ChatRequest, _: str = Depends(verify_api_key)):
    return StreamingResponse(
        generate_stream(request.query, request.max_tokens),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

# ── Non-streaming endpoint ──
@app.post("/v1/chat")
async def chat(request: ChatRequest, _: str = Depends(verify_api_key)):
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": request.query}],
        max_tokens=request.max_tokens
    )
    return {"answer": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens}

# ── Health check (required for load balancers) ──
@app.get("/health", response_model=HealthResponse)
async def health():
    start = time.time()
    # Test connectivity to dependencies
    try:
        await aclient.models.list()
    except Exception:
        raise HTTPException(status_code=503, detail="LLM API unreachable")
    return {"status": "healthy", "latency_ms": (time.time() - start) * 1000}

# ── Docker Compose for full stack ──
# docker-compose.yml:
# services:
#   api:
#     build: .
#     ports: ["8000:8000"]
#     environment:
#       - OPENAI_API_KEY=${OPENAI_API_KEY}
#       - API_KEYS=${API_KEYS}
#       - REDIS_URL=redis://redis:6379
#       - QDRANT_URL=http://qdrant:6333
#   redis:
#     image: redis:7-alpine
#     ports: ["6379:6379"]
#   qdrant:
#     image: qdrant/qdrant:latest
#     ports: ["6333:6333"]
#     volumes: ["./qdrant_data:/qdrant/storage"]
```

**Key streaming metrics to track**:

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| TTFT (Time to First Token) | Request → first token byte | < 500ms (chat), < 200ms (autocomplete) |
| TPOT (Time Per Output Token) | Inter-token interval | < 30ms |
| E2E Latency | Full response time | Depends on output length |
| Perceived Latency | What the user experiences | TTFT dominates this completely |

---

### Caching Strategies

```python
import hashlib
import json
from functools import lru_cache
import redis

class LLMCache:
    """
    Multi-layer LLM response caching.
    
    Layer 1: In-memory LRU cache (fastest, limited capacity)
    Layer 2: Redis cache (fast, persistent, shared across instances)
    Layer 3: Semantic cache (cache based on meaning, not exact match)
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl  # Cache TTL in seconds
        self._memory_cache = {}
    
    def _cache_key(self, prompt: str, model: str, temperature: float) -> str:
        """Create deterministic cache key."""
        key_data = json.dumps({
            "prompt": prompt,
            "model": model,
            "temperature": temperature
        }, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, temperature: float) -> str | None:
        """Check cache layers in order."""
        key = self._cache_key(prompt, model, temperature)
        
        # L1: memory cache
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # L2: Redis cache
        cached = self.redis.get(key)
        if cached:
            response = cached.decode()
            self._memory_cache[key] = response  # Populate L1
            return response
        
        return None
    
    def set(self, prompt: str, model: str, temperature: float, response: str) -> None:
        """Store in all cache layers."""
        key = self._cache_key(prompt, model, temperature)
        self._memory_cache[key] = response
        self.redis.setex(key, self.ttl, response)
    
    @property
    def stats(self) -> dict:
        return {
            "memory_entries": len(self._memory_cache),
            "redis_keys": self.redis.dbsize()
        }

# Semantic caching: cache when queries are semantically similar
# (useful for FAQ systems where users ask the same question differently)

from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticCache:
    """Cache based on semantic similarity of queries."""
    
    def __init__(self, similarity_threshold: float = 0.92):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = similarity_threshold
        self.cache = []  # List of (embedding, query, response)
    
    def get(self, query: str) -> str | None:
        if not self.cache:
            return None
        
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        
        # Find most similar cached query
        embeddings = np.array([item[0] for item in self.cache])
        similarities = embeddings @ query_emb
        
        best_idx = np.argmax(similarities)
        if similarities[best_idx] >= self.threshold:
            print(f"Semantic cache hit (similarity={similarities[best_idx]:.3f})")
            return self.cache[best_idx][2]  # Return cached response
        
        return None
    
    def set(self, query: str, response: str) -> None:
        embedding = self.model.encode([query], normalize_embeddings=True)[0]
        self.cache.append((embedding, query, response))
```

---

## Topic 2: MLOps for AI Systems

### Experiment Tracking with Weights & Biases

```python
import wandb
import torch
from datetime import datetime

# Initialize W&B run
wandb.init(
    project="llm-finetuning",
    name=f"lora-llama3-databricks-{datetime.now().strftime('%Y%m%d-%H%M')}",
    config={
        "model": "meta-llama/Meta-Llama-3.1-8B",
        "lora_r": 16,
        "lora_alpha": 16,
        "learning_rate": 2e-4,
        "batch_size": 8,
        "epochs": 3,
        "dataset_size": 5000,
        "task": "databricks-qa"
    },
    tags=["fine-tuning", "lora", "databricks"]
)

# Log metrics during training
def log_training_step(step: int, loss: float, lr: float, grad_norm: float):
    wandb.log({
        "train/loss": loss,
        "train/learning_rate": lr,
        "train/gradient_norm": grad_norm,
        "train/step": step
    })

def log_evaluation(eval_metrics: dict):
    wandb.log({
        f"eval/{k}": v for k, v in eval_metrics.items()
    })

# Log model artifact
def save_model_artifact(model_path: str, metadata: dict):
    artifact = wandb.Artifact(
        name="lora-adapters",
        type="model",
        metadata=metadata
    )
    artifact.add_dir(model_path)
    wandb.log_artifact(artifact)

# Log example generations (critical for LLM evaluation)
def log_generations(prompts: list[str], responses: list[str], step: int):
    table = wandb.Table(columns=["step", "prompt", "response"])
    for p, r in zip(prompts, responses):
        table.add_data(step, p, r)
    wandb.log({"generations": table}, step=step)

# Always close the run
wandb.finish()
```

---

### CI/CD for ML

```yaml
# .github/workflows/ml_ci.yml
# CI/CD pipeline for ML models

name: ML Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'training/**'
      - 'evaluation/**'
      - 'data/**'

jobs:
  data_validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Validate training data
        run: |
          python scripts/validate_data.py \
            --data_path data/train.jsonl \
            --min_examples 500 \
            --schema_file schemas/training_schema.json

  model_training:
    needs: data_validation
    runs-on: [self-hosted, gpu]  # GPU runner
    steps:
      - uses: actions/checkout@v3
      
      - name: Fine-tune model
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          python training/finetune.py \
            --config configs/lora_config.yaml \
            --output_dir models/lora_adapters_${{ github.sha }}

  evaluation:
    needs: model_training
    runs-on: [self-hosted, gpu]
    steps:
      - name: Evaluate model
        run: |
          python evaluation/evaluate.py \
            --model_path models/lora_adapters_${{ github.sha }} \
            --test_set data/test.jsonl \
            --baseline_model models/lora_adapters_main \
            --min_improvement 0.05  # Fail if not 5% better than baseline

  deploy:
    needs: evaluation
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Register model in MLflow
        run: |
          python scripts/register_model.py \
            --model_path models/lora_adapters_${{ github.sha }} \
            --name "databricks-qa-assistant" \
            --stage "staging"
```

---

### Drift Monitoring

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TextOverviewPreset
import pandas as pd

class LLMDriftMonitor:
    """Monitor for LLM application drift over time."""
    
    def log_interaction(
        self,
        prompt: str,
        response: str,
        metadata: dict,
        db  # Your database connection
    ) -> None:
        """Log each LLM interaction for later drift analysis."""
        db.insert({
            "timestamp": pd.Timestamp.now(),
            "prompt": prompt,
            "response": response,
            "prompt_length": len(prompt.split()),
            "response_length": len(response.split()),
            "model": metadata.get("model"),
            "latency_ms": metadata.get("latency_ms"),
            "cost_usd": metadata.get("cost_usd"),
        })
    
    def compute_drift_report(
        self,
        reference_df: pd.DataFrame,  # Historical baseline
        current_df: pd.DataFrame     # Recent window
    ) -> dict:
        """Detect changes in prompt/response patterns."""
        report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        column_mapping = ColumnMapping(text_features=["prompt", "response"])
        report.run(reference_data=reference_df, current_data=current_df, 
                   column_mapping=column_mapping)
        
        result = report.as_dict()
        
        # Extract drift metrics
        drift_detected = result['metrics'][0]['result']['dataset_drift']
        drift_score = result['metrics'][0]['result']['drift_share']
        
        return {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "n_reference": len(reference_df),
            "n_current": len(current_df),
        }

# What to monitor in production LLM applications:
monitoring_metrics = {
    "Input distribution": "Are prompts still in the expected distribution? New topics?",
    "Response quality": "LLM-as-judge quality score over time",
    "Latency": "p50/p95/p99 latency by model and endpoint",
    "Cost": "Cost per request, daily/weekly totals",
    "Error rate": "API errors, timeout rate, invalid JSON rate",
    "Cache hit rate": "Is your cache helping? (target: >30%)",
    "Hallucination rate": "Use LLM-as-judge to flag potential hallucinations",
    "User feedback": "Thumbs up/down, explicit corrections",
}
```

---

## Topic 3: LLMOps

### Prompt Versioning and Management

```python
# Prompts are code. Version control them.
# Anti-pattern: hardcode prompts in application code
# Best practice: store prompts in a configuration system

import yaml
from pathlib import Path
from datetime import datetime

class PromptRegistry:
    """
    Version-controlled prompt registry.
    Store prompts in YAML, load by name and version.
    """
    
    def __init__(self, prompts_dir: str = "prompts/"):
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(exist_ok=True)
    
    def register(self, name: str, system: str, user_template: str, 
                 version: str, notes: str = "") -> None:
        """Register a new prompt version."""
        prompt_file = self.prompts_dir / f"{name}.yaml"
        
        # Load existing versions
        if prompt_file.exists():
            with open(prompt_file) as f:
                data = yaml.safe_load(f)
        else:
            data = {"name": name, "versions": []}
        
        data["versions"].append({
            "version": version,
            "created_at": datetime.now().isoformat(),
            "notes": notes,
            "system": system,
            "user_template": user_template,
        })
        
        with open(prompt_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def get(self, name: str, version: str = "latest") -> dict:
        """Get a prompt by name and version."""
        prompt_file = self.prompts_dir / f"{name}.yaml"
        
        if not prompt_file.exists():
            raise ValueError(f"Prompt '{name}' not found")
        
        with open(prompt_file) as f:
            data = yaml.safe_load(f)
        
        if version == "latest":
            return data["versions"][-1]
        
        for v in data["versions"]:
            if v["version"] == version:
                return v
        
        raise ValueError(f"Version '{version}' not found for prompt '{name}'")
    
    def render(self, name: str, variables: dict, version: str = "latest") -> dict:
        """Render a prompt template with variables."""
        prompt = self.get(name, version)
        return {
            "system": prompt["system"].format(**variables),
            "user": prompt["user_template"].format(**variables),
        }

# Usage
registry = PromptRegistry()

registry.register(
    name="databricks_qa",
    system="You are a Databricks expert. Answer questions about Spark, Delta Lake, and MLflow.",
    user_template="Context: {context}\n\nQuestion: {question}",
    version="1.0.0",
    notes="Initial version"
)

registry.register(
    name="databricks_qa",
    system="You are a senior Databricks engineer with 10+ years experience. Answer questions concisely with working code examples.",
    user_template="Relevant documentation:\n{context}\n\nQuestion: {question}\n\nInclude a code example if applicable.",
    version="1.1.0",
    notes="Improved system prompt, added code example instruction"
)

rendered = registry.render("databricks_qa", {
    "context": "Delta Lake supports ACID transactions...",
    "question": "How do I run a MERGE operation?"
}, version="latest")
```

---

### LangSmith: Tracing LLM Applications

```python
# LangSmith provides full tracing for LLM applications:
# - Record every LLM call (prompt, response, latency, cost)
# - Trace multi-step agent runs
# - A/B test prompt versions
# - Filter and inspect failed runs

from langsmith import Client, traceable
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "my-rag-application"

# @traceable decorator records every function call
@traceable(name="retrieve_documents")
def retrieve(query: str, collection) -> list[str]:
    results = collection.query(query_texts=[query], n_results=5)
    return results["documents"][0]

@traceable(name="generate_answer")
def generate(query: str, context: list[str], client) -> str:
    context_str = "\n\n".join(context)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using only the provided context."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

@traceable(name="rag_pipeline")
def rag(query: str, collection, llm_client) -> str:
    docs = retrieve(query, collection)
    answer = generate(query, docs, llm_client)
    return answer

# Every call is now logged to LangSmith:
# - Full prompt and response
# - Token counts and cost
# - Latency at each step
# - Parent-child relationship between steps
```

---

### Guardrails: Input/Output Safety

```python
# NeMo Guardrails (NVIDIA): declarative safety rules for LLM apps
# pip install nemoguardrails

from nemoguardrails import RailsConfig, LLMRails

# Define guardrails in Colang (domain-specific language)
colang_content = """
define user ask about internal data
  "what are the internal metrics?"
  "show me employee salaries"
  "what is our confidential strategy?"

define flow
  user ask about internal data
  bot refuse to share confidential information

define bot refuse to share confidential information
  "I can only share publicly available information. For internal data, please use the internal portal."

define user ask for harmful content
  "how do I hack..."
  "write malware..."
  "provide instructions for..."

define flow
  user ask for harmful content
  bot refuse to provide harmful content
"""

# Alternative: simple rule-based guardrails
import re
from typing import Optional

class SimpleGuardrails:
    """
    Lightweight guardrails for LLM input/output validation.
    """
    
    def __init__(self):
        # Input blocklist patterns
        self.input_patterns = [
            (r"ignore (previous|all) instructions", "Potential prompt injection"),
            (r"you are now", "Potential role switching attempt"),
            (r"(system:|<system>)", "Potential system prompt injection"),
        ]
        
        # Output validators
        self.output_validators = []
    
    def validate_input(self, user_input: str) -> tuple[bool, Optional[str]]:
        """Check if input is safe to process."""
        for pattern, reason in self.input_patterns:
            if re.search(pattern, user_input.lower()):
                return False, f"Blocked: {reason}"
        return True, None
    
    def validate_output(self, response: str) -> tuple[bool, Optional[str]]:
        """Check if output is safe to return."""
        # Check for PII patterns
        pii_patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "SSN pattern detected"),
            (r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b", "Credit card pattern"),
        ]
        for pattern, reason in pii_patterns:
            if re.search(pattern, response):
                return False, f"PII in output: {reason}"
        return True, None
    
    def safe_llm_call(self, user_input: str, llm_fn, **kwargs) -> tuple[str, dict]:
        """Safe LLM call with input and output validation."""
        # Validate input
        input_ok, input_reason = self.validate_input(user_input)
        if not input_ok:
            return "I cannot process that request.", {"blocked": True, "reason": input_reason}
        
        # Call LLM
        response = llm_fn(user_input, **kwargs)
        
        # Validate output
        output_ok, output_reason = self.validate_output(response)
        if not output_ok:
            return "I encountered an issue generating a safe response.", {"blocked": True, "reason": output_reason}
        
        return response, {"blocked": False}
```

---

### LLM-as-Judge Evaluation

```python
# LLM-as-Judge: use a strong LLM to evaluate another LLM's outputs
# This is the practical alternative to expensive human evaluation

from openai import OpenAI
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def llm_judge(
    question: str,
    model_response: str,
    ground_truth: str = None,
    criteria: list[str] = None
) -> dict:
    """
    Use GPT-4 to evaluate a model response.
    
    Criteria:
    - correctness (with ground_truth)
    - completeness
    - clarity
    - hallucination_free
    """
    if criteria is None:
        criteria = ["correctness", "completeness", "clarity"]
    
    ground_truth_section = f"\n\nGround Truth: {ground_truth}" if ground_truth else ""
    
    judge_prompt = f"""Evaluate the following AI response objectively.

Question: {question}
AI Response: {model_response}{ground_truth_section}

Evaluate on these criteria (score 1-5 each):
{chr(10).join(f"- {c}" for c in criteria)}

Also detect hallucinations: any claims in the response that are likely incorrect.

Output ONLY valid JSON:
{{
    "scores": {{{", ".join(f'"{c}": 1-5' for c in criteria)}}},
    "overall": 1-5,
    "hallucinations": ["list of potential hallucinations"],
    "reasoning": "brief explanation of scores"
}}"""
    
    response = client.chat.completions.create(
        model="gpt-4o",  # Use best model for judging
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"},
        temperature=0  # Deterministic judging
    )
    
    return json.loads(response.choices[0].message.content)

# Batch evaluation
def evaluate_system(
    test_cases: list[dict],  # [{"question": ..., "response": ..., "expected": ...}]
    model_name: str = "my-rag-system"
) -> dict:
    """Evaluate all test cases and produce a report."""
    results = []
    
    for case in test_cases:
        scores = llm_judge(
            question=case["question"],
            model_response=case["response"],
            ground_truth=case.get("expected")
        )
        results.append({**case, "evaluation": scores})
    
    # Aggregate metrics
    import numpy as np
    
    overall_scores = [r["evaluation"]["overall"] for r in results]
    hallucinations = sum(
        1 for r in results 
        if r["evaluation"].get("hallucinations")
    )
    
    report = {
        "model": model_name,
        "n_examples": len(test_cases),
        "avg_overall_score": np.mean(overall_scores),
        "hallucination_rate": hallucinations / len(test_cases),
        "score_distribution": {
            str(i): overall_scores.count(i) for i in range(1, 6)
        }
    }
    
    return report
```

---

## Topic 4: AI Security

### OWASP LLM Top 10

The [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) defines the top security risks for LLM applications:

| Rank | Vulnerability | Description | Mitigation |
|------|--------------|-------------|------------|
| LLM01 | **Prompt Injection** | Adversarial inputs override system instructions | Input sanitization, output validation, privilege separation |
| LLM02 | **Insecure Output Handling** | LLM output used without validation (XSS, SQL injection, code execution) | Always sanitize LLM output before use; treat as untrusted |
| LLM03 | **Training Data Poisoning** | Malicious data in training set | Data provenance tracking, filtering, anomaly detection |
| LLM04 | **Model Denial of Service** | Crafted inputs causing resource exhaustion | Rate limiting, timeout enforcement, input size limits |
| LLM05 | **Supply Chain Vulnerabilities** | Malicious models/plugins from third parties | Model verification, audit checkpoints from trusted sources |
| LLM06 | **Sensitive Information Disclosure** | Model leaks PII or confidential data | Output scanning, PII detection, access controls |
| LLM07 | **Insecure Plugin Design** | Plugins with excessive permissions | Least privilege, sandboxing, explicit user confirmation |
| LLM08 | **Excessive Agency** | LLM given too much real-world authority | Granular permissions, human-in-the-loop for high-impact actions |
| LLM09 | **Overreliance** | Users trust LLM output without verification | Uncertainty quantification, citation enforcement, human review |
| LLM10 | **Model Theft** | Extraction of model via crafted queries | Rate limiting, output monitoring, watermarking |

```python
# Security implementation examples

# LLM01: Prompt Injection defense
class SecurePromptHandler:
    INJECTION_PATTERNS = [
        "ignore previous instructions",
        "forget everything above",
        "you are now",
        "new system prompt",
        "disregard your training",
    ]
    
    def is_injection(self, user_input: str) -> bool:
        lower = user_input.lower()
        return any(pattern in lower for pattern in self.INJECTION_PATTERNS)
    
    def sanitize(self, user_input: str) -> str:
        """Encode potential injection markers."""
        # Escape characters that might interfere with XML-like tags
        sanitized = user_input.replace("<", "&lt;").replace(">", "&gt;")
        return sanitized

# LLM02: Secure output handling
def safe_code_extraction(llm_response: str) -> str | None:
    """Extract code from LLM response safely - never exec directly."""
    import ast
    
    # Extract code blocks
    import re
    code_blocks = re.findall(r'```(?:python)?\n(.*?)```', llm_response, re.DOTALL)
    
    if not code_blocks:
        return None
    
    code = code_blocks[0]
    
    # AST parse to check for dangerous constructs
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    
    # Check for dangerous operations
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ["exec", "eval", "compile", "__import__"]:
                    return None  # Reject dangerous code
    
    return code

# LLM08: Excessive agency - require confirmation for high-impact actions
class SafeAgent:
    HIGH_IMPACT_ACTIONS = ["delete", "drop_table", "send_email", "push_to_production"]
    
    def execute_action(self, action: str, params: dict, auto_approve: bool = False) -> dict:
        if action in self.HIGH_IMPACT_ACTIONS and not auto_approve:
            print(f"HIGH IMPACT ACTION REQUESTED: {action}")
            print(f"Parameters: {params}")
            confirmation = input("Confirm? (yes/no): ")
            if confirmation.lower() != "yes":
                return {"status": "cancelled", "reason": "User did not confirm"}
        
        # Execute action
        return {"status": "success"}
```

---

## Topic 5: AI Evaluation Frameworks

```python
# Standard evaluation frameworks

# HELM (Holistic Evaluation of Language Models):
# Standardized evaluation across 7 scenarios × N models
# Metrics: accuracy, calibration, robustness, fairness, efficiency

# HumanEval (for code):
from datasets import load_dataset

humaneval = load_dataset("openai_humaneval")
# 164 programming problems with test cases
# Metric: pass@k (does the model produce code that passes tests in k attempts)

def evaluate_code_generation(model_fn, n_samples: int = 100) -> dict:
    """Evaluate code generation with pass@k metric."""
    dataset = load_dataset("openai_humaneval")["test"]
    
    results = []
    for example in dataset.select(range(n_samples)):
        prompt = example["prompt"]
        test_code = example["test"]
        
        # Generate completion
        completion = model_fn(prompt)
        full_code = prompt + completion + "\n" + test_code
        
        # Test the completion
        try:
            exec(full_code, {})
            passed = True
        except Exception:
            passed = False
        
        results.append(passed)
    
    pass_rate = sum(results) / len(results)
    return {"pass@1": pass_rate, "n_evaluated": len(results)}

# RAGAS (covered in Phase 5b):
# faithfulness, answer_relevancy, context_precision, context_recall

# MT-Bench (conversational AI benchmark):
# 80 challenging multi-turn questions across 8 categories
# GPT-4 is the judge

# Evals framework (OpenAI):
# pip install evals
# yaml-based evaluation definitions
# Run: oaieval gpt-4 your_eval_name
```

### Standard LLM Benchmarks — The Complete Map

Understanding industry benchmarks is required knowledge for any LLM engineering role.

| Benchmark | What It Tests | Score Range | Practical Use |
|-----------|--------------|:-----------:|---------------|
| **MMLU** | 57-subject academic knowledge (math, law, medicine…) | 0–100% | Overall knowledge breadth; GPT-4: ~87%, GPT-4o-mini: ~82% |
| **HumanEval** | Code generation (164 Python problems, pass@1) | 0–100% | Code model selection; GPT-4: ~87%, Claude 3.5: ~92% |
| **MBPP** | 374 Python programming problems | 0–100% | Simpler code benchmark; complements HumanEval |
| **BIG-bench** | 200+ diverse tasks, tests emergent abilities | Task-specific | Research benchmark; requires large models |
| **HELM** | Holistic: accuracy + calibration + fairness + efficiency | Multi-metric | Rigorous model comparison |
| **MT-Bench** | 80 multi-turn conversations, GPT-4 as judge | 1–10 | Conversational quality for chat models |
| **LMSYS Chatbot Arena** | Human preference pairwise comparisons (ELO) | ELO score | Real-world user preference; crowd-sourced |
| **GSM8K** | 8500 grade-school math word problems | 0–100% | Mathematical reasoning; critical for agents |
| **MATH** | Competition math problems | 0–100% | Advanced math; harder than GSM8K |
| **HellaSwag** | Commonsense reasoning (sentence completion) | 0–100% | Basic reasoning; now saturated by large models |
| **TruthfulQA** | 817 questions designed to elicit false beliefs | 0–100% | Factuality and calibration (humans ~94%) |
| **RAGAS** | RAG pipeline quality (faithfulness, relevancy…) | 0–1 | Your RAG system evaluation (covered in Phase 5b) |

```python
# Running standard benchmarks with lm-evaluation-harness
# pip install lm-eval

# Command line (evaluates any HuggingFace model):
# lm_eval --model hf \
#   --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
#   --tasks mmlu,gsm8k,humaneval \
#   --device cuda:0 \
#   --batch_size 8 \
#   --output_path results/

# Via Python:
import lm_eval

results = lm_eval.simple_evaluate(
    model="openai-chat-completions",
    model_args="model=gpt-4o-mini",
    tasks=["mmlu", "gsm8k"],
    num_fewshot=5,
)
print(lm_eval.utils.make_table(results))
```

**How to interpret benchmark results**:

1. **Contamination**: if the model was trained on benchmark data, scores are inflated. Check training data transparency.
2. **Prompt sensitivity**: MMLU can swing 5-10% with different prompt formats. Always compare with the same format.
3. **Few-shot vs. zero-shot**: always note how many examples were in the prompt (5-shot MMLU ≠ 0-shot MMLU).
4. **LMSYS Chatbot Arena is the most reliable**: it's crowd-sourced human preferences on real conversations — much harder to game.
5. **Task-specific matters most**: if you're building a code generation system, HumanEval + MBPP matter far more than MMLU.

```python
# Building a custom eval set for your use case
# (more valuable than generic benchmarks for your specific domain)

custom_eval_cases = [
    {
        "id": "databricks_01",
        "question": "How do you read a Delta table in PySpark?",
        "reference_answer": "spark.read.format('delta').load('path') or spark.table('name')",
        "tags": ["databricks", "pyspark", "easy"]
    },
    # ... 50-200 domain-specific cases
]

# Score with LLM-as-judge (your custom domain judge)
def domain_judge(question: str, model_answer: str, reference: str) -> float:
    response = judge_llm.chat.completions.create(
        model="gpt-4o",   # Use the strongest available model as judge
        messages=[{"role": "user", "content": f"""
Rate how well this answer addresses the question on a scale of 1-5.

Question: {question}
Reference answer: {reference}
Model answer: {model_answer}

Criteria:
1 = Completely wrong or missing
3 = Partially correct
5 = Complete and accurate

Output ONLY a single integer 1-5."""}]
    ).choices[0].message.content
    return int(response.strip()) / 5.0
```

---

## Topic 6: Databricks AI Engineering

> This section is your professional leverage point. Most AI engineers don't know Databricks. You do. Here is how to combine both into systems that nobody else on the team can build.

### The Lakehouse AI Architecture

```
Raw Data (Delta Lake)
        ↓ [PySpark ETL]
Feature Table (Delta + Unity Catalog)
        ↓ [Feature Engineering]
Training Dataset (Delta ML tables)
        ↓ [GPU Cluster fine-tuning]
MLflow Tracking (experiments, params, metrics, artifacts)
        ↓ [Model Registry promotion]
Unity Catalog Model (databricks.schema.model_name)
        ↓ [Serving endpoint provisioning]
Databricks Model Serving (REST endpoint, auto-scaling)
        ↓
Databricks AI Gateway (rate limiting, cost tracking, fallbacks)
        +
Mosaic AI Vector Search (real-time ANN index on Delta)
```

### Databricks Model Serving

```python
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput

# ── Step 1: Define a custom pyfunc model (wraps your RAG chain) ──
class RAGModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import yaml
        self.config = yaml.safe_load(open(context.artifacts["config"]))
        # Initialize your vector store, LLM client, etc.
    
    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        queries = model_input["query"].tolist()
        results = []
        for q in queries:
            answer, sources = self._rag_query(q)
            results.append({"answer": answer, "sources": str(sources)})
        return pd.DataFrame(results)
    
    def _rag_query(self, query: str):
        # Your RAG logic here
        return "answer", ["source1", "source2"]

# ── Step 2: Log to MLflow ──
with mlflow.start_run(run_name="rag-v3-hybrid"):
    mlflow.log_params({
        "embedding_model": "databricks-gte-large-en",
        "chunk_size": 512,
        "retrieval_k": 5,
        "llm": "databricks-dbrx-instruct",
    })
    
    # Log the model
    model_info = mlflow.pyfunc.log_model(
        artifact_path="rag_chain",
        python_model=RAGModel(),
        artifacts={"config": "config.yaml"},
        signature=infer_signature(
            pd.DataFrame({"query": ["What is the return policy?"]}),
            pd.DataFrame({"answer": ["30 days."], "sources": ["[\"doc1\"]"]})
        ),
        registered_model_name="main.ai_team.rag_production",  # Unity Catalog format
        pip_requirements=["openai>=1.0", "qdrant-client>=1.7"]
    )

# ── Step 3: Register and promote in Unity Catalog ──
client = WorkspaceClient()

# Set the "production" alias to version 3
client.registered_models.set_alias(
    full_name="main.ai_team.rag_production",
    alias="production",
    version_num=3
)

# ── Step 4: Deploy as a serving endpoint ──
client.serving_endpoints.create(
    name="rag-production",
    config=EndpointCoreConfigInput(
        served_entities=[ServedEntityInput(
            entity_name="main.ai_team.rag_production",
            entity_version="3",
            workload_size="Medium",     # Small / Medium / Large
            scale_to_zero_enabled=True  # 0 cost when idle
        )]
    )
)

# ── Step 5: Call the endpoint ──
import requests, os

response = requests.post(
    "https://<workspace>.azuredatabricks.net/serving-endpoints/rag-production/invocations",
    headers={"Authorization": f"Bearer {os.getenv('DATABRICKS_TOKEN')}"},
    json={"dataframe_records": [{"query": "What is the refund policy?"}]}
)
print(response.json())
```

### Mosaic AI Vector Search

```python
from databricks.vector_search.client import VectorSearchClient

vs_client = VectorSearchClient()

# ── Create a self-managed index from a Delta table ──
index = vs_client.create_delta_sync_index(
    endpoint_name="vs-endpoint",                           # pre-created VS endpoint
    source_table_name="main.documents.chunks",             # Delta table with text
    index_name="main.documents.chunks_index",
    primary_key="chunk_id",
    embedding_source_column="content",                     # column to auto-embed
    embedding_model_endpoint_name="databricks-gte-large-en"  # Databricks-hosted model
)

# ── Or bring your own embeddings ──
index_precomputed = vs_client.create_delta_sync_index(
    endpoint_name="vs-endpoint",
    source_table_name="main.documents.chunks_with_embeddings",
    index_name="main.documents.precomputed_index",
    primary_key="chunk_id",
    embedding_vector_column="embedding",    # pre-computed embedding column
    embedding_dimension=1024
)

# ── Query the index ──
results = index.similarity_search(
    query_text="What are the hardware requirements?",
    columns=["chunk_id", "content", "document_name", "page_number"],
    num_results=5,
    filters={"document_name": ("!=", "outdated_manual_2020.pdf")}  # metadata filtering
)
for r in results["result"]["data_array"]:
    print(f"Score: {r[-1]:.3f} | {r[1][:100]}")
```

### MLflow LLM Experiment Tracking

```python
import mlflow

mlflow.set_experiment("/Shared/rag-experiments")

with mlflow.start_run(run_name="rag-v4-contextual-compression"):
    # Log all hyperparameters
    mlflow.log_params({
        "chunk_strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "embedding_model": "databricks-gte-large-en",
        "retrieval_k": 5,
        "hybrid_alpha": 0.7,        # 0=sparse only, 1=dense only
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "llm": "databricks-dbrx-instruct",
        "context_compression": True,
        "max_context_tokens": 2000
    })
    
    # Log evaluation results
    mlflow.log_metrics({
        "ragas_faithfulness": 0.89,
        "ragas_answer_relevancy": 0.93,
        "ragas_context_precision": 0.87,
        "ragas_context_recall": 0.82,
        "p50_latency_ms": 312,
        "p95_latency_ms": 687,
        "cost_per_query_usd": 0.0042
    })
    
    # Log example generations as a table (viewable in MLflow UI)
    eval_df = pd.DataFrame({
        "query": queries,
        "answer": answers,
        "ground_truth": ground_truths,
        "faithfulness": faithfulness_scores
    })
    mlflow.log_table(data=eval_df, artifact_file="eval_results.json")
    
    # Compare experiments in the UI:
    # mlflow ui --host 0.0.0.0 --port 5000
    # or use Databricks ML Experiments page directly
```

### Databricks AI Gateway

```python
# Databricks AI Gateway: unified API in front of multiple LLMs
# Benefits: cost tracking, rate limiting, model fallbacks, usage quotas

# Configure via Databricks CLI or SDK:
# databricks ai-gateway routes create \
#   --name "production-gpt4" \
#   --provider openai \
#   --model gpt-4o \
#   --rate-limit-calls 100 \
#   --rate-limit-period minute

# Then call via the gateway (same API format):
import openai

openai_client = openai.OpenAI(
    api_key=os.getenv("DATABRICKS_TOKEN"),
    base_url="https://<workspace>.azuredatabricks.net/serving-endpoints"
)

response = openai_client.chat.completions.create(
    model="databricks-dbrx-instruct",    # or your gateway route name
    messages=[{"role": "user", "content": "Explain Delta Lake in 2 sentences."}]
)
# The gateway logs cost, latency, and usage — visible in Databricks workspace
```

### Running Fine-Tuning on Databricks GPU Clusters

```python
# Databricks Mosaic AI Managed Training (pay-per-use fine-tuning)
# Alternatively: bring your own cluster

from databricks.sdk.service.ml import CreateRun
import subprocess

# Method 1: Mosaic AI Model Training API
# Submit directly from the Databricks UI or via REST API
# Supports: LLaMA, Mistral, MPT base models
# Input: Delta table with chat format data

# Method 2: Custom training on GPU cluster
# 1. Attach to a GPU cluster (e.g., g5.4xlarge = 1× A10G)
# 2. Use TorchDistributor for multi-GPU

from pyspark.ml.torch.distributor import TorchDistributor

def train_lora(rank: int, num_proc: int):
    """Fine-tuning function to run on each GPU worker."""
    from transformers import TrainingArguments
    from trl import SFTTrainer
    # ... your fine-tuning code ...

distributor = TorchDistributor(
    num_processes=4,    # 4 GPUs
    local_mode=False,   # distributed across cluster
    use_gpu=True
)
distributor.run(train_lora)
```

---

## Resources

| Rank | Resource | Type | Cost | Why |
|------|----------|------|------|-----|
| 1 | [ML Engineering by Andriy Burkov](http://www.mlebook.com/) | Book | Pay-what-you-want | Best production ML book |
| 2 | [LangSmith Documentation](https://docs.smith.langchain.com/) | Docs | Free | Essential for LLM observability |
| 3 | [OWASP Top 10 for LLMs](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | Docs | Free | Security reference |
| 4 | [Chip Huyen "Designing ML Systems"](https://www.oreilly.com/library/view/designing-machine-learning/9781098107963/) | Book | ~$50 | Best ML systems design book |
| 5 | [ML in Production (full stack deep learning)](https://fullstackdeeplearning.com/course/) | Course | Free | End-to-end production ML |
| 6 | [Evidently AI Documentation](https://docs.evidentlyai.com/) | Docs | Free | Drift monitoring for ML/LLM |

---

## Projects

### Project 22: Production RAG System with Full Observability
**Difficulty**: 9/10 | **Time**: 3 weeks

Build a production-grade RAG system with:
1. **Application**: FastAPI endpoint with streaming response
2. **Caching**: Redis semantic cache
3. **Observability**: LangSmith tracing for every request
4. **Evaluation**: Automated RAGAS scores on 50 test cases (weekly)
5. **Guardrails**: Input sanitization + output PII detection
6. **Monitoring**: Grafana dashboard with latency, cost, quality metrics
7. **CI/CD**: GitHub Actions with automatic evaluation on PR

---

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| No tracing in production | Can't debug failures | Add LangSmith/Langfuse from day 1 |
| Ignoring security | Prompt injection vulnerabilities | Implement OWASP Top 10 mitigations |
| Manual evaluation only | Can't scale quality measurement | Automate LLM-as-judge evaluation |
| No prompt versioning | Can't reproduce past results | Version control prompts like code |
| Monitoring only latency | Miss quality degradation | Monitor latency AND quality |
| No caching | High costs, slow responses | Add semantic cache for repeated queries |

---

## Mastery Checklist

### System Design
- [ ] Can draw the architecture of a production RAG system with all components
- [ ] Can reason about latency/quality/cost tradeoffs for a given scenario
- [ ] Can explain when to use RAG vs. fine-tuning vs. both

### MLOps
- [ ] Set up W&B experiment tracking
- [ ] Built CI/CD pipeline that runs evaluation on each PR
- [ ] Set up drift monitoring on LLM inputs/outputs

### LLMOps
- [ ] Implemented prompt versioning system
- [ ] Set up LangSmith tracing
- [ ] Implemented input/output guardrails
- [ ] Built LLM-as-judge evaluation pipeline

### Security
- [ ] Can explain each of the OWASP LLM Top 10 vulnerabilities
- [ ] Implemented prompt injection protection
- [ ] Implemented safe output handling (never exec LLM output directly)

---

## Moving to Phase 10

**Before proceeding to [Phase 10: Research & Open Source](./11_Phase10_Research_and_OpenSource.md), confirm:**

- [ ] Production RAG system deployed with monitoring
- [ ] Can systematically evaluate and compare AI systems
- [ ] Security mitigations implemented

**Why Phase 10 comes next**: You're now a production AI engineer. Phase 10 teaches you how to stay at the frontier — reading research papers and contributing to the open-source projects that power the field.

---

## Phase Completion & Readiness Assessment

> Complete this assessment before Phase 10. Production AI engineering is what separates hobby projects from things that get you hired and keep systems running.

---

### 1. Knowledge Checklist

**System Design for AI**
- [ ] RAG vs. fine-tuning decision framework (when to use each)
- [ ] AI system architecture patterns: single LLM, RAG, agent, fine-tuned
- [ ] Latency budgets: SLA definition and how to meet them
- [ ] Cost modelling: input/output tokens × price, cost per query
- [ ] Caching strategies: exact match, semantic cache, Redis TTL

**MLOps for LLMs**
- [ ] Experiment tracking with W&B: metrics, artifacts, hyperparameter sweeps
- [ ] CI/CD for ML: GitHub Actions pipeline for model evaluation
- [ ] Model versioning: MLflow, HuggingFace Hub, Delta with Unity Catalog
- [ ] Prompt versioning: why and how to version prompts like code
- [ ] A/B testing for model updates: traffic splitting, guardrail metrics

**Monitoring & Observability**
- [ ] LLM tracing with LangSmith: traces, runs, feedback
- [ ] Production metrics: TTFT, TBT, tokens/sec, cost/query, error rate
- [ ] Drift detection: input distribution shift, output quality drift
- [ ] Alerting: what to alert on, what to just log

**Safety & Security**
- [ ] OWASP LLM Top 10: most important risks
- [ ] Prompt injection: detection and mitigation
- [ ] PII detection and redaction in outputs
- [ ] Input/output guardrails
- [ ] Safe code execution: sandboxing

**Evaluation**
- [ ] LLM-as-judge with rubrics
- [ ] HumanEval pass@k for code evaluation
- [ ] RAGAS for RAG evaluation
- [ ] Evaluation test set design: coverage, edge cases, adversarial

---

### 2. Practical Skills Checklist

- [ ] Deploy a RAG system with FastAPI + streaming + error handling
- [ ] Set up LangSmith tracing and annotate traces with feedback
- [ ] Implement a semantic cache with Redis
- [ ] Write a GitHub Actions workflow that runs evaluation on every PR
- [ ] Implement LLM drift monitoring (quality metrics over time)
- [ ] Build input and output guardrails (injection detection, PII redaction)
- [ ] Version prompts in a YAML registry and load them in code
- [ ] Write an LLM-as-judge evaluation pipeline with structured rubric

---

### 3. Coding Challenges

**Challenge A — Production RAG with Full Middleware**
```python
# Build a production-ready RAG API with ALL of the following:
# 1. FastAPI with streaming (StreamingResponse)
# 2. Input guardrails: detect prompt injection, validate input length
# 3. Semantic cache: Redis, embed query, cosine similarity >= 0.95 → return cached
# 4. Hybrid retrieval (BM25 + dense + reranker)
# 5. Context compression: LLM extracts only relevant sentences
# 6. Generation with citations
# 7. Output guardrails: detect PII in output
# 8. LangSmith trace on every request
# 9. Prometheus metrics: latency histogram, cache hit counter, error counter
# 10. Docker Compose deployment
```

**Challenge B — Automated Evaluation Pipeline**
```python
# Build a CI/CD evaluation pipeline:
# - GitHub Actions workflow triggered on every PR
# - Runs your RAG system on 50 test questions
# - Computes: RAGAS faithfulness, answer relevancy, context precision
# - Computes: LLM-as-judge quality score (avg 1-5)
# - Fails the PR if: any metric drops > 5% from main branch
# - Posts evaluation report as PR comment
# Use GitHub Actions cache to avoid re-embedding the test corpus on every run
```

**Challenge C — Cost Optimiser**
```python
# Write a function that recommends the cheapest model for a given SLA:
# Inputs: max_latency_ms, min_quality_score (1-5), queries_per_day
# Options: gpt-4o ($5/1M in), gpt-4o-mini ($0.15/1M in), claude-haiku ($0.25/1M in),
#          local llama-3.1-8b ($0/query but hosting cost)
# Algorithm:
#   1. For each model: run 20 sample queries, measure latency and quality
#   2. Filter models meeting SLA
#   3. Calculate monthly cost for each
#   4. Return cheapest model meeting both latency and quality SLA
# Output: recommendation report with cost comparison table
```

---

### 4. Mini Project

**Production RAG System** (Project 22 from the roadmap):
- Full system architecture: FastAPI → guardrails → semantic cache → RAG → streaming
- LangSmith tracing on every request
- RAGAS automated evaluation (weekly cron job)
- Grafana dashboard: p50/p95 latency, cost/query, cache hit rate, quality score
- CI/CD: GitHub Actions evaluation gate on every PR
- Docker Compose deployment with documentation

---

### 5. Capstone Project

**AI Platform for Your Team**: Build a platform that:
- Wraps your fine-tuned Databricks expert model (Phase 7a)
- Behind a FastAPI service with: rate limiting, auth (API key), streaming
- With a RAG layer using Databricks docs as the knowledge base
- Full LangSmith observability and weekly automated evaluation
- GitHub Actions CI/CD that prevents regressions
- Grafana monitoring dashboard
- Security: input validation, output filtering, audit logging

---

### 6. Interview Questions

**Beginner**

1. **Q: What is MLOps and why does it matter for LLMs specifically?**
   A: MLOps applies DevOps principles to ML: versioning, CI/CD, monitoring. For LLMs specifically: models degrade as world knowledge becomes outdated; prompts behave like code (must be versioned and tested); output quality is subjective and requires automated evaluation; cost needs monitoring. Without MLOps, production LLM systems become unmaintainable.

2. **Q: What is LangSmith and what does it provide?**
   A: LangSmith is an LLM observability platform. It traces every LLM call, tool call, and chain execution as a tree structure. Provides: detailed latency per step, cost per call, inputs/outputs at every node, human feedback collection, dataset creation from traces, automated evaluation.

3. **Q: What is a semantic cache and when does it provide a cache hit?**
   A: A semantic cache stores (query, response) pairs. On a new query: embed it and find the nearest cached query by cosine similarity. If similarity > threshold (e.g., 0.95), return the cached response. Benefits: identical questions hit the cache; very similar rephrasings also hit. Saves cost and latency.

4. **Q: What is a guardrail in an AI system?**
   A: Guardrails are validation layers that check inputs and outputs. Input guardrails: detect prompt injection, validate length, filter disallowed topics. Output guardrails: check for PII, detect harmful content, verify format. They prevent misuse and improve reliability.

5. **Q: What is prompt versioning and why is it important?**
   A: Prompts are code — they produce different behaviour when changed. Versioning: store prompts in YAML, commit to git, test before merging. Benefits: reproducibility, rollback on regression, A/B testing. Without versioning, you can't explain why outputs changed after a "small" prompt tweak.

6. **Q: What are CI/CD evaluation gates for AI systems?**
   A: Automated tests that run on every PR: run the AI system on a fixed test set, compute quality metrics (RAGAS, LLM-as-judge), compare to the baseline (main branch), fail the PR if metrics regress. Same as unit tests for regular code but measuring AI output quality.

7. **Q: What is TTFT (Time To First Token) and why does it matter for UX?**
   A: TTFT is the latency from sending the request to receiving the first token. It determines perceived responsiveness. Users feel a response is slow if TTFT > 1-2 seconds even if subsequent tokens stream fast. Optimise TTFT with: smaller prompts, faster prefill (FlashAttention), server-side batching.

**Intermediate**

8. **Q: How would you design a cost monitoring system for a production LLM application?**
    A: (1) Track input and output tokens per request (LangSmith or custom). (2) Multiply by model price per token. (3) Aggregate by: user, endpoint, model, time period. (4) Alert on: cost/day exceeding budget, cost/query anomalies. (5) Weekly report: highest-cost queries, cost breakdown by feature.

9. **Q: What is LLM drift and how do you detect it?**
    A: LLM drift: API model updates (gpt-4o-2024-05-13 → gpt-4o-2024-08-06), base model behaviour changes, or your domain data drifts. Detection: (1) run a fixed eval set periodically; (2) monitor LLM-as-judge quality scores over time; (3) track output format compliance; (4) alert on statistically significant metric changes.

10. **Q: How would you safely deploy a new version of an LLM-powered feature?**
    A: Blue-green deployment: run old version alongside new. Canary: route 5% of traffic to new version, monitor metrics, gradually increase. Evaluation gate: must pass CI evaluation before merging. Rollback plan: keep old model/prompt version deployed and ready to reactivate. Shadow mode: run new version in parallel but don't serve its output until validated.

11. **Q: What is the OWASP LLM Top 10 and which vulnerabilities are most important?**
    A: Most critical: (1) Prompt Injection — malicious input overrides instructions; (2) Insecure Output Handling — using LLM output in SQL/shell without sanitisation; (3) Sensitive Information Disclosure — model leaks training data or system prompts; (4) Over-reliance — treating LLM output as authoritative without verification; (5) Model Denial of Service — extremely long inputs exhaust compute.

12. **Q: How do you prevent prompt injection in a RAG system?**
    A: (1) Classify user input before processing: LLM or classifier detects "ignore previous instructions" patterns. (2) Use separate context markers: clearly delimit "retrieved content" vs. "user query" in the prompt. (3) Privilege separation: process retrieved content with different trust level than user input. (4) Output validation: check if output suspiciously references system instructions.

13. **Q: What is the difference between pass@1 and pass@k in code evaluation?**
    A: pass@1: generate 1 solution, check if it passes tests. pass@k: generate k solutions, the problem is "solved" if any 1 passes. pass@k is estimated using statistical sampling to avoid the variance of a single run. HumanEval uses pass@1 (single sample) and pass@10, pass@100 for comparison.

**Advanced**

14. **Q: Design a production RAG system for 1 million queries per day.**
    A: (1) Redis semantic cache: ~70% cache hit rate → reduces LLM calls to 300k/day. (2) vLLM serving: horizontal scaling with load balancer. (3) Qdrant distributed: multi-node vector search. (4) Async architecture: FastAPI + asyncio + background RAGAS evaluation. (5) Cost: 300k × 1000 tokens × $0.15/1M = $45/day for input. (6) Monitoring: Prometheus + Grafana, alert on p95 latency and error rate.

15. **Q: How would you handle model versioning when you have 1000+ fine-tuned adapters?**
    A: (1) MLflow Model Registry: each adapter registered with metadata (task, dataset, eval scores, date). (2) Tags: promoted to "Staging" → "Production" based on eval gates. (3) Lazy loading: load adapters on-demand with LRU cache. (4) A/B testing: route specific user segments to different adapters. (5) Automatic rollback: if quality drops below threshold, revert to previous version.

16. **Q: What is shadow mode deployment and when is it useful?**
    A: Shadow mode: run the new model on every request in parallel with production, log outputs, but don't serve the new model's output to users. Use to: (1) validate new model behaviour at scale before cutover; (2) collect comparison data for human evaluation; (3) detect edge cases that don't appear in offline testing. Useful when offline eval is insufficient.

17. **Q: How would you implement rate limiting for an LLM API to prevent abuse?**
    A: (1) Token bucket per API key: allow N requests/minute, refill at rate r. Redis-based implementation: INCR counter + EXPIRE TTL. (2) Cost-based limiting: track token spend per user, limit monthly spend. (3) Slow response for rate-limited users (degrade gracefully). (4) Hard limits + alerts for unusual patterns (potential scraping or DDoS).

18. **Q: What is evaluation dataset design for production AI and what makes a good test set?**
    A: Good test set: (1) representative of production queries (sample from actual traffic); (2) covers edge cases (empty input, very long input, ambiguous questions); (3) includes adversarial examples (prompt injection, confusing phrasings); (4) has ground truth answers verified by domain experts; (5) stable — same questions used across versions for comparability. Size: 50-500 questions depending on task complexity.

19. **Q: How do you implement PII detection in LLM outputs?**
    A: (1) Regex patterns for obvious PII: SSNs, credit cards, phone numbers, emails. (2) Named entity recognition (spaCy, HuggingFace NER): detect names, addresses, organisations. (3) LLM-based PII classifier for less structured PII. (4) Redaction: replace detected PII with `[REDACTED]` or `[NAME]`. (5) Log PII detections for security audit (without the PII itself).

20. **Q: How would you handle a situation where the LLM produces different outputs for the same prompt?**
    A: (1) Use temperature=0 for deterministic outputs where consistency is critical. (2) Set a fixed seed if the API supports it. (3) For evaluations: run 3-5 samples and average scores. (4) For production: if consistency is a business requirement, cache responses or use a database of approved responses. (5) Accept some variability as a feature if diverse outputs are desired.

---

### 7. Self-Assessment Quiz

- [ ] What is TTFT and how do you optimise it?
- [ ] What is LangSmith and what does a trace look like?
- [ ] What is a semantic cache hit rate and what threshold do you use?
- [ ] Name 3 OWASP LLM Top 10 risks.
- [ ] What is a guardrail in an AI system?
- [ ] What is prompt versioning and why is it important?
- [ ] What is an evaluation gate in CI/CD?
- [ ] What is shadow mode deployment?
- [ ] What is pass@k in code evaluation?
- [ ] What is LLM drift and how do you detect it?
- [ ] What is the difference between monitoring and observability?
- [ ] What is the role of MLflow in production AI?
- [ ] What is cost per query and what drives it?
- [ ] What is canary deployment?
- [ ] How do you handle PII in LLM outputs?
- [ ] What is token budget and why does it matter for SLA?
- [ ] What is the difference between a latency SLA and a throughput SLA?
- [ ] What does W&B track for experiments?
- [ ] What is a rollback in the context of a production LLM system?
- [ ] What is the difference between online and offline evaluation?
- [ ] What is LLM-as-judge and what are its limitations?
- [ ] What is RAGAS and what 4 metrics does it compute?
- [ ] What is streaming in LLM APIs?
- [ ] What is rate limiting in an AI API context?
- [ ] What is a Prometheus metric and how is it different from a log?

**Scoring**: 22–25 ✅ = Ready. 17–21 = Review weak areas. Below 17 = Spend more time on Phase 9.

---

### 8. Common Mistakes

| Mistake | Why It Happens | How to Avoid |
|---------|---------------|--------------|
| Not versioning prompts | "They're just strings" | Prompts are code — commit them, track eval scores |
| Touching the test set during development | Tempting to look | Freeze test set before development; treat it as the held-out test set |
| No rollback plan | Happy path thinking | Always have the previous version ready to reactivate before deploying |
| Not measuring baseline before building cache | Assume cache helps | Measure cache hit rate after 1 week; if <30%, the queries are too diverse |
| Trusting LLM-as-judge without validating the judge | Circular evaluation | Run human evaluation on 10% of LLM-judge-scored examples to validate calibration |
| Deploying without load testing | Works for 1 user | Always load test with locust before launch; LLM APIs have surprising failure modes at scale |
| Alerting on everything | Everything seems important | Alert only on: latency SLA breach, error rate spike, cost anomaly; log everything else |

---

### 9. Readiness Criteria

You are ready for Phase 10 when **all** of the following are true:

- [ ] I built the Production RAG API with all middleware (Challenge A)
- [ ] I set up a GitHub Actions evaluation pipeline that can fail a PR (Challenge B)
- [ ] I deployed a system with LangSmith tracing and reviewed real traces
- [ ] I built a Grafana dashboard monitoring at least 3 production metrics
- [ ] I scored 22/25 or higher on the Self-Assessment Quiz
- [ ] I can answer at least 16/20 Interview Questions correctly
- [ ] I can design a production AI system architecture on a whiteboard

---

### 10. Revision Summary

```
PRODUCTION AI ARCHITECTURE
─────────────────────────────────────────────────────
Request  → Input Guardrails → Semantic Cache → RAG → LLM → Output Guardrails → Response
                                     ↓                         ↓
                               Cache hit → return       LangSmith trace
                                                         RAGAS eval (async)

KEY PRODUCTION CONCEPTS
─────────────────────────────────────────────────────
Versioning:    prompts + models = code; commit, test, rollback
CI/CD:         eval gate on every PR (RAGAS + LLM-as-judge)
Monitoring:    TTFT, TBT, cost/query, cache hit rate, quality score
Security:      OWASP LLM Top 10; prompt injection; PII detection
Deployment:    canary (5%) → shadow → blue-green → full rollout

EVALUATION
─────────────────────────────────────────────────────
Offline:   fixed test set, automated metrics, before deployment
Online:    production traffic sampling, human feedback, A/B test
LLM judge: structured rubric, validate calibration against human
```

---

### 11. Next Phase Prerequisites

**What Phase 10 (Research & Open Source) requires from Phase 9:**

| Phase 9 Concept | How Phase 10 Uses It |
|----------------|----------------------|
| Production system design | Writing design documents for open source contributions |
| Evaluation methodology | Evaluating and comparing research papers |
| MLOps practices | Applying rigorous benchmarking to paper implementations |
| System reliability | Contributing production-quality code to open source projects |
| Cost/performance analysis | Practical lens for evaluating research claims |

**The critical dependency**: Phase 10 is about contributing to and staying current with the research frontier. Doing this effectively requires production experience — you'll evaluate papers not just theoretically but practically ("would this work in the production system I built?"). This lens makes you a much better reviewer and contributor.

---

*Phase 9 | Part of the [GenAI Engineer Roadmap](./00_README.md)*
