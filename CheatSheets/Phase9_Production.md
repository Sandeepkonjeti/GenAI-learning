# Phase 9 — Production AI Engineering Cheat Sheet

[← CheatSheets Index](./README.md) | [Full Phase File](../10_Phase9_Production_AI_Engineering.md)

---

## Production API Pattern (FastAPI)

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio, time, redis.asyncio as redis

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

# --- STREAMING ENDPOINT ---
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    async def token_generator():
        async with openai_client.chat.completions.stream(
            model="gpt-4o", messages=[{"role": "user", "content": req.message}]
        ) as stream:
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield f"data: {chunk.choices[0].delta.content}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(token_generator(), media_type="text/event-stream")
```

---

## Latency Metrics

| Metric | Definition | Good Target |
|--------|-----------|:-----------:|
| TTFT | Time to First Token | < 500ms |
| TPOT | Time Per Output Token | < 30ms |
| Total latency | TTFT + TPOT × tokens | < 5s for 100 tokens |
| P95 latency | 95th percentile | < 2× median |

---

## Semantic Cache (Redis)

```python
# Pattern: hash query embedding → lookup in Redis before calling LLM
async def cached_complete(query: str) -> str:
    embedding = embed(query)
    
    # Check cache
    cached = await redis_client.get(embedding_hash(embedding))
    if cached:
        return cached.decode()
    
    # Call LLM if miss
    response = await llm_call(query)
    
    # Store with TTL
    await redis_client.setex(embedding_hash(embedding), 3600, response)
    return response
```

**Cache hit rate target**: 30–70% depending on query diversity. Measure and tune similarity threshold.

---

## MLOps / LLMOps Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Experiment tracking | MLflow, W&B | Log params, metrics, artifacts |
| LLM tracing | LangSmith | Trace agent/chain calls |
| Deployment | FastAPI + Docker | Serve model API |
| Orchestration | Kubernetes / Databricks | Scale serving |
| Monitoring | Prometheus + Grafana | Metrics, alerts |
| Evaluation CI | RAGAS + GitHub Actions | Quality gate on every PR |

---

## Databricks AI Stack

| Service | What it Does | When to Use |
|---------|-------------|-------------|
| Mosaic AI Model Serving | Serverless LLM deployment | Production inference |
| Mosaic AI Vector Search | Managed Qdrant-like service | RAG in Databricks |
| AI Gateway | Centralized LLM proxy (rate limit, logging) | Multi-team LLM governance |
| MLflow (LLM tracking) | Log prompts, responses, evals | Every experiment |
| TorchDistributor | Distributed PyTorch on Spark | Fine-tuning at scale |

```python
# MLflow LLM tracking
import mlflow

mlflow.set_experiment("rag-evaluation")
with mlflow.start_run():
    mlflow.log_params({"model": "gpt-4o", "chunk_size": 512, "top_k": 5})
    mlflow.log_metrics({"faithfulness": 0.92, "relevance": 0.87, "latency_p95": 1.3})
    mlflow.log_text(prompt_template, "prompt_template.txt")
```

---

## LLM Observability Checklist

For every production LLM request, log:
- [ ] Request ID + timestamp
- [ ] Model name + version
- [ ] Prompt (sanitized)
- [ ] Response
- [ ] Token usage (prompt + completion)
- [ ] Latency (TTFT, TPOT)
- [ ] Cost estimate
- [ ] User feedback (if available)

---

## OWASP LLM Top 10 (Quick Reference)

| # | Risk | Mitigation |
|---|------|-----------|
| LLM01 | Prompt Injection | Input sanitization, output validation |
| LLM02 | Insecure Output Handling | Validate/sanitize LLM output before use |
| LLM03 | Training Data Poisoning | Data validation, provenance tracking |
| LLM04 | Model Denial of Service | Rate limiting, token limits |
| LLM05 | Supply Chain Vulnerabilities | Pin model versions, verify checksums |
| LLM06 | Sensitive Information Disclosure | PII detection, response filtering |
| LLM07 | Insecure Plugin Design | Validate tool inputs/outputs |
| LLM08 | Excessive Agency | Principle of least privilege for tools |
| LLM09 | Overreliance | Human-in-the-loop for critical decisions |
| LLM10 | Model Theft | Access controls, watermarking |

---

## CI/CD Evaluation Gate

```yaml
# .github/workflows/eval.yml
- name: Run RAG evaluation
  run: |
    python evaluate.py \
      --test-set tests/golden_set.json \
      --min-faithfulness 0.85 \
      --min-relevance 0.80
    
- name: Check performance regression
  run: |
    python benchmark.py \
      --max-latency-p95 2000 \  # ms
      --min-throughput 50        # req/s
```

---

## Common Mistakes

- **No rate limiting** — single user can exhaust LLM budget
- **Logging full prompts** in plain text — PII risk; sanitize first
- **No fallback** when primary model is down — implement model fallback chain
- **Sync endpoint for LLM calls** — use async to handle concurrent requests
- **No token budget** — set `max_tokens` always; LLMs can generate unboundedly

---

## Interview Quick-Hits

**Q: How do you design a production RAG system from scratch?**  
A: FastAPI async endpoints → semantic cache (Redis) → vector search (Qdrant/Databricks Vector Search) → LLM with streaming → LangSmith tracing → RAGAS evaluation CI gate → Prometheus metrics → Grafana dashboards. Each layer decoupled and independently scalable.

**Q: How do you handle LLM cost at scale?**  
A: (1) Semantic caching (30–70% hit rate). (2) Smaller models for simple tasks. (3) Prompt compression (remove redundant context). (4) Batching for async workloads. (5) Rate limiting per user/tenant. (6) Cost monitoring with budget alerts.

**Q: What metrics do you monitor for a production LLM API?**  
A: TTFT, TPOT, token/s throughput, error rate, cache hit rate, cost per query, faithfulness/relevance scores from eval pipeline, user feedback signal.

**Q: How do you gate quality in an LLM CI/CD pipeline?**  
A: Maintain a golden test set with expected answers. On each PR, run RAGAS evaluation — block merge if faithfulness drops below threshold. Also benchmark latency to catch regressions.
