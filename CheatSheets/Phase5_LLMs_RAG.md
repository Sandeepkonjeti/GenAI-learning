# Phase 5 — LLMs, Prompting & RAG Cheat Sheet

[← CheatSheets Index](./README.md) | [Full Phase Files](../06_Phase5_Part1_LLMs_and_Prompt_Engineering.md)

---

## LLM API Quick Reference

```python
# --- OPENAI ---
from openai import OpenAI
client = OpenAI()  # uses OPENAI_API_KEY env var

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain attention in one paragraph."},
    ],
    temperature=0.0,
    max_tokens=500,
)
text = response.choices[0].message.content

# --- STREAMING ---
with client.chat.completions.stream(model="gpt-4o", messages=messages) as stream:
    for text_chunk in stream.text_stream:
        print(text_chunk, end="", flush=True)

# --- ANTHROPIC ---
import anthropic
client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    system="You are a data engineering expert.",
    messages=[{"role": "user", "content": "..."}],
)
```

---

## Structured Outputs (Production Standard)

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    summary: str
    key_findings: list[str]
    confidence: float  # 0.0–1.0

client = instructor.from_openai(OpenAI())

result = client.chat.completions.create(
    model="gpt-4o",
    response_model=AnalysisResult,  # returns typed object
    max_retries=3,                  # auto-retries on validation failure
    messages=[{"role": "user", "content": "Analyze this data..."}],
)
# result.summary, result.key_findings — type-safe!
```

---

## Prompting Techniques

| Technique | When to Use | Example Pattern |
|-----------|-------------|----------------|
| Zero-shot | General tasks | `"Classify: [text]"` |
| Few-shot | Needs format examples | 3–5 examples before query |
| CoT | Reasoning tasks | `"Think step by step:"` |
| Self-consistency | Reliability on math/logic | Sample N times, majority vote |
| ReAct | Agent tasks | `Thought → Action → Observation` loop |
| System prompt | Persistent persona/rules | Set once per conversation |

```python
# FEW-SHOT TEMPLATE
few_shot = """
Classify sentiment. Respond with: Positive, Negative, or Neutral.

Input: "The product exceeded my expectations!"
Output: Positive

Input: "Terrible quality, broke after one day."
Output: Negative

Input: "{user_input}"
Output:"""
```

---

## Prompt Security

| Attack | Description | Mitigation |
|--------|------------|-----------|
| Prompt injection | User input overrides system prompt | Input sanitization, output validation |
| Jailbreak | Bypass safety guardrails | Use guardrails library or content filters |
| Data exfiltration | LLM leaks context data | Never put secrets in context |

---

## RAG Architecture

```
User Query
    ↓ Embed query → vector
    ↓ Search vector DB (cosine similarity, top-k)
    ↓ Retrieve relevant chunks
    ↓ Augment prompt: "Context:\n{chunks}\n\nQuestion:{query}"
    ↓ LLM generates answer grounded in context
```

### Chunking Strategies

| Strategy | Chunk Size | Use When |
|----------|:----------:|----------|
| Fixed size | 512–1024 tokens | General purpose |
| Sentence-based | Sentence boundary | Conversational text |
| Semantic | Embedding similarity | Technical docs |
| Recursive | Hierarchy (section → paragraph) | Long documents |
| **Contextual** | + document-level context prepended | **Best retrieval quality** |

---

## Embedding + Vector DB Quick Reference

```python
# Embed with OpenAI
from openai import OpenAI
client = OpenAI()

def embed(text: str) -> list[float]:
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding  # 1536-dim vector

# Store + search with Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

qc = QdrantClient(":memory:")
qc.create_collection("docs", vectors_config=VectorParams(size=1536, distance=Distance.COSINE))

# Insert
qc.upsert("docs", points=[PointStruct(id=1, vector=embed("doc text"), payload={"text": "..."})])

# Search
results = qc.search("docs", query_vector=embed("user query"), limit=5)
```

---

## RAG Evaluation (RAGAS)

| Metric | Measures | Range |
|--------|---------|:-----:|
| Faithfulness | Is answer grounded in context? | 0–1 |
| Answer Relevance | Does answer address the question? | 0–1 |
| Context Precision | Are retrieved chunks relevant? | 0–1 |
| Context Recall | Are relevant chunks retrieved? | 0–1 |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
```

---

## Common Mistakes

- **Temperature=1.0 in production** — use 0.0–0.2 for factual tasks
- **No chunking overlap** — context at chunk boundary is split; use 10–20% overlap
- **Embedding query differently than documents** — use same model
- **No validation on structured outputs** — use Pydantic + instructor
- **Trusting LLM output without retrieval grounding** — causes hallucination

---

## Interview Quick-Hits

**Q: How do you reduce hallucination in RAG?**  
A: (1) Improve retrieval (better chunking, hybrid search, reranking). (2) Validate faithfulness with RAGAS or LLM-as-judge. (3) Add a "no answer" option when retrieved context is insufficient. (4) Use temperature=0 for factual tasks.

**Q: What is the difference between semantic search and keyword search, and when combine them?**  
A: Semantic search (embedding similarity) catches meaning but misses exact matches. Keyword search (BM25) is exact but misses paraphrases. Hybrid search combines both; use a reranker (cross-encoder) to merge results.

**Q: How would you evaluate a RAG system before deploying?**  
A: Build a ground-truth dataset of Q&A pairs. Evaluate: RAGAS faithfulness + relevance metrics, latency P50/P95, retrieval MRR@k, end-to-end accuracy on held-out test set, A/B test vs. previous version.

**Q: What is "contextual retrieval" (Anthropic 2024)?**  
A: Before embedding each chunk, prepend a short document-level context summary (generated by Claude). This gives each chunk more semantic information and dramatically improves retrieval recall on document subsets.
