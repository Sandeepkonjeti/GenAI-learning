# Project Index — All 22+ Projects
**Complete catalog for the GenAI Engineer Roadmap**

> [← Index](./INDEX.md) | [Progress Tracker](./PROGRESS_TRACKER.md) | [Full Project Guide](./12_Projects_Beginner_to_Expert.md)

---

## Summary

| Category | Projects | Count |
|----------|---------|:-----:|
| Beginner (Phase 0–2) | Environment setup through ML pipeline | 6 |
| Intermediate (Phase 3–4) | Backprop to transformers | 5 |
| Advanced (Phase 5–6) | LLMs, RAG, agents | 8 |
| Expert (Phase 7–9) | Fine-tuning, production | 5 |
| **Total** | | **24** |

**Skills unlocked by completing all projects**: Python, NumPy, ML from scratch, PyTorch, NLP, transformers, LLMs, prompt engineering, embeddings, RAG, agents, MCP, LoRA fine-tuning, RLHF/DPO, multimodal, inference optimization, production deployment, MLOps

---

## Beginner Projects (Months 1–6)

### Project 1 — Python AI Toolkit Setup

| Field | Value |
|-------|-------|
| **Phase** | 0 |
| **Difficulty** | 1/10 |
| **Time** | 1 day |
| **Technologies** | WSL2, conda/uv, Python 3.11, Docker |
| **Skills** | Environment setup, dependency management, secrets |
| **Resume Value** | Low (expected) |
| **Interview Value** | Enables demonstrating other projects |
| **Output** | Working dev environment + `setup_test.py` |
| **Status** | 🔲 |

---

### Project 2 — NumPy Matrix Operations Library

| Field | Value |
|-------|-------|
| **Phase** | 0 |
| **Difficulty** | 2/10 |
| **Time** | 3 days |
| **Technologies** | NumPy, Matplotlib |
| **Skills** | Vectorized ops, broadcasting, einsum, benchmarking |
| **Resume Value** | Low (prerequisite skill) |
| **Interview Value** | Demonstrates numerical Python fluency |
| **Output** | Benchmarked notebook showing 100–1000× speedup |
| **Status** | 🔲 |

---

### Project 3 — Linear Regression from Scratch

| Field | Value |
|-------|-------|
| **Phase** | 1–2 |
| **Difficulty** | 3/10 |
| **Time** | 1 week |
| **Technologies** | NumPy, Matplotlib, sklearn (comparison only) |
| **Skills** | Gradient descent, L1/L2 regularization, loss visualization |
| **Resume Value** | Medium — shows you understand training, not just APIs |
| **Interview Value** | High — "implement gradient descent" is a classic |
| **Output** | Notebook with implementation vs. sklearn comparison |
| **Extension** | Add momentum and Adam; compare convergence |
| **Status** | 🔲 |

---

### Project 4 — Neural Network from Scratch (No PyTorch)

| Field | Value |
|-------|-------|
| **Phase** | 1–3 |
| **Difficulty** | 5/10 |
| **Time** | 2 weeks |
| **Technologies** | NumPy only, Matplotlib |
| **Skills** | Backpropagation, activation functions, cross-entropy gradient |
| **Resume Value** | **High** — rare, demonstrates real understanding |
| **Interview Value** | **Very high** — "how does backprop work" with code |
| **Output** | Neural net library, MNIST >95% accuracy |
| **Extension** | Add batch norm, dropout |
| **Status** | 🔲 |

---

### Project 5 — Machine Learning Pipeline

| Field | Value |
|-------|-------|
| **Phase** | 2 |
| **Difficulty** | 4/10 |
| **Time** | 1.5 weeks |
| **Technologies** | Pandas, sklearn, XGBoost, SHAP |
| **Skills** | EDA, feature engineering, cross-validation, model selection |
| **Resume Value** | Medium |
| **Interview Value** | High for ML roles |
| **Output** | Reusable pipeline module + analysis notebook |
| **Status** | 🔲 |

---

### Project 6 — Backpropagation Visualizer

| Field | Value |
|-------|-------|
| **Phase** | 3a |
| **Difficulty** | 4/10 |
| **Time** | 1 week |
| **Technologies** | Python, Graphviz |
| **Skills** | Autograd mechanics, computation graphs |
| **Resume Value** | Medium — strong signal of deep understanding |
| **Interview Value** | High — "how does autograd work" |
| **Reference** | Karpathy's micrograd |
| **Output** | Working autograd engine + visualization |
| **Status** | 🔲 |

---

## Intermediate Projects (Months 7–11)

### Project 7 — PyTorch Image Classifier

| Field | Value |
|-------|-------|
| **Phase** | 3b |
| **Difficulty** | 5/10 |
| **Time** | 1 week |
| **Technologies** | PyTorch, torchvision, W&B or TensorBoard |
| **Skills** | CNN, training loop, LR scheduler, early stopping |
| **Resume Value** | Medium |
| **Interview Value** | Medium — baseline PyTorch skill |
| **Output** | Trained CNN (>99% MNIST, >90% CIFAR-10) |
| **Status** | 🔲 |

---

### Project 8 — NLP Pipeline

| Field | Value |
|-------|-------|
| **Phase** | 4a |
| **Difficulty** | 4/10 |
| **Time** | 1 week |
| **Technologies** | NLTK, sklearn, sentence-transformers |
| **Skills** | Text preprocessing, TF-IDF, Word2Vec, sentiment analysis |
| **Resume Value** | Medium |
| **Interview Value** | Medium |
| **Status** | 🔲 |

---

### Project 9 — Transformer from Scratch

| Field | Value |
|-------|-------|
| **Phase** | 4b |
| **Difficulty** | 8/10 |
| **Time** | 2 weeks |
| **Technologies** | PyTorch only |
| **Skills** | Multi-head attention, GQA, KV cache, causal masking, positional encoding |
| **Resume Value** | **Very High** — extremely rare capability |
| **Interview Value** | **Very High** — demonstrates true transformer understanding |
| **Output** | GPT implementation + trained on Shakespeare |
| **Status** | 🔲 |

---

### Project 10 — GPT-2 Conversational Demo

| Field | Value |
|-------|-------|
| **Phase** | 4b |
| **Difficulty** | 5/10 |
| **Time** | 1 week |
| **Technologies** | Transformers (HuggingFace), Gradio |
| **Skills** | Using pretrained models, tokenization, generation strategies |
| **Resume Value** | Medium |
| **Interview Value** | Low (too common) |
| **Status** | 🔲 |

---

### Project 11 — LLM Code Documentation Generator

| Field | Value |
|-------|-------|
| **Phase** | 5a |
| **Difficulty** | 5/10 |
| **Time** | 1 week |
| **Technologies** | OpenAI API, instructor, Pydantic |
| **Skills** | Prompt engineering, structured outputs, AST parsing |
| **Resume Value** | High — practical tool with real use |
| **Interview Value** | High — demonstrates production LLM patterns |
| **Status** | 🔲 |

---

## Advanced Projects (Months 12–15)

### Project 12 — Multi-Model Benchmark Tool

| Field | Value |
|-------|-------|
| **Phase** | 5a |
| **Difficulty** | 5/10 |
| **Time** | 1 week |
| **Technologies** | OpenAI, Anthropic, HuggingFace APIs |
| **Skills** | API design, evaluation, cost tracking, benchmarking |
| **Resume Value** | High |
| **Interview Value** | High — shows you think about evaluation |
| **Status** | 🔲 |

---

### Project 13 — Chain-of-Thought Math Solver

| Field | Value |
|-------|-------|
| **Phase** | 5a |
| **Difficulty** | 4/10 |
| **Time** | 4 days |
| **Technologies** | OpenAI API |
| **Skills** | CoT prompting, few-shot, self-consistency |
| **Resume Value** | Medium |
| **Interview Value** | Medium |
| **Status** | 🔲 |

---

### Project 14 — Document QA System

| Field | Value |
|-------|-------|
| **Phase** | 5b |
| **Difficulty** | 6/10 |
| **Time** | 1.5 weeks |
| **Technologies** | sentence-transformers, ChromaDB/Qdrant, OpenAI |
| **Skills** | Chunking, embedding, retrieval, RAG pipeline |
| **Resume Value** | **High** — most common production AI system |
| **Interview Value** | **Very High** — asked in most LLM interviews |
| **Status** | 🔲 |

---

### Project 15 — RAG Evaluation Benchmark

| Field | Value |
|-------|-------|
| **Phase** | 5b |
| **Difficulty** | 7/10 |
| **Time** | 1.5 weeks |
| **Technologies** | RAGAS, Qdrant, LangSmith |
| **Skills** | RAGAS metrics, LLM-as-judge, evaluation dataset curation |
| **Resume Value** | **High** — shows rigor |
| **Interview Value** | **Very High** — "how do you evaluate RAG?" |
| **Status** | 🔲 |

---

### Project 16 — Personal Research Agent

| Field | Value |
|-------|-------|
| **Phase** | 6 |
| **Difficulty** | 7/10 |
| **Time** | 2 weeks |
| **Technologies** | LangGraph, OpenAI, search APIs |
| **Skills** | ReAct, tool use, memory, LangGraph state machines |
| **Resume Value** | **High** |
| **Interview Value** | **Very High** — agent architecture is hot |
| **Status** | 🔲 |

---

### Project 17 — Databricks MCP Server

| Field | Value |
|-------|-------|
| **Phase** | 6 |
| **Difficulty** | 7/10 |
| **Time** | 1 week |
| **Technologies** | MCP SDK (Python), Databricks SDK |
| **Skills** | MCP protocol, tool exposure, AI gateway integration |
| **Resume Value** | **Very High** — uniquely rare capability |
| **Interview Value** | **Very High** — differentiating project |
| **Status** | 🔲 |

---

### Project 17b — Text-to-SQL Agent

| Field | Value |
|-------|-------|
| **Phase** | 6 |
| **Difficulty** | 6/10 |
| **Time** | 1 week |
| **Technologies** | OpenAI API, SQLite/DuckDB, simpleeval |
| **Skills** | Schema-aware prompting, error correction loops, SQL safety |
| **Resume Value** | **High** — perfect for data engineering background |
| **Interview Value** | **High** — combines SQL + LLM knowledge |
| **Status** | 🔲 |

---

## Expert Projects (Months 15–18+)

### Project 18 — Domain Expert Fine-Tune

| Field | Value |
|-------|-------|
| **Phase** | 7a |
| **Difficulty** | 8/10 |
| **Time** | 2 weeks |
| **Technologies** | Unsloth/TRL, HuggingFace, W&B, QLoRA |
| **Skills** | Dataset preparation, QLoRA, training monitoring, eval |
| **Resume Value** | **Very High** — fine-tuning is a scarce skill |
| **Interview Value** | **Very High** — "how do you fine-tune an LLM?" |
| **Output** | Fine-tuned Databricks expert model |
| **Status** | 🔲 |

---

### Project 19 — DPO Alignment Fine-Tune

| Field | Value |
|-------|-------|
| **Phase** | 7b |
| **Difficulty** | 9/10 |
| **Time** | 2 weeks |
| **Technologies** | TRL (DPO Trainer), HuggingFace |
| **Skills** | Preference data curation, DPO training, alignment evaluation |
| **Resume Value** | **Very High** |
| **Interview Value** | **Very High** — alignment is frontier work |
| **Status** | 🔲 |

---

### Project 20 — Multimodal Document QA

| Field | Value |
|-------|-------|
| **Phase** | 8 |
| **Difficulty** | 7/10 |
| **Time** | 1.5 weeks |
| **Technologies** | LLaVA or GPT-4V, PDF parsing, Qdrant |
| **Skills** | Vision-language models, multimodal RAG |
| **Resume Value** | High |
| **Interview Value** | High — multimodal is current hot topic |
| **Status** | 🔲 |

---

### Project 21 — Optimized LLM Inference Server

| Field | Value |
|-------|-------|
| **Phase** | 8 |
| **Difficulty** | 8/10 |
| **Time** | 1.5 weeks |
| **Technologies** | vLLM, GPTQ/AWQ, Prometheus, Grafana |
| **Skills** | Quantization, PagedAttention, throughput benchmarking |
| **Resume Value** | **Very High** — infra knowledge differentiates |
| **Interview Value** | **Very High** — production systems interviews |
| **Status** | 🔲 |

---

### Project 22 — Production RAG with Full Observability

| Field | Value |
|-------|-------|
| **Phase** | 9 |
| **Difficulty** | 9/10 |
| **Time** | 3 weeks |
| **Technologies** | FastAPI, Redis, Qdrant, LangSmith, Prometheus, Grafana, GitHub Actions, Docker Compose |
| **Skills** | All production patterns: caching, guardrails, streaming, CI/CD eval, monitoring |
| **Resume Value** | **Portfolio Centerpiece** |
| **Interview Value** | **Maximum** — demonstrates production-grade thinking |
| **Output** | Deployed API + monitoring dashboard + CI/CD pipeline |
| **Status** | 🔲 |

---

## Project Portfolio Strategy

### GitHub Profile Structure

```
github.com/[yourname]/
├── genai-fundamentals/       # Projects 1–6 (NumPy, ML from scratch)
├── transformer-from-scratch/ # Project 9 ⭐ (most impressive)
├── rag-systems/              # Projects 14–15
├── ai-agents/                # Projects 16–17b ⭐
├── llm-finetuning/           # Projects 18–19 ⭐
└── production-ai/            # Project 22 ⭐ (portfolio centerpiece)
```

### Three Projects Every LLM Engineer Must Have

1. **Transformer from scratch** (Project 9) — proves you understand the architecture
2. **End-to-end RAG system** (Project 14–15) — most common production use case
3. **Production AI API** (Project 22) — proves you can ship things

### Resume Writing Guide

| Project | How to describe it |
|---------|-------------------|
| Project 4 | "Implemented neural network backpropagation from scratch in NumPy; achieved 97.3% test accuracy on MNIST" |
| Project 9 | "Built transformer architecture from scratch in PyTorch including GQA, KV cache, and RoPE positional encoding" |
| Project 18 | "Fine-tuned LLaMA-3-8B on proprietary Databricks documentation using QLoRA; reduced hallucination rate from 34% to 8%" |
| Project 22 | "Deployed production RAG API serving 1000 RPM; integrated LangSmith tracing, semantic cache (67% hit rate), and RAGAS evaluation CI/CD gate" |

---

*[← Index](./INDEX.md) | [Project Guide](./12_Projects_Beginner_to_Expert.md) | [Progress Tracker](./PROGRESS_TRACKER.md)*
