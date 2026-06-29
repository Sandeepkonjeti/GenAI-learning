# Learning Graph — Dependency Map
**Visual guide to the GenAI curriculum structure**

> [← Index](./INDEX.md) | Use this to plan alternative paths or understand prerequisites

---

## Core Learning Path

```mermaid
flowchart TD
    P0["Phase 0\nPython + CS Fundamentals\n~4 weeks"]
    P1["Phase 1\nMathematics\n~8 weeks"]
    P2["Phase 2\nClassical ML\n~8 weeks"]
    P3a["Phase 3a\nDeep Learning Theory\n~4 weeks"]
    P3b["Phase 3b\nPyTorch\n~4 weeks"]
    P4a["Phase 4a\nNLP Fundamentals\n~4 weeks"]
    P4b["Phase 4b\nTransformers\n~6 weeks"]
    P5a["Phase 5a\nLLMs + Prompting\n~4 weeks"]
    P5b["Phase 5b\nEmbeddings + RAG\n~6 weeks"]
    P6["Phase 6\nAgents + MCP\n~6 weeks"]
    P7a["Phase 7a\nFine-Tuning + LoRA\n~6 weeks"]
    P7b["Phase 7b\nRLHF + DPO\n~4 weeks"]
    P8["Phase 8\nAdvanced Topics\n~8 weeks"]
    P9["Phase 9\nProduction AI\n~8 weeks"]
    P10["Phase 10\nResearch + Open Source\n~ongoing"]

    P0 --> P1
    P0 --> P2
    P1 --> P2
    P2 --> P3a
    P3a --> P3b
    P3b --> P4a
    P4a --> P4b
    P4b --> P5a
    P5a --> P5b
    P5b --> P6
    P5a --> P7a
    P4b --> P7a
    P7a --> P7b
    P6 --> P8
    P7b --> P8
    P8 --> P9
    P5b --> P9
    P9 --> P10
```

---

## Prerequisite Detail Map

```mermaid
flowchart LR
    subgraph "Foundation (Can learn in parallel)"
        direction TB
        NUMPY["NumPy / Pandas"]
        LINALG["Linear Algebra"]
        CALCULUS["Calculus / Backprop"]
        PROB["Probability / Stats"]
    end

    subgraph "ML Core"
        direction TB
        CLASSML["Classical ML\n(sklearn, XGBoost)"]
        EVAL["Model Evaluation\n(CV, metrics)"]
    end

    subgraph "Deep Learning"
        direction TB
        NN["Neural Networks\n(from scratch)"]
        PYTORCH["PyTorch\n(autograd, GPU)"]
    end

    subgraph "Language Models"
        direction TB
        NLP["NLP Basics\n(tokenization, embeddings)"]
        ATTN["Attention Mechanism\n(QKV, GQA, RoPE)"]
        TRANSFORM["Transformer\n(GPT architecture)"]
    end

    subgraph "GenAI Applications"
        direction TB
        LLM["LLM APIs\n(OpenAI, Anthropic)"]
        RAG["RAG Systems"]
        AGENTS["AI Agents\n(LangGraph, MCP)"]
    end

    subgraph "Production"
        direction TB
        FINETUNE["Fine-Tuning\n(LoRA, QLoRA)"]
        RLHF["RLHF / DPO"]
        INFER["Inference Optimization\n(vLLM, quantization)"]
        DEPLOY["Production Deployment\n(FastAPI, MLOps)"]
    end

    NUMPY --> NN
    LINALG --> ATTN
    CALCULUS --> NN
    PROB --> CLASSML
    CLASSML --> EVAL
    NN --> PYTORCH
    PYTORCH --> TRANSFORM
    NLP --> ATTN
    ATTN --> TRANSFORM
    TRANSFORM --> LLM
    LLM --> RAG
    LLM --> AGENTS
    RAG --> AGENTS
    TRANSFORM --> FINETUNE
    FINETUNE --> RLHF
    PYTORCH --> INFER
    FINETUNE --> INFER
    AGENTS --> DEPLOY
    RAG --> DEPLOY
    INFER --> DEPLOY
```

---

## Alternative Learning Paths

### Path 1: LLM App Engineer (Fastest — ~9 months)

```mermaid
flowchart LR
    A["Python\n(Phase 0)\n4 wks"] --> B["Math Essentials\n(Phase 1 Part A)\n4 wks"]
    B --> C["ML Overview\n(Phase 2 light)\n2 wks"]
    C --> D["Transformers\n(Phase 4b conceptual)\n3 wks"]
    D --> E["LLMs + Prompting\n(Phase 5a)\n4 wks"]
    E --> F["RAG Systems\n(Phase 5b)\n6 wks"]
    F --> G["Agents\n(Phase 6)\n6 wks"]
    G --> H["Production AI\n(Phase 9)\n8 wks"]

    style A fill:#4CAF50,color:#fff
    style H fill:#2196F3,color:#fff
```

**Skip**: Phase 3 (PyTorch deep dive), Phase 7 (fine-tuning), Phase 8 (advanced)
**Best for**: Engineers who want to build LLM applications quickly

---

### Path 2: MLOps / LLMOps Specialist (~12 months)

```mermaid
flowchart LR
    A["Python + CS\n(Phase 0)"] --> B["Math (Phase 1)"]
    B --> C["Classical ML\n(Phase 2)"]
    C --> D["PyTorch\n(Phase 3b)"]
    D --> E["LLMs + RAG\n(Phase 5a+5b)"]
    E --> F["Production AI\n(Phase 9)"]
    F --> G["Advanced Topics\n(Phase 8 partial)"]

    style F fill:#FF5722,color:#fff
```

**Focus on**: MLflow, W&B, LangSmith, CI/CD evaluation, Databricks AI stack, Docker
**Skip**: Phase 7b (RLHF), Phase 10 (research), Phase 6 (deep agent work)

---

### Path 3: AI Agent Specialist (~12 months)

```mermaid
flowchart LR
    A["Python + CS\n(Phase 0)"] --> B["Math (Phase 1)"]
    B --> C["ML Fundamentals\n(Phase 2)"]
    C --> D["PyTorch\n(Phase 3b)"]
    D --> E["Transformers\n(Phase 4b)"]
    E --> F["LLMs + Prompting\n(Phase 5a)"]
    F --> G["RAG\n(Phase 5b)"]
    G --> H["Agents + MCP\n(Phase 6) ⭐"]
    H --> I["Production AI\n(Phase 9)"]

    style H fill:#9C27B0,color:#fff
    style I fill:#2196F3,color:#fff
```

**Focus on**: LangGraph, multi-agent systems, MCP protocol, function calling, tool design
**Projects to prioritize**: 16, 17, 17b, 22

---

### Path 4: Fine-Tuning / Alignment Specialist (~14 months)

```mermaid
flowchart LR
    A["Python + CS"] --> B["Full Math\n(Phase 1)"]
    B --> C["ML + DL\n(Phase 2+3)"]
    C --> D["Transformers\n(Phase 4b)"]
    D --> E["LLMs\n(Phase 5a)"]
    E --> F["Fine-Tuning LoRA\n(Phase 7a) ⭐"]
    F --> G["RLHF + DPO\n(Phase 7b) ⭐"]
    G --> H["Inference Optimization\n(Phase 8 partial)"]
    H --> I["Production AI\n(Phase 9)"]

    style F fill:#E91E63,color:#fff
    style G fill:#E91E63,color:#fff
```

**Focus on**: QLoRA, DPO, GPU fundamentals, vLLM, PEFT
**Projects to prioritize**: 9, 18, 19, 21

---

### Path 5: Research Path (~18+ months, full curriculum)

```mermaid
flowchart TD
    A["Full Phase 0–5"] --> B["Agents (Phase 6)"]
    A --> C["Fine-Tuning (Phase 7a+7b)"]
    B --> D["Advanced Topics\n(Phase 8)"]
    C --> D
    D --> E["Production AI\n(Phase 9)"]
    E --> F["Research + OSS\n(Phase 10) ⭐"]
    F --> G["Paper Implementation"]
    F --> H["Open Source Contribution"]
    F --> I["Own Research"]

    style F fill:#FF9800,color:#fff
```

---

## Topic Dependency Detail

### What blocks what (critical dependencies)

| If you skip... | You cannot do... |
|----------------|-----------------|
| Linear algebra (Phase 1) | Understand attention QKV projections |
| Calculus + chain rule (Phase 1) | Understand backpropagation derivation |
| Backpropagation (Phase 3a) | Understand why training fails |
| PyTorch autograd (Phase 3b) | Fine-tune models (Phase 7) |
| Transformer architecture (Phase 4b) | Understand KV cache, GQA, context window tricks |
| RAG fundamentals (Phase 5b) | Build production agents (Phase 6, 9) |
| LoRA math (Phase 7a) | Do RLHF/DPO (Phase 7b) |

### Safe shortcuts (optional topics per path)

| Topic | Required for | Optional if |
|-------|-------------|-------------|
| CNNs (Phase 3b) | Computer vision only | Not doing multimodal |
| Classic NLP (TF-IDF, Word2Vec) (Phase 4a) | Understanding history | You only care about transformers |
| Mixture of Experts (Phase 8) | Research track | Just building apps |
| Speculative decoding (Phase 8) | Inference optimization | Not doing Phase 21 project |
| Paper reading (Phase 10) | Research track | Just building products |

---

## Phase Unlock Gates

You must pass each gate before proceeding (self-assessment quiz ≥ 22/25):

```
Phase 0 → Gate 0: Can implement matrix multiply, write async code, explain Git internals
Phase 1 → Gate 1: Can derive backprop, compute eigendecomposition, explain KL divergence
Phase 2 → Gate 2: Can implement XGBoost pipeline, explain bias-variance, do feature selection
Phase 3 → Gate 3: Can implement transformer attention from scratch in NumPy
Phase 4 → Gate 4: Can build GPT-2 that trains on toy text
Phase 5 → Gate 5: Can build RAG system with evaluation metrics
Phase 6 → Gate 6: Can build multi-agent system with persistent memory
Phase 7 → Gate 7: Can fine-tune LLM with LoRA and measure improvement
Phase 8 → Gate 8: Can run vLLM with quantized model and benchmark throughput
Phase 9 → Gate 9: Can deploy production API with streaming, caching, monitoring
```

---

*[← Index](./INDEX.md) | [Study Schedule](./13_Study_Schedule_18_Months.md)*
