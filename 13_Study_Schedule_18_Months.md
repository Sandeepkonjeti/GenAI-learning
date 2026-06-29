# 18-Month Study Schedule
**Complete week-by-week learning plan | Starting from Data Engineer background**

> **Back to Roadmap**: [README](./00_README.md) | **All Projects**: [Projects](./12_Projects_Beginner_to_Expert.md)

---

## Table of Contents

- [How to Use This Schedule](#how-to-use-this-schedule)
- [Month-by-Month Milestone Overview](#month-by-month-milestone-overview)
- [Phase 0: Weeks 1–8 (CS Fundamentals & Python)](#phase-0-weeks-18)
- [Phase 1: Weeks 9–16 (Mathematics for ML)](#phase-1-weeks-916)
- [Phase 2: Weeks 17–24 (Machine Learning)](#phase-2-weeks-1724)
- [Phase 3: Weeks 25–36 (Deep Learning)](#phase-3-weeks-2536)
- [Phase 4: Weeks 37–44 (NLP & Transformers)](#phase-4-weeks-3744)
- [Phase 5: Weeks 45–52 (LLMs, RAG & Embeddings)](#phase-5-weeks-4552)
- [Phase 6: Weeks 53–56 (Agents & MCP)](#phase-6-weeks-5356)
- [Phase 7: Weeks 57–64 (Fine-Tuning & Alignment)](#phase-7-weeks-5764)
- [Phase 8: Weeks 65–68 (Advanced Topics)](#phase-8-weeks-6568)
- [Phase 9: Weeks 69–72 (Production Engineering)](#phase-9-weeks-6972)
- [Month 18+: Research & Open Source](#month-18-research--open-source)
- [Revision Weeks Schedule](#revision-weeks-schedule)
- [Interview Preparation Milestones](#interview-preparation-milestones)
- [Paper Reading Milestones](#paper-reading-milestones)
- [The Daily Habit](#the-daily-habit)

---

## How to Use This Schedule

**Daily commitment**: 1–2 hours (1 hr minimum for progress, 2 hrs for accelerated track)

**Key principles**:
1. **Depth over breadth**: understand each concept before moving on
2. **Build, don't just read**: every concept has a project attached
3. **Revision weeks**: scheduled review prevents forgetting
4. **Flexible pacing**: some weeks are harder — adjust, don't skip
5. **The checkpoint**: before moving to the next phase, complete the mastery checklist

**If you fall behind**: Don't skip — compress. Reduce project scope, but keep the conceptual learning.

**If you're ahead**: Go deeper on projects, start optional extensions, read additional papers.

---

## Month-by-Month Milestone Overview

| Month | Phase | Key Achievement | Project |
|-------|-------|----------------|---------|
| 1 | Phase 0 | WSL2 + Python + NumPy mastery | Project 1 & 2 |
| 2 | Phase 0–1 | Linear algebra from scratch | Project 3 |
| 3 | Phase 1 | Full math toolkit (calculus, prob, stats) | — |
| 4 | Phase 2 | ML fundamentals, scikit-learn | Project 4 & 5 |
| 5 | Phase 2 | XGBoost, advanced ML techniques | Project 5 (complete) |
| 6 | Phase 3a | Backpropagation from scratch | Project 6 |
| 7 | Phase 3b | PyTorch training loop mastery | Project 7 |
| 8 | Phase 3b | CNN, RNN, training best practices | — |
| 9 | Phase 4a | NLP fundamentals, tokenization | Project 8 |
| 10 | Phase 4b | Transformers from scratch (**critical month**) | Project 9 |
| 11 | Phase 4b | GPT implemented and running | Project 9 (complete), 10 |
| 12 | Phase 5a | LLM APIs, prompt engineering | Project 11, 12, 13 |
| 13 | Phase 5b | RAG system built and evaluated | Project 14, 15 |
| 14 | Phase 6 | Agents, tools, MCP server | Project 16, 17 |
| 15 | Phase 7a | Fine-tuned your first LLM | Project 18 |
| 16 | Phase 7b | DPO alignment complete | Project 19 |
| 17 | Phase 8–9 | Inference optimization + Production | Project 21, 22 |
| 18 | Phase 9–10 | Portfolio complete, contributions | Project 22, 28 |

---

## Phase 0: Weeks 1–8

**File**: [01_Phase0_CS_Fundamentals_and_Python.md](./01_Phase0_CS_Fundamentals_and_Python.md)

### Week 1: Environment Setup + NumPy Foundations
- [ ] Install WSL2 and configure Python environment
- [ ] Complete Project 1 (environment setup)
- [ ] NumPy: array creation, shapes, indexing
- [ ] Read: NumPy broadcasting documentation

### Week 2: NumPy Deep Dive
- [ ] NumPy: vectorized operations, views vs. copies, memory layout
- [ ] Implement matrix operations from scratch (both Python loops and NumPy)
- [ ] Benchmark and visualize speedup differences
- [ ] Complete Project 2 (NumPy library)

### Week 3: Python for AI
- [ ] Dataclasses, type hints, context managers
- [ ] Generators and iterators (critical for data loading)
- [ ] Decorators (used in PyTorch hooks, LangChain tools)
- [ ] Write 3 utility functions using each concept

### Week 4: Async Python
- [ ] asyncio basics: async/await, gather
- [ ] Implement async LLM streaming (using OpenAI API)
- [ ] Review Phase 0 mastery checklist
- [ ] Start linear regression project (Project 3 setup)

### Weeks 5–6: Docker + Git
- [ ] Docker basics: build, run, volumes, compose
- [ ] Containerize a Python application
- [ ] Git advanced: branching, rebasing, conventional commits
- [ ] GitHub Actions: basic CI pipeline

### Weeks 7–8: Project 3 + Review
- [ ] Complete Project 3 (linear regression from scratch)
- [ ] Phase 0 revision: review all notebooks
- [ ] **Checkpoint**: Can you implement dot product, matrix multiply, and softmax from scratch?
- [ ] Watch: 3Blue1Brown "Essence of Linear Algebra" (review)

---

## Phase 1: Weeks 9–16

**File**: [02_Phase1_Mathematics_for_ML.md](./02_Phase1_Mathematics_for_ML.md)

### Weeks 9–10: Linear Algebra for ML
- [ ] Vectors, matrices, linear combinations (implement from scratch)
- [ ] Eigenvalues and eigenvectors (understand PCA connection)
- [ ] SVD: implement and connect to data compression
- [ ] Build mental model: SVD → PCA → LoRA connection

### Weeks 11–12: Calculus for ML
- [ ] Derivatives: rules, partial derivatives, chain rule
- [ ] Chain rule = backpropagation: trace gradient through a 2-layer network by hand
- [ ] Implement gradient descent from scratch with Adam optimizer
- [ ] Verify against PyTorch autograd

### Weeks 13–14: Probability for ML
- [ ] Probability distributions: Gaussian, Bernoulli, Categorical
- [ ] MLE: derive cross-entropy loss from first principles
- [ ] KL divergence: understand connection to information theory and RLHF
- [ ] Implement: likelihood estimation, sample from distributions

### Weeks 15–16: Statistics + Review
- [ ] Hypothesis testing, confidence intervals
- [ ] Bayesian updating (Bayes' theorem for ML)
- [ ] **Phase 1 revision week**: go back to hardest concepts
- [ ] **Checkpoint**: Can you derive backpropagation from first principles?

---

## Phase 2: Weeks 17–24

**File**: [03_Phase2_Machine_Learning.md](./03_Phase2_Machine_Learning.md)

### Weeks 17–18: Linear and Logistic Regression
- [ ] Implement linear regression with gradient descent from scratch
- [ ] Implement logistic regression with cross-entropy loss
- [ ] Add L1/L2 regularization to both
- [ ] Visualize: decision boundaries, regularization paths

### Weeks 19–20: Tree Methods
- [ ] Implement decision tree with information gain from scratch
- [ ] Random forests: understand variance reduction (bagging)
- [ ] XGBoost: understand gradient boosting; apply with SHAP
- [ ] Complete Project 5 (ML pipeline)

### Weeks 21–22: Unsupervised + Evaluation
- [ ] K-means from scratch
- [ ] PCA: implement and visualize 2D projection
- [ ] Bias-variance tradeoff: experiments with polynomial regression
- [ ] Evaluation: ROC, AUC, precision-recall tradeoffs

### Weeks 23–24: Project 5 + Review
- [ ] Complete Project 5 (end-to-end ML pipeline)
- [ ] **Phase 2 revision week**
- [ ] **Checkpoint**: Can you choose and evaluate the right model for a classification task?
- [ ] Read: "The Elements of Statistical Learning" (selected chapters)

---

## Phase 3: Weeks 25–36

**Files**: [04_Phase3_Part1_Deep_Learning_Theory.md](./04_Phase3_Part1_Deep_Learning_Theory.md), [04_Phase3_Part2_PyTorch.md](./04_Phase3_Part2_PyTorch.md)

### Weeks 25–26: Neural Network Theory
- [ ] Implement DenseLayer with forward/backward from scratch
- [ ] Universal Approximation Theorem: intuition
- [ ] All activation functions: implement and compare
- [ ] Chain rule through a 2-layer network: manual backprop

### Weeks 27–28: Training Deep Networks
- [ ] Implement SGD, Momentum, Adam from scratch
- [ ] Implement cosine annealing scheduler
- [ ] BatchNorm and LayerNorm: implement and compare
- [ ] Complete Project 6 (Backpropagation visualizer)

### Weeks 29–30: PyTorch Fundamentals
- [ ] Tensors: all operations, indexing, broadcasting
- [ ] Autograd: requires_grad, backward, gradient inspection
- [ ] nn.Module: build your own linear layer
- [ ] Dataset/DataLoader: create a custom dataset

### Weeks 31–32: Training Loop Mastery
- [ ] Implement canonical 7-step training loop
- [ ] Mixed precision (autocast + GradScaler)
- [ ] Model saving and loading (state_dict)
- [ ] Complete Project 7 (MNIST CNN, >99% accuracy)

### Weeks 33–34: CNNs and RNNs
- [ ] Convolution: understand operation, padding, stride
- [ ] ResNet skip connections: implement and understand gradient flow
- [ ] LSTM: understand gate mechanism, implement from scratch
- [ ] Vanishing gradient: demonstrate and visualize

### Weeks 35–36: torch.compile + Review
- [ ] torch.compile: benchmark speedup on your models
- [ ] **Phase 3 revision week**: revisit backprop and PyTorch
- [ ] **Checkpoint**: Can you implement a complete training loop for any task?
- [ ] Watch: Karpathy "micrograd from scratch" video

---

## Phase 4: Weeks 37–44

**Files**: [05_Phase4_Part1_NLP_Fundamentals.md](./05_Phase4_Part1_NLP_Fundamentals.md), [05_Phase4_Part2_Transformers.md](./05_Phase4_Part2_Transformers.md)

### Weeks 37–38: NLP Fundamentals
- [ ] Text preprocessing pipeline: tokenization, normalization, stemming
- [ ] TF-IDF from scratch: implement and build search engine
- [ ] Word2Vec: train with gensim; word arithmetic experiments
- [ ] Complete Project 8 (sentiment analysis comparison)

### Weeks 39–40: Attention Mechanism
- [ ] Read: "Attention Is All You Need" paper (Pass 1 and 2)
- [ ] Read: Jay Alammar "The Illustrated Transformer"
- [ ] Implement scaled dot-product attention
- [ ] Understand Q/K/V intuition deeply

### Weeks 41–42: Transformers from Scratch (CRITICAL)
- [ ] Watch: Karpathy "Let's build GPT from scratch" (full 2 hours)
- [ ] Implement alongside the video: do not copy, type every line
- [ ] Multi-head attention, transformer block, full GPT
- [ ] Train on Shakespeare dataset

### Weeks 43–44: GPT Project + Review
- [ ] Complete Project 9 (GPT from scratch, generating text)
- [ ] Complete Project 10 (attention visualization)
- [ ] **Phase 4 revision week**: re-read paper, re-watch key video sections
- [ ] **Checkpoint**: Can you implement transformers from scratch without reference?

---

## Phase 5: Weeks 45–52

**Files**: [06_Phase5_Part1_LLMs_and_Prompt_Engineering.md](./06_Phase5_Part1_LLMs_and_Prompt_Engineering.md), [06_Phase5_Part2_Embeddings_VectorDB_RAG.md](./06_Phase5_Part2_Embeddings_VectorDB_RAG.md)

### Weeks 45–46: LLM Theory
- [ ] Watch: Karpathy "State of GPT" (2023)
- [ ] Read: GPT-3 paper (Brown et al., 2020)
- [ ] Read: Chinchilla paper (key sections)
- [ ] Understand: in-context learning, emergent capabilities, hallucination

### Weeks 47–48: Prompt Engineering
- [ ] Complete DeepLearning.AI Prompt Engineering course (free, 2 hours)
- [ ] Implement: zero-shot, few-shot, CoT, self-consistency
- [ ] Implement: structured output with Pydantic
- [ ] Complete Projects 11, 12, 13

### Weeks 49–50: Embeddings + Vector Databases
- [ ] Embedding models: test all-MiniLM-L6-v2 and BAAI/bge-large-en-v1.5
- [ ] ChromaDB: implement CRUD with metadata filtering
- [ ] HNSW: read the paper summary; understand ANN tradeoffs
- [ ] Start Project 14 (Document QA)

### Weeks 51–52: RAG + Review
- [ ] Complete Project 14 (basic RAG system)
- [ ] Complete Project 15 (RAG evaluation benchmark)
- [ ] Implement: hybrid BM25 + dense search, CrossEncoder reranker
- [ ] **Checkpoint**: Can you build a RAG system with RAGAS evaluation?

---

## Phase 6: Weeks 53–56

**File**: [07_Phase6_AI_Agents_and_MCP.md](./07_Phase6_AI_Agents_and_MCP.md)

### Weeks 53–54: Agent Fundamentals
- [ ] Read: ReAct paper (Yao et al., 2022)
- [ ] Implement ReAct agent from scratch (no LangChain)
- [ ] OpenAI function calling: build tool-use agent
- [ ] Start Project 16 (Research Agent)

### Weeks 55–56: LangGraph + MCP
- [ ] LangGraph quickstart: build simple stateful agent
- [ ] Add persistence/checkpointing
- [ ] Read: MCP documentation (Anthropic)
- [ ] Complete Project 16 (Research Agent)
- [ ] Complete Project 17 (Databricks MCP Server)
- [ ] **Checkpoint**: MCP server connected to Claude Desktop and working

---

## Phase 7: Weeks 57–64

**Files**: [08_Phase7_Part1_FineTuning_and_LoRA.md](./08_Phase7_Part1_FineTuning_and_LoRA.md), [08_Phase7_Part2_RLHF_and_DPO.md](./08_Phase7_Part2_RLHF_and_DPO.md)

### Weeks 57–58: LoRA Theory + Dataset
- [ ] Read: LoRA paper (Hu et al., 2021)
- [ ] Implement LoRALinear from scratch
- [ ] Understand: r, alpha, target_modules hyperparameters
- [ ] Collect and format dataset for fine-tuning (Project 18 prep)

### Weeks 59–60: QLoRA Fine-Tuning
- [ ] Read: QLoRA paper (Dettmers et al., 2023)
- [ ] Set up Colab GPU environment with Unsloth
- [ ] Run first fine-tuning job
- [ ] Monitor with W&B: training loss, validation loss, generations
- [ ] Complete Project 18 (Domain Expert Fine-Tune)

### Weeks 61–62: RLHF + DPO Theory
- [ ] Read: InstructGPT paper (Ouyang et al., 2022)
- [ ] Read: DPO paper (Rafailov et al., 2023)
- [ ] Understand reward model training (implement Bradley-Terry loss)
- [ ] Understand KL divergence penalty

### Weeks 63–64: DPO Training
- [ ] Build preference dataset from Project 18 model outputs
- [ ] Run DPO training with TRL
- [ ] Compare SFT vs. DPO using LLM-as-judge
- [ ] Complete Project 19 (DPO fine-tune)
- [ ] **Checkpoint**: Fine-tuned model on HuggingFace Hub

---

## Phase 8: Weeks 65–68

**File**: [09_Phase8_Advanced_Topics.md](./09_Phase8_Advanced_Topics.md)

### Weeks 65–66: Multimodal + Diffusion
- [ ] Read: ViT paper (Dosovitskiy et al.)
- [ ] Build CLIP zero-shot classifier
- [ ] Use LLaVA for image QA
- [ ] Generate images with Stable Diffusion
- [ ] Understand diffusion process (DDPM paper summary)

### Weeks 67–68: GPU Optimization + Inference
- [ ] Read: FlashAttention paper
- [ ] Profile a model with PyTorch profiler
- [ ] Quantize a model with AWQ
- [ ] Deploy with vLLM
- [ ] Complete Project 21 (Inference Server)
- [ ] **Checkpoint**: vLLM server running with 3-5x throughput vs. baseline

---

## Phase 9: Weeks 69–72

**File**: [10_Phase9_Production_AI_Engineering.md](./10_Phase9_Production_AI_Engineering.md)

### Weeks 69–70: MLOps + LLMOps
- [ ] Set up W&B experiment tracking
- [ ] Implement prompt versioning system
- [ ] Set up LangSmith tracing
- [ ] Implement LLM-as-judge evaluation pipeline

### Weeks 71–72: Production System
- [ ] Complete Project 22 (Production RAG with full observability)
- [ ] Security: implement all relevant OWASP LLM mitigations
- [ ] Deploy: Docker Compose + Grafana monitoring
- [ ] **Final Checkpoint**: Production RAG with monitoring, evaluation, and security

---

## Month 18+: Research & Open Source

**File**: [11_Phase10_Research_and_OpenSource.md](./11_Phase10_Research_and_OpenSource.md)

### Ongoing habits:
- [ ] Daily: Hugging Face Daily Papers (15 min)
- [ ] Weekly: 1 paper (Pass 1 on all, Pass 2 on best)
- [ ] Monthly: 1 implementation, 1 blog post
- [ ] 3 months: 1 open source contribution

### Tier 1 papers to read (in order):
- [ ] Attention Is All You Need
- [ ] GPT-3 (Language Models are Few-Shot Learners)
- [ ] InstructGPT
- [ ] LoRA
- [ ] RAG (Lewis et al.)

---

## Revision Weeks Schedule

Scheduled revision prevents forgetting. These weeks are NOT for new content.

| Week | Revise |
|------|--------|
| 8 | Phase 0 (Python, NumPy, Docker) |
| 16 | Phase 1 (Math: linear algebra, calculus, probability) |
| 24 | Phase 2 (ML: regression, trees, evaluation) |
| 36 | Phase 3 (Deep learning, PyTorch) |
| 44 | Phase 4 (NLP, transformers) |
| 52 | Phase 5 (LLMs, RAG, embeddings) |
| 64 | Phase 6+7 (Agents, fine-tuning) |
| 72 | All phases (final comprehensive review) |

**What to do in revision weeks**:
1. Re-read your notes from that phase
2. Re-run your project and explain each component aloud
3. Complete the mastery checklist for that phase
4. Try to implement one concept from memory

---

## Interview Preparation Milestones

If your goal is an AI Engineer role, use these as preparation guides:

| Target Date | Interview Topic | What to Prepare |
|-------------|----------------|-----------------|
| Month 6 | ML Fundamentals | Decision trees, bias-variance, regularization, evaluation metrics |
| Month 11 | Deep Learning | Backprop, CNNs, RNNs, attention, transformer architecture |
| Month 13 | LLMs | Scaling laws, ICL, hallucination, RAG system design |
| Month 15 | Systems Design | Design a production RAG system; discuss tradeoffs |
| Month 16 | Fine-Tuning | LoRA math, QLoRA setup, when to fine-tune vs. RAG |
| Month 18 | Production AI | MLOps, LLMOps, security, evaluation, monitoring |

**Mock interview topics by phase:**

```
Technical coding:
- Implement attention from scratch (Phase 4 checkpoint)
- Build RAG pipeline with LangChain (Phase 5)
- Fix a broken training loop (Phase 3)
- Debug a slow Spark job using your existing expertise

System design:
- "Design a customer support AI that uses your company's documentation"
  → RAG system with vector DB, guardrails, monitoring
- "We want to fine-tune a model for our domain. How do you approach this?"
  → Decision framework: when fine-tune vs RAG vs prompt engineering
- "How would you build an AI coding assistant for internal use?"
  → Agent with code execution, RAG on codebase, MCP for IDE integration

Behavioral (your narrative):
- "Tell me about a production AI system you built"
  → Project 22 (Production RAG) — use this as your story
- "Describe a time you improved ML model performance"
  → Project 15 (RAG evaluation benchmark) — systematic improvement
```

---

## Paper Reading Milestones

| Month | Paper | Reading Depth |
|-------|-------|--------------|
| 4 | Attention Is All You Need | Pass 1 (first read) |
| 10 | Attention Is All You Need | Pass 2 (after building transformers) |
| 10 | Jay Alammar "The Illustrated Transformer" | Full read |
| 12 | GPT-3 (Brown et al., 2020) | Pass 1 |
| 13 | RAG (Lewis et al., 2020) | Pass 1 |
| 14 | ReAct (Yao et al., 2022) | Pass 1 |
| 15 | LoRA (Hu et al., 2021) | Pass 2 (after implementing) |
| 15 | QLoRA (Dettmers et al., 2023) | Pass 2 |
| 16 | InstructGPT (Ouyang et al., 2022) | Pass 2 |
| 16 | DPO (Rafailov et al., 2023) | Pass 2 |
| 17 | FlashAttention (Dao et al., 2022) | Pass 1 |
| 18 | Chinchilla (Hoffmann et al., 2022) | Pass 1 |

---

## The Daily Habit

The most important thing in this schedule is consistency. Here's what 1 hour/day looks like:

```
Day 1–3: Learn concept
  - 30 min: Read the relevant section from the phase file
  - 30 min: Try the code in a notebook

Day 4–6: Build
  - 60 min: Implement the week's project task

Day 7: Review
  - 30 min: Review notes from the week
  - 15 min: Revisit any concept that felt unclear
  - 15 min: Plan next week

Non-negotiable minimum (when time is short):
  - Even on a 20-minute day: open your project and run it
  - The goal: never go more than 2 days without touching the material
```

---

*Study Schedule | Part of the [GenAI Engineer Roadmap](./00_README.md)*
