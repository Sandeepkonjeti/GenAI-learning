# All Projects: Beginner to Expert
**Complete portfolio of 30+ projects across all phases**

> **Back to Roadmap**: [README](./00_README.md)

---

## Table of Contents

- [How to Use This File](#how-to-use-this-file)
- [Beginner Projects (Months 1–6)](#beginner-projects-months-16)
- [Intermediate Projects (Months 7–11)](#intermediate-projects-months-711)
- [Advanced Projects (Months 12–15)](#advanced-projects-months-1215)
- [Expert / Portfolio Projects (Months 15–18+)](#expert--portfolio-projects-months-1518)
- [Project Selection by Goal](#project-selection-by-goal)
- [GitHub Portfolio Guide](#github-portfolio-guide)

---

## How to Use This File

Each project has:
- **Phase**: which roadmap phase it belongs to
- **Difficulty**: 1–10 scale
- **Time**: realistic estimate for 1–2 hrs/day
- **Skills**: what you'll build and learn
- **Tech**: specific libraries/tools
- **Output**: what you'll have when done
- **Extension**: how to take it further

---

## Beginner Projects (Months 1–6)

### Project 1: Python AI Toolkit Setup
**Phase**: 0 | **Difficulty**: 1/10 | **Time**: 1 day

**Goal**: Create a fully configured Python environment for AI development.

**Tasks**:
- Set up WSL2 on Windows
- Install Python 3.11, conda/pyenv, VS Code with extensions
- Install NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, Transformers
- Write a test script that imports and runs each library
- Configure Jupyter with the environment

**Output**: Working AI development environment; `setup_test.py` that passes all imports

**Tech**: WSL2, conda, Python 3.11, requirements.txt

---

### Project 2: NumPy Matrix Operations Library
**Phase**: 0 | **Difficulty**: 2/10 | **Time**: 3 days

**Goal**: Deep understanding of vectorized NumPy operations.

**Tasks**:
- Implement dot product, matrix multiply, transpose from scratch (loops first, then NumPy)
- Benchmark: Python loops vs. NumPy vs. PyTorch (same operation)
- Implement batch normalization using only NumPy
- Implement softmax function (scalar, vector, and batch versions)
- Write a Jupyter notebook comparing implementations with timing charts

**Output**: Notebook showing 100-1000x speedup from vectorization

**Tech**: NumPy, Matplotlib, time module

---

### Project 3: Linear Regression from Scratch
**Phase**: 1–2 | **Difficulty**: 3/10 | **Time**: 1 week

**Goal**: Understand gradient descent at the implementation level.

**Tasks**:
- Implement linear regression with gradient descent (no sklearn)
- Add L1 and L2 regularization
- Visualize the loss landscape and gradient steps
- Compare with sklearn's LinearRegression
- Apply to a real dataset (Boston Housing or California Housing)
- Plot: learning curves, residuals, coefficient visualization

**Output**: Notebook with full implementation and analysis

**Tech**: NumPy, Matplotlib, sklearn (for comparison only)

**Extension**: Add momentum and Adam optimizer; compare convergence

---

### Project 4: Neural Network from Scratch (No PyTorch)
**Phase**: 1–3 | **Difficulty**: 5/10 | **Time**: 2 weeks

**Goal**: Build backpropagation intuition that frameworks hide.

**Tasks**:
- Implement DenseLayer class with forward() and backward()
- Implement activation functions: Sigmoid, ReLU, Tanh
- Implement cross-entropy loss with gradient
- Chain layers into a neural network
- Train on MNIST (flattened images, 784 input)
- Achieve >95% test accuracy
- Visualize: learned weights as images, loss curves, confusion matrix

**Output**: Working neural network library without any autograd framework

**Tech**: NumPy only, Matplotlib

**Extension**: Add dropout, batch normalization, compare training dynamics

---

### Project 5: Machine Learning Pipeline
**Phase**: 2 | **Difficulty**: 4/10 | **Time**: 1.5 weeks

**Goal**: End-to-end ML pipeline on a real dataset.

**Tasks**:
- Choose dataset: Titanic, Census Income, or Credit Default
- EDA: distribution plots, correlation heatmap, missing value analysis
- Feature engineering: encode categoricals, handle nulls, scale numerics
- Build sklearn Pipeline with preprocessing + model
- Try: LogReg, RandomForest, XGBoost
- Cross-validation with proper splitting
- SHAP feature importance
- Confusion matrix, precision/recall, ROC curve

**Output**: Jupyter notebook + Python module with reusable pipeline code

**Tech**: Pandas, sklearn, XGBoost, SHAP, Matplotlib/Seaborn

**Extension**: Add model drift simulation; AutoML comparison with FLAML

---

### Project 6: Backpropagation Visualizer
**Phase**: 3a | **Difficulty**: 4/10 | **Time**: 1 week

**Goal**: Deeply understand automatic differentiation.

**Tasks**:
- Follow Karpathy's micrograd (build your own autograd engine)
- Implement Value class with + - * / ** operations and .backward()
- Build a simple MLP using your Value class
- Train on XOR problem
- Visualize: computation graph with gradient values at each node

**Output**: Working autograd engine + visualization

**Tech**: Python, Graphviz (for visualization)

**Reference**: [micrograd by Karpathy](https://github.com/karpathy/micrograd)

---

## Intermediate Projects (Months 7–11)

### Project 7: PyTorch Image Classifier (MNIST/CIFAR)
**Phase**: 3b | **Difficulty**: 5/10 | **Time**: 1 week

**Goal**: Master the PyTorch training loop pattern.

**Tasks**:
- Build CNN with Conv2d, BatchNorm, ReLU, MaxPool, Dropout
- Implement the canonical training loop (6 steps)
- Add: learning rate scheduler, early stopping, gradient clipping
- Track: loss, accuracy per epoch
- Visualize: training curves in TensorBoard or W&B
- Run inference on test images and display predictions

**Output**: Trained model (>99% MNIST, >90% CIFAR-10) + training curves

**Tech**: PyTorch, torchvision, W&B or TensorBoard

---

### Project 8: Sentiment Analysis with Classical NLP
**Phase**: 4a | **Difficulty**: 4/10 | **Time**: 1 week

**Goal**: Understand classical NLP representations before transformers.

**Tasks**:
- Implement TF-IDF vectorizer from scratch
- Train: Logistic Regression, Naive Bayes, SVM on IMDb reviews
- Implement Word2Vec embeddings (use gensim)
- Train: LSTM sentiment classifier using pre-trained Word2Vec
- Compare: TF-IDF vs. Word2Vec vs. transformer embeddings (all-MiniLM)
- Error analysis: what types of examples does each model fail on?

**Output**: Comparison table + error analysis notebook

**Tech**: gensim, sklearn, PyTorch (LSTM), sentence-transformers

---

### Project 9: GPT from Scratch
**Phase**: 4b | **Difficulty**: 7/10 | **Time**: 2–3 weeks

**Goal**: Implement and train a GPT model — the most important project in the roadmap.

**Tasks**:
- Follow Karpathy's "Let's build GPT from scratch" video exactly
- Implement: ScaledDotProductAttention, MultiHeadAttention, TransformerBlock, GPT
- Add: causal masking, positional encoding, weight tying
- Train character-level model on Shakespeare (~1MB)
- Generate text samples at multiple temperatures
- Visualize: attention weights per head per layer
- Extend: add KV cache; measure speedup during generation

**Output**: Working GPT with ~10M parameters that generates coherent English text

**Tech**: PyTorch from scratch (no transformers library)

**Reference**: [nanoGPT by Karpathy](https://github.com/karpathy/nanoGPT)

---

### Project 10: Attention Visualization Dashboard
**Phase**: 4b | **Difficulty**: 5/10 | **Time**: 1 week

**Goal**: Build intuition for what attention heads learn.

**Tasks**:
- Load BERT from HuggingFace
- Extract attention weights for sample sentences
- Build interactive visualization (Streamlit or Gradio app)
- Users can: enter text, select layer and head, see attention heatmap
- Annotate what you observe: which heads attend to syntax? semantics?

**Output**: Deployed Streamlit/Gradio app

**Tech**: HuggingFace Transformers, Streamlit or Gradio, Seaborn

---

### Project 11: LLM Code Documentation Generator
**Phase**: 5a | **Difficulty**: 4/10 | **Time**: 1 week

**Goal**: Build practical LLM application with structured outputs.

**Tasks**:
- Parse Python files using the `ast` module to extract functions and classes
- Generate docstrings using an LLM (GPT-4o-mini or Claude Haiku)
- Force consistent format: NumPy-style docstrings with Args, Returns, Examples
- Write updated files with injected docstrings
- Test on this codebase! (notebook/ directory)
- Measure: how often does it generate wrong/hallucinated types?

**Output**: CLI tool: `python docgen.py my_file.py`

**Tech**: OpenAI or Anthropic API, Python ast module, Pydantic (structured output)

---

### Project 12: Multi-Model Benchmark
**Phase**: 5a | **Difficulty**: 5/10 | **Time**: 1 week

**Goal**: Learn to evaluate LLMs systematically.

**Tasks**:
- Define 50 test cases in your domain (Databricks/PySpark/data engineering)
- Send each to: GPT-4o-mini, Claude Haiku, LLaMA-3 (via Ollama)
- Measure: latency, token count, cost per query
- Evaluate quality: LLM-as-judge (GPT-4o as judge, 1-5 score)
- Build: comparison table and radar chart of model performance

**Output**: Benchmarking script + results report in Markdown

**Tech**: OpenAI, Anthropic, Ollama, Pandas, Matplotlib

---

### Project 13: CoT Math Problem Solver
**Phase**: 5a | **Difficulty**: 4/10 | **Time**: 3 days

**Goal**: Measure impact of chain-of-thought prompting.

**Tasks**:
- Load GSM8K dataset (grade school math problems)
- Implement: direct prompting, CoT prompting, self-consistency (5 samples)
- Evaluate accuracy on 200 problems for each method
- Statistical significance: bootstrap confidence intervals
- Write up: when does CoT help most? Failure case analysis.

**Output**: Results table + analysis notebook

**Tech**: datasets (HuggingFace), OpenAI API, scipy (stats)

---

### Project 14: Document QA System (Basic RAG)
**Phase**: 5b | **Difficulty**: 6/10 | **Time**: 2 weeks

**Goal**: Build a production-quality RAG system from scratch.

**Tasks**:
- Implement: document ingestion (PDF, TXT, MD), chunking, embedding, storage
- Vector store: ChromaDB with persistent storage
- Retrieval: semantic search + metadata filtering
- Generation: answer with citations, grounded in retrieved documents
- Evaluation: RAGAS (faithfulness, relevancy, precision, recall)
- Test on: Databricks documentation or your own documents

**Output**: CLI/Gradio app + RAGAS evaluation scores

**Tech**: LangChain, ChromaDB, sentence-transformers, RAGAS, Gradio

**Extension**: Add hybrid BM25 + dense search; add CrossEncoder reranking

---

## Advanced Projects (Months 12–15)

### Project 15: RAG Evaluation Benchmark
**Phase**: 5b | **Difficulty**: 7/10 | **Time**: 1 week

**Goal**: Build systematic RAG evaluation framework.

**Tasks**:
- Create 50 test QA pairs from your document corpus
- Compare 4 configurations: (basic chunks, semantic chunks) × (no reranking, with reranking)
- Run RAGAS on all configurations
- Visualize: metric comparison across configurations
- Conclusion: which config wins, and why?

**Output**: Evaluation framework + definitive configuration recommendation

**Tech**: RAGAS, pandas, matplotlib, sentence-transformers, CrossEncoder

---

### Project 16: Research Agent with LangGraph
**Phase**: 6 | **Difficulty**: 7/10 | **Time**: 2 weeks

**Goal**: Build a stateful multi-step agent.

**Tasks**:
- Build LangGraph agent with: search tool, calculator, web reader
- State: maintains research notes across steps
- Persistence: checkpoint with SQLite (can resume interrupted runs)
- Output: structured report with citations and confidence scores
- Test on: research questions in your domain

**Output**: Research agent + 5 sample research reports

**Tech**: LangGraph, Tavily Search API, OpenAI, SQLite checkpointing

---

### Project 17: Databricks MCP Server
**Phase**: 6 | **Difficulty**: 7/10 | **Time**: 1 week | **Unique to your background**

**Goal**: Expose your Databricks workspace to AI assistants via MCP.

**Tasks**:
- Implement MCP server with tools: list_tables, run_sql, get_job_status, list_clusters, list_notebooks
- Add resources: table schemas, recent job runs
- Add prompts: data_analysis_prompt, job_debugging_prompt
- Connect to Claude Desktop
- Demo: ask Claude natural language questions about your Databricks workspace

**Output**: Working MCP server + demo video

**Tech**: MCP Python SDK, Databricks SDK, asyncio

---

### Project 18: Domain Expert Fine-Tune
**Phase**: 7a | **Difficulty**: 7/10 | **Time**: 2–3 weeks

**Goal**: Fine-tune an LLM to be an expert on your domain.

**Tasks**:
- Collect 2000+ QA pairs from Databricks docs, Stack Overflow, your own experience
- Clean and format as ChatML instruction-response pairs
- Quality filter: review 100 random examples manually
- Fine-tune LLaMA 3.1 8B with QLoRA (r=16, alpha=16) on free Colab GPU
- Evaluate: 50 domain-specific test questions vs. GPT-4o-mini
- Push adapters to HuggingFace Hub

**Output**: Fine-tuned model on HuggingFace + evaluation report

**Tech**: Unsloth, TRL, PEFT, bitsandbytes, W&B

---

### Project 19: DPO Alignment Fine-Tune
**Phase**: 7b | **Difficulty**: 8/10 | **Time**: 2 weeks

**Goal**: Align a fine-tuned model using DPO.

**Tasks**:
- Start from Project 18's fine-tuned model (SFT base)
- Generate 500 responses to domain prompts
- Use GPT-4 to create preferred alternatives for low-quality responses
- Build (prompt, chosen, rejected) dataset
- Train with DPO (beta=0.1) using TRL DPOTrainer
- Compare: SFT vs. DPO on 30 test prompts using LLM-as-judge

**Output**: DPO-aligned model + comparison report showing improvement

**Tech**: TRL DPOTrainer, PEFT, W&B, OpenAI API (judge)

---

### Project 20: Multimodal Document QA
**Phase**: 8 | **Difficulty**: 8/10 | **Time**: 2 weeks

**Goal**: Build a system that understands both text and images.

**Tasks**:
- Process PDFs that contain charts, tables, and images
- Use LLaVA to extract information from visual elements
- Combine visual and text chunks in a single vector database
- Build RAG pipeline that retrieves both text and image information
- Answer questions that require cross-modal reasoning

**Output**: Multimodal RAG system + demo with 20 test questions

**Tech**: LLaVA (via HuggingFace), PyMuPDF, ChromaDB, sentence-transformers

---

### Project 21: Optimized LLM Inference Server
**Phase**: 8 | **Difficulty**: 8/10 | **Time**: 1 week

**Goal**: Deploy an LLM serving stack optimized for production.

**Tasks**:
- Deploy LLaMA 3.1 8B Instruct with vLLM
- Apply 4-bit AWQ quantization
- Benchmark: tokens/sec, latency p50/p95/p99, VRAM usage
- Implement request batching
- Expose OpenAI-compatible API
- Compare: HuggingFace baseline vs. vLLM vs. vLLM+AWQ

**Output**: Deployed server + performance benchmark report

**Tech**: vLLM, AutoAWQ, FastAPI (for wrapper), locust (load testing)

---

## Expert / Portfolio Projects (Months 15–18+)

### Project 22: Production RAG System with Full Observability
**Phase**: 9 | **Difficulty**: 9/10 | **Time**: 3 weeks | **PORTFOLIO PROJECT**

**Goal**: Production-quality AI system with all engineering best practices.

**Architecture**:
```
User → FastAPI endpoint
         ↓
    Input Guardrails (injection detection)
         ↓
    Semantic Cache (Redis)    ← Cache hit: return immediately
         ↓ (cache miss)
    Query Rewriter (LLM)
         ↓
    Hybrid Retrieval (BM25 + Dense + Reranker)
         ↓
    Context Compression (LLM)
         ↓
    Answer Generation (streaming)
         ↓
    Output Guardrails (PII detection)
         ↓
    LangSmith trace logged
         ↓
    RAGAS evaluation (async)
         ↓
    Response streamed to user
```

**Tasks**:
- All components above implemented and working
- FastAPI with streaming support
- LangSmith tracing on every request
- Automated RAGAS evaluation (weekly job)
- Grafana dashboard: latency, cost, quality, cache hit rate
- CI/CD: GitHub Actions that runs evaluation on every PR
- Docker Compose deployment

**Output**: Deployed application + runbook + monitoring dashboard

**Tech**: FastAPI, LangSmith, Redis, ChromaDB/Qdrant, vLLM, Grafana, Docker

---

### Project 23: End-to-End ML Platform (Databricks)
**Phase**: 9 | **Difficulty**: 9/10 | **Time**: 3 weeks | **PORTFOLIO PROJECT**

**Goal**: Leverage your Databricks expertise to build a complete ML platform.

**Tasks**:
- Data pipeline: PySpark + Delta Lake ingestion of text data
- Embedding generation: Spark UDF with sentence-transformers (distributed)
- Vector store: Databricks Vector Search index
- MLflow: track experiments, register models, serve models
- Databricks Jobs: scheduled training and evaluation jobs
- Unity Catalog: version and govern ML artifacts
- Serving: Databricks Model Serving for real-time inference

**Output**: Complete Databricks ML platform; demonstrates data engineering → AI engineering bridge

**Tech**: Databricks, PySpark, Delta Lake, MLflow, Unity Catalog, Databricks Vector Search

---

### Project 24: AI Code Review Bot
**Phase**: 9 | **Difficulty**: 9/10 | **Time**: 2 weeks | **PORTFOLIO PROJECT**

**Goal**: Real-world AI tool with business value.

**Tasks**:
- GitHub App that automatically reviews PRs
- LLM analyzes: code quality, security issues, performance, test coverage
- Generates actionable inline comments
- Configurable: via .ai-review.yaml in repo
- Supports: Python, SQL, Scala (for Databricks/Spark users)
- Integrates with LangSmith for review quality tracking

**Output**: Deployed GitHub App + 5 real PRs reviewed with it

**Tech**: GitHub API/webhooks, FastAPI, OpenAI/Anthropic, LangSmith

---

### Project 25: Personal AI Research Assistant
**Phase**: 9–10 | **Difficulty**: 8/10 | **Time**: 2 weeks | **PORTFOLIO PROJECT**

**Goal**: A system that helps you stay current with AI research.

**Tasks**:
- Daily cron: fetch new papers from ArXiv (cs.LG, cs.CL, cs.AI)
- Filter by relevance to your interests (embedding similarity to your profile)
- Summarize selected papers using an LLM
- Store in personal vector database
- Chat interface: ask questions about your paper library
- Weekly digest: top 5 papers with summaries, sent via email

**Output**: Running system that manages your paper reading

**Tech**: ArXiv API, LangGraph (for scheduling), vector DB, email (SendGrid)

---

### Project 26: Kafka-to-LLM Real-Time Intelligence Pipeline
**Phase**: 9 | **Difficulty**: 9/10 | **Time**: 3 weeks | **Unique to your background**

**Goal**: Apply your Kafka expertise to build real-time AI enrichment.

**Tasks**:
- Consume messages from Kafka topic (financial news, social media, support tickets)
- Real-time enrichment: LLM extraction (entity, sentiment, classification)
- Stream enriched events back to Kafka output topic
- Vector indexing: async update ChromaDB/Qdrant with new embeddings
- Dashboard: Grafana showing message rates, LLM latency, cost per message
- Handle backpressure: rate limiting LLM calls appropriately

**Output**: Running pipeline + architecture diagram + cost analysis

**Tech**: Apache Kafka (Python consumer), LangChain, ChromaDB, Grafana

---

### Project 27: LLM Inference Optimization Study
**Phase**: 8–9 | **Difficulty**: 9/10 | **Time**: 2 weeks | **Research-oriented**

**Goal**: Systematic study of inference optimization techniques.

**Tasks**:
- Baseline: HuggingFace inference, FP16, batch=1
- Compare 10 configurations:
  - FP16 vs. BF16 vs. INT8 vs. INT4 (GPTQ) vs. INT4 (AWQ)
  - Single request vs. batch=4 vs. batch=16
  - With/without FlashAttention
  - With/without torch.compile
  - HuggingFace vs. vLLM vs. llama.cpp
- Metrics: tokens/sec, latency p50/p95, VRAM, quality (perplexity)

**Output**: Comprehensive benchmark report; which config to use for each scenario

**Tech**: HuggingFace, vLLM, llama.cpp, auto-gptq, awq, nvidia-smi, PyTorch profiler

---

### Project 28: Open Source Contribution
**Phase**: 10 | **Difficulty**: 9/10 | **Time**: 3–4 weeks | **Community Impact**

**Goal**: Make a meaningful contribution to a major AI open source project.

**Ideas**:
- HuggingFace Transformers: Add a new model or improve an existing model's documentation
- LangChain: Build a Databricks Community Edition integration
- vLLM: Add a new model support or improve documentation
- MCP: Build and publish a Databricks MCP server package
- Your own: open-source your Kafka-to-LLM pipeline or Databricks ML Platform

**Process**:
1. Find a good first issue (or propose one with RFC)
2. Comment "I'll work on this"
3. Open draft PR early to get feedback
4. Iterate until merged

**Output**: Merged PR in a major repository + LinkedIn post

---

### Project 29: AI Engineering Blog Series
**Phase**: 10 | **Difficulty**: 7/10 | **Time**: Ongoing**

**Goal**: Build your reputation and crystallize learning by writing.

**Article ideas**:
- "Building a Production RAG System for Databricks Documentation"
- "Why I Chose DPO Over PPO for Fine-Tuning My Domain Model"
- "Optimizing LLM Inference: From 10 tok/s to 100 tok/s"
- "MCP: The Protocol That Will Change How You Use AI Tools"
- "Lessons Learned from 18 Months Learning GenAI as a Data Engineer"

**Platforms**: Medium, Substack, Hashnode, dev.to, HuggingFace Blog

**Output**: 5+ published articles with > 100 views each

---

### Project 30: Your AI Startup / Side Project
**Phase**: 10+ | **Difficulty**: 10/10 | **Time**: 6–12 months

**Goal**: Apply everything to build something people actually use.

**Ideas inspired by your background**:
- **DataEngineer.ai**: AI assistant for Databricks/Spark workflows (answers questions about your data)
- **PipelineGuard**: AI-powered monitoring that explains anomalies in data pipelines in English
- **NSRInsight**: Intelligent interface for the content externalization work you already do
- **SparkCoach**: AI tutor for PySpark optimization (you have the domain expertise)

**Output**: Something you're proud to show in an interview or LinkedIn

---

## Project Selection by Goal

| Goal | Recommended Projects |
|------|---------------------|
| Understand foundations deeply | 4 (NN from scratch), 6 (Backprop viz), 9 (GPT from scratch) |
| Build a portfolio fast | 9, 14, 16, 22, 24 |
| Leverage Databricks background | 17 (MCP), 23 (ML Platform), 26 (Kafka-LLM) |
| Learn fine-tuning | 18 (LoRA), 19 (DPO) |
| Land AI Engineer job | 9, 14, 16, 17, 22 (these 5 cover all interview topics) |
| Start open source contribution | 28 (choose HuggingFace or vLLM) |
| Build for production | 21, 22, 23 |

---

## GitHub Portfolio Guide

```
Recommended GitHub profile structure for AI Engineers:

Pinned repositories (show 6 total):
  1. gpt-from-scratch           ← Project 9 — shows deep fundamentals
  2. production-rag-system       ← Project 22 — shows production skills
  3. databricks-mcp-server      ← Project 17 — unique to your background
  4. llm-finetuning-lora        ← Project 18 — shows fine-tuning skills
  5. rag-evaluation-benchmark   ← Project 15 — shows systematic thinking
  6. kafka-llm-pipeline         ← Project 26 — bridges your expertise

Each repo README should have:
  - Architecture diagram (Mermaid or image)
  - What you built and why
  - Key technical decisions and tradeoffs
  - How to run it
  - Sample output or demo GIF
  - What you learned

What NOT to include:
  × Tutorial notebooks with no original work
  × Projects with no README
  × Forked repos you didn't modify
  × Half-finished work
```

---

*Projects Reference | Part of the [GenAI Engineer Roadmap](./00_README.md)*
