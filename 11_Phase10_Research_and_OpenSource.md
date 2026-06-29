# Phase 10: Research & Open Source Contributions
**Month 18+ | Difficulty: 10/10 | Label: 🟢 Optional (Long-term)**

> **Previous Phase**: [Phase 9 — Production AI Engineering](./10_Phase9_Production_AI_Engineering.md)  
> **Back to Start**: [Roadmap README](./00_README.md)

---

## Table of Contents

- [Phase Overview](#phase-overview)
- [Part A: Reading Research Papers](#part-a-reading-research-papers)
  - [The 5-Pass Method](#the-5-pass-method)
  - [How to Critique a Paper](#how-to-critique-a-paper)
  - [Staying Current](#staying-current)
- [Part B: The 15 Foundational Papers](#part-b-the-15-foundational-papers)
  - [Tier 1: Non-Negotiable (Read First)](#tier-1-non-negotiable-read-first)
  - [Tier 2: Essential (Read After Tier 1)](#tier-2-essential-read-after-tier-1)
  - [Tier 3: Highly Recommended](#tier-3-highly-recommended)
- [Part C: The 10 Seminal Papers (History)](#part-c-the-10-seminal-papers-history)
- [Part D: Paper Reading Resources](#part-d-paper-reading-resources)
- [Part E: Open Source Contribution Guide](#part-e-open-source-contribution-guide)
  - [How to Find Good First Issues](#how-to-find-good-first-issues)
  - [Contributing to Hugging Face Transformers](#contributing-to-hugging-face-transformers)
  - [Contributing to LangChain](#contributing-to-langchain)
  - [Contributing to vLLM](#contributing-to-vllm)
- [Part F: Building Your Research Practice](#part-f-building-your-research-practice)
- [Resources](#resources)
- [Final Mastery Checklist](#final-mastery-checklist)

---

## Phase Overview

| Attribute | Details |
|-----------|---------|
| **Duration** | Month 18+ (ongoing) |
| **Daily Time** | 30–60 min (sustainable long-term) |
| **Difficulty** | 10/10 |
| **Label** | 🟢 Optional |
| **Prerequisites** | All prior phases |
| **Outcome** | Can read and critique research papers; contributes to AI open source |

This phase is not a destination — it's the beginning of a continuous practice. The goal is to develop the habits and skills to stay at the frontier of AI indefinitely.

---

## Part A: Reading Research Papers

### The 5-Pass Method

Reading research papers is a learnable skill. Use this systematic approach:

```
Pass 1: The 5-Minute Survey (decide if worth reading fully)
─────────────────────────────────────────────────────────
□ Read: Title, authors, abstract
□ Read: Section headings and subheadings (scan structure)
□ Read: Conclusion
□ Glance: Figures and tables (find the "money figure")
□ Decide: Is this paper relevant? Rate 1-3
  1 = Skip
  2 = Read summary/blog post later
  3 = Read fully

Pass 2: The Full Read (90 minutes, skip difficult math)
──────────────────────────────────────────────────────
□ Read introduction carefully (what problem? what's the contribution?)
□ Read related work lightly (note what you don't know)
□ Read proposed method carefully (understand the key idea)
□ Examine all figures and tables in detail
□ Read experimental setup briefly
□ Read results and discussion carefully
□ Skip: detailed proofs, math derivations (first pass)
□ Write: 3-sentence summary after completing

Pass 3: The Deep Dive (2-4 hours, if paper is very important)
─────────────────────────────────────────────────────────────
□ Work through mathematical proofs by hand
□ Reproduce key equations from scratch
□ Identify every assumption made
□ Find limitations mentioned and unstated
□ Note: What experiment would disprove this?
□ Implement a simplified version in code

Pass 4: Critical Analysis
─────────────────────────
□ What are the strongest claims?
□ What evidence supports each claim?
□ Are there cherry-picked experiments?
□ Do the ablations actually isolate what they claim?
□ Are baselines fair and up-to-date?
□ How do results generalize beyond the test cases shown?

Pass 5: Implementation (only for key papers)
────────────────────────────────────────────
□ Implement the core algorithm from scratch
□ Reproduce at least one result from the paper
□ Write a blog post or notebook explaining it
```

---

### How to Critique a Paper

```
Questions to ask every paper:

1. NOVELTY: Is this actually new?
   - Is this an incremental improvement or a paradigm shift?
   - How does this compare to concurrent work?

2. VALIDITY: Do the experiments prove the claims?
   - Are baselines strong and fairly compared?
   - Are there enough seeds/runs to establish significance?
   - Is the evaluation dataset appropriate?

3. REPRODUCIBILITY: Can you redo this?
   - Are hyperparameters reported completely?
   - Is code released?
   - Are datasets publicly available?

4. LIMITATIONS: What doesn't the paper tell you?
   - What failure cases does it not show?
   - What settings was it NOT tested on?
   - What assumptions must hold for this to work?

5. IMPACT: Will this matter in 5 years?
   - Does it solve a real problem at scale?
   - Can practitioners use it without a PhD?
   - Does it unify existing ideas or fragment further?
```

---

### Staying Current

```python
# Your daily/weekly paper reading routine

DAILY = [
    "Hugging Face Daily Papers: huggingface.co/papers",  # Curated, community-voted
    "ArXiv Sanity: arxiv-sanity-lite.com",               # Most cited recent papers
    "Twitter/X: @karpathy, @ylecun, @fchollet, @GaryMarcus",  # Researchers
]

WEEKLY = [
    "Papers With Code (paperswithcode.com) — track SOTA across benchmarks",
    "The Batch (deeplearning.ai/the-batch) — Andrew Ng's weekly newsletter",
    "Import AI (importai.substack.com) — Jack Clark's newsletter",
]

MONTHLY = [
    "Review your paper reading notes",
    "Implement one paper from the month",
    "Update your mental model of the field",
]

# Tools for managing papers
paper_tools = {
    "Connected Papers":    "connectedpapers.com — visualize paper citation network",
    "Semantic Scholar":    "semanticscholar.org — AI-powered paper search with citations",
    "Zotero":              "Free reference manager, browser extension",
    "ResearchRabbit":      "Research discovery and organization",
    "Elicit":              "AI research assistant for literature review",
}
```

---

## Part B: The 15 Foundational Papers

### Tier 1: Non-Negotiable (Read First)

These 5 papers are the foundation. Read them before anything else.

| Priority | Paper | Year | Why Read |
|----------|-------|------|---------|
| 🔴 1 | **Attention Is All You Need** (Vaswani et al.) | 2017 | Every LLM is built on this. 11 pages. Read 3 times. |
| 🔴 2 | **Language Models are Few-Shot Learners** (Brown et al. — GPT-3) | 2020 | Proves that scale → emergent capabilities. Defines few-shot learning. |
| 🔴 3 | **Training Language Models to Follow Instructions with Human Feedback** (Ouyang et al. — InstructGPT) | 2022 | How ChatGPT was born. RLHF in detail. |
| 🔴 4 | **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al.) | 2021 | The dominant fine-tuning method. Used everywhere. |
| 🔴 5 | **Retrieval-Augmented Generation** (Lewis et al.) | 2020 | Foundation of all RAG systems. |

---

### Tier 2: Essential (Read After Tier 1)

| Priority | Paper | Year | Why Read |
|----------|-------|------|---------|
| 🟡 6 | **Scaling Laws for Neural Language Models** (Kaplan et al.) | 2020 | Why bigger models are better. Chinchilla follows from this. |
| 🟡 7 | **Training Compute-Optimal Large Language Models** (Hoffmann et al. — Chinchilla) | 2022 | Correct the scaling laws. Optimal data:parameter ratio. |
| 🟡 8 | **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al.) | 2023 | The open-source LLM revolution. How Meta changed the field. |
| 🟡 9 | **FlashAttention** (Dao et al.) | 2022 | How to make attention memory-efficient. Used in every modern framework. |
| 🟡 10 | **Direct Preference Optimization** (Rafailov et al.) | 2023 | Simpler alternative to RLHF. Now the standard alignment method. |

---

### Tier 3: Highly Recommended

| Priority | Paper | Year | Why Read |
|----------|-------|------|---------|
| 🟢 11 | **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al.) | 2023 | Fine-tuning 65B models on a single GPU. Enables everyone to fine-tune. |
| 🟢 12 | **Efficient Estimation of Word Representations in Vector Space** (Mikolov et al. — Word2Vec) | 2013 | Foundation of modern embeddings. Understand the jump to dense representations. |
| 🟢 13 | **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al.) | 2018 | The encoder model. Understand encoder vs. decoder architectures. |
| 🟢 14 | **Chain-of-Thought Prompting Elicits Reasoning** (Wei et al.) | 2022 | Why "think step by step" works. Foundational for agent reasoning. |
| 🟢 15 | **Constitutional AI: Harmlessness from AI Feedback** (Bai et al. — Anthropic) | 2022 | How to align AI without human labelers. Anthropic's approach. |

---

## Part C: The 10 Seminal Papers (History)

These papers built the foundations before transformers. Reading them gives deep context.

| Year | Paper | Contribution |
|------|-------|-------------|
| 1986 | **Learning Representations by Back-propagating Errors** (Rumelhart et al.) | Backpropagation — makes neural nets trainable |
| 1997 | **Long Short-Term Memory** (Hochreiter, Schmidhuber) | LSTM — solved vanishing gradient for sequences |
| 1998 | **Gradient-Based Learning Applied to Document Recognition** (LeCun et al.) | CNNs and LeNet — first practical deep net |
| 2012 | **ImageNet Classification with Deep CNNs** (Krizhevsky et al. — AlexNet) | Started the deep learning revolution |
| 2014 | **Generative Adversarial Nets** (Goodfellow et al.) | GANs — the first generative model revolution |
| 2014 | **Sequence to Sequence Learning with NNs** (Sutskever et al.) | Encoder-Decoder architecture, precursor to transformers |
| 2015 | **Deep Residual Learning for Image Recognition** (He et al. — ResNet) | Skip connections — made very deep networks trainable |
| 2015 | **Neural Machine Translation by Jointly Learning to Align and Translate** (Bahdanau et al.) | Attention mechanism — the precursor to transformers |
| 2020 | **An Image is Worth 16×16 Words** (Dosovitskiy et al. — ViT) | Transformers for vision |
| 2021 | **Denoising Diffusion Probabilistic Models** (Ho et al. — DDPM) | Foundation of modern image generation |

---

## Part D: Paper Reading Resources

```python
# Essential channels for paper explanations

youtube_channels = {
    "Yannic Kilcher":      "@YannicKilcher — Best paper walkthroughs. Start here.",
    "Two Minute Papers":   "@TwoMinutePapers — Quick paper summaries, enthusiastic",
    "Andrej Karpathy":     "@karpathy — Implementation + intuition, rare but gold",
    "Aleksa Gordić":       "@TheAIEpiphany — Detailed transformer paper walkthroughs",
}

blogs = {
    "Lilian Weng":    "lilianweng.github.io — Deep, comprehensive, math included",
    "Jay Alammar":    "jalammar.github.io — Visual explanations of transformers",
    "Sebastian Ruder":"ruder.io — NLP research, best survey posts",
    "The Gradient":   "thegradient.pub — Long-form research essays",
}

# How to find a paper to read tonight:
# 1. Go to huggingface.co/papers
# 2. Sort by "most liked today"
# 3. Pick a paper in your current study phase
# 4. Watch Yannic Kilcher's video if available
# 5. Then read the paper
```

---

## Part E: Open Source Contribution Guide

### How to Find Good First Issues

```bash
# GitHub labels to search for beginner-friendly issues
Labels: "good first issue", "help wanted", "beginner friendly", "easy"

# Effective search:
# github.com/search?q=label%3A"good+first+issue"+language%3APython+topic%3Allm

# What makes a good first contribution:
# 1. Documentation fix (typos, missing examples) — very easy
# 2. Test addition (increase test coverage) — well-defined
# 3. Small bug fix with a clear reproduction case — satisfying
# 4. Implementing a well-specified feature request — challenging but clear
```

---

### Contributing to Hugging Face Transformers

```python
# github.com/huggingface/transformers
# One of the most impactful AI repositories

# Good first contributions:
# 1. Add a model card / model documentation
# 2. Fix a typo in docstrings
# 3. Add missing test for an edge case
# 4. Improve error messages (make them more helpful)

# Process:
# 1. Fork the repository
# 2. Clone your fork: git clone https://github.com/YOUR_USERNAME/transformers
# 3. Create a branch: git checkout -b fix/improve-attention-docs
# 4. Make your changes
# 5. Run tests: make test
# 6. Submit PR with clear description

# Example: Adding a new model
# 1. Read the existing model implementation (e.g., modeling_llama.py)
# 2. Follow the template for adding new architectures
# 3. Add configuration class (configuration_modelname.py)
# 4. Add tokenizer if different (tokenization_modelname.py)
# 5. Add model class (modeling_modelname.py)
# 6. Add tests (tests/models/test_modeling_modelname.py)
# 7. Update docs (docs/source/en/model_doc/modelname.md)
# 8. Add to auto-classes in utils/
```

---

### Contributing to LangChain

```python
# github.com/langchain-ai/langchain
# Most widely-used LLM application framework

# Good contribution areas:
# 1. New tool integrations (connect a service to LangChain)
# 2. New retriever implementations
# 3. Documentation examples (Jupyter notebooks)
# 4. Bug fixes (search GitHub issues)

# Building a new LangChain tool integration:
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class DatabricksQueryInput(BaseModel):
    """Input for the Databricks SQL tool."""
    query: str = Field(description="SQL query to execute on Databricks")
    warehouse_id: str = Field(description="Databricks SQL warehouse ID")

class DatabricksSQLTool(BaseTool):
    """Tool for executing SQL queries on Databricks."""
    
    name: str = "databricks_sql"
    description: str = """
    Execute SQL queries on Databricks. Use this tool to:
    - Query Delta tables
    - Explore data schemas
    - Run analytical queries
    Input: SQL query and warehouse ID
    Output: Query results as JSON
    """
    args_schema: Type[BaseModel] = DatabricksQueryInput
    
    host: str
    token: str
    
    def _run(self, query: str, warehouse_id: str) -> str:
        """Execute the SQL query."""
        from databricks import sql as databricks_sql
        import json
        
        with databricks_sql.connect(
            server_hostname=self.host,
            http_path=f"/sql/1.0/warehouses/{warehouse_id}",
            access_token=self.token
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchmany(100)
                result = [dict(zip(columns, row)) for row in rows]
        
        return json.dumps(result[:20], indent=2, default=str)

# Submitting to LangChain:
# 1. Create community/ folder entry or partner integration
# 2. Add comprehensive docstrings
# 3. Add unit tests with mocked external calls
# 4. Add integration tests (optional, marked as such)
# 5. Write a README for the integration
```

---

### Contributing to vLLM

```python
# github.com/vllm-project/vllm
# The leading LLM inference engine

# Contribution areas:
# 1. Model support: port a new model architecture
# 2. Quantization: add a new quantization format
# 3. Benchmarking: improve benchmark scripts
# 4. Documentation: API docs, examples

# Adding a new model to vLLM (advanced):
# 1. Study an existing model (vllm/model_executor/models/llama.py)
# 2. Create vllm/model_executor/models/your_model.py
# 3. Implement: __init__, forward, load_weights methods
# 4. Register in vllm/model_executor/models/__init__.py
# 5. Add to supported models list
# 6. Write a test: tests/models/test_your_model.py
```

---

## Part F: Building Your Research Practice

```
Your sustainable research routine (30-60 min/day):

Monday:    Read one paper (Pass 1: all papers; Pass 2: important ones)
Tuesday:   Implement something from the paper or week's learning
Wednesday: Read a blog post or watch a YouTube paper walkthrough
Thursday:  Work on open source contribution (or project)
Friday:    Review week's learnings, write notes

Monthly:
  - Write a summary/blog post of the most interesting paper you read
  - Review your GitHub contributions
  - Update your mental model of the field
  - Identify new research directions to explore

Yearly:
  - Read a book (deep learning textbook, or a technical AI book)
  - Attend a conference (NeurIPS, ICML, ICLR, EMNLP) — watch livestreams
  - Do one larger open source contribution
  - Assess: what are the hot topics? Where do you have gaps?
```

---

## Resources

| Rank | Resource | Type | Cost | Why |
|------|----------|------|------|-----|
| 1 | [Papers With Code](https://paperswithcode.com/) | Tool | Free | State-of-the-art tracking + code implementations |
| 2 | [Yannic Kilcher YouTube](https://www.youtube.com/@YannicKilcher) | YouTube | Free | Best channel for understanding ML papers deeply |
| 3 | [Semantic Scholar](https://www.semanticscholar.org/) | Tool | Free | AI-powered paper search with citation graphs |
| 4 | [HuggingFace Daily Papers](https://huggingface.co/papers) | Newsletter | Free | Best curated daily paper feed |
| 5 | [How to Read a Paper (Keshav)](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf) | Paper | Free | The definitive guide to reading papers (3-pass method) |
| 6 | [Lilian Weng's Blog](https://lilianweng.github.io/) | Blog | Free | Comprehensive, math-included paper reviews |

---

## Final Mastery Checklist

### Paper Reading
- [ ] Read all 5 Tier 1 papers (attention, GPT-3, InstructGPT, LoRA, RAG)
- [ ] Read at least 5 Tier 2 papers
- [ ] Implemented at least 2 papers from scratch
- [ ] Used the 5-pass method consistently
- [ ] Developed a weekly paper reading routine

### Open Source
- [ ] Submitted at least 1 PR to a major AI repository
- [ ] At least 1 PR merged
- [ ] Familiar with contribution workflows (fork, branch, PR, review)

### Research Practice
- [ ] Can read and critically evaluate a new AI paper independently
- [ ] Staying current with the field via daily/weekly reading habits
- [ ] Can explain any recent breakthrough to a non-expert

---

## What Comes After This Roadmap?

At this point, you are no longer following a roadmap — you are at the frontier.

```
Signs you've completed the roadmap:
✓ You read papers and immediately understand the background and context
✓ You can look at a new model architecture and know what's interesting about it
✓ You can design AI systems from scratch for novel use cases
✓ You contribute to the tools others use
✓ You have strong opinions about the field (backed by evidence)
✓ Other engineers ask you for your take on new developments

What comes next:
→ Specialize in one sub-area (agents, fine-tuning, inference, safety, multimodal)
→ Write your own research paper or technical blog
→ Build a product using your skills
→ Mentor others through this roadmap
→ Apply for roles at AI labs (Anthropic, Google DeepMind, Meta FAIR, etc.)
→ Start an AI company
```

---

## Phase Completion & Readiness Assessment

> This is the culmination of the entire roadmap. There is no "next phase" — but the field keeps moving. This assessment measures whether you can operate at the research frontier indefinitely.

---

### 1. Knowledge Checklist

**Paper Reading**
- [ ] 5-pass method: skim → structure → detailed read → critical analysis → connect to practice
- [ ] How to extract the core contribution in one sentence
- [ ] How to identify what an author is claiming vs. actually proving
- [ ] How to spot common research weaknesses (limited baselines, cherry-picked examples)
- [ ] How to implement a paper from scratch given only the math

**Research Ecosystem**
- [ ] Tier 1 papers: Attention Is All You Need, BERT, GPT-3, InstructGPT, LoRA
- [ ] Tier 2 papers: LLaMA, Mixtral MoE, FlashAttention, HellaSwag, RLHF survey
- [ ] Tier 3 papers: Constitutional AI, Toolformer, Chain-of-Thought, Sparks of AGI
- [ ] Where to find papers: arXiv cs.CL/cs.LG, Papers With Code, HuggingFace Daily Papers
- [ ] How to evaluate a paper: rigor, novelty, reproducibility, practical impact

**Staying Current**
- [ ] Daily routine: arXiv new submissions, HF daily papers
- [ ] Weekly: Ahead of AI, The Batch, ML news roundups
- [ ] Monthly: read one foundational paper in full
- [ ] How to maintain a paper reading log

**Open Source Contribution**
- [ ] Fork → clone → branch workflow on GitHub
- [ ] How to find good first issues on HuggingFace, LangChain, or vLLM
- [ ] How to write a PR description that gets merged
- [ ] Contribution types: documentation, tests, bug fixes, new features
- [ ] How to respond to code review comments

---

### 2. Practical Skills Checklist

- [ ] Read a new arXiv paper and extract the core contribution in under 30 minutes
- [ ] Implement a simplified version of a major technique (LoRA, attention, BPE) from the math
- [ ] Write a clear, structured paper summary (background, contribution, method, results, limitations)
- [ ] Find and fork a project on GitHub, identify a good first issue
- [ ] Submit a real PR (even documentation or a test) to an open-source project
- [ ] Explain a paper to a non-ML colleague using only analogies
- [ ] Build a personal paper reading system (Obsidian, Notion, Markdown)

---

### 3. Coding Challenges

**Challenge A — Implement Attention From Scratch**
```python
# Implement scaled dot-product attention (single head) from the Attention paper:
# Q, K, V = query, key, value matrices (batch, seq, d_model)
# Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
# Requirements:
# 1. Pure PyTorch, no torch.nn.MultiheadAttention
# 2. Support batch dimension
# 3. Apply causal mask (upper triangular -inf) for decoder
# 4. Verify your implementation matches torch.nn.functional.scaled_dot_product_attention
# 5. Test with: batch=2, seq=16, d_model=64, d_k=64

import torch
# Your implementation here...
```

**Challenge B — Implement LoRA From Scratch**
```python
# Implement LoRA (Low-Rank Adaptation) from the LoRA paper:
# W' = W + B @ A   where B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)
# Requirements:
# 1. Create a LoRALinear(nn.Module) class that wraps nn.Linear
# 2. Only B and A are trainable; W is frozen
# 3. rank r is a hyperparameter
# 4. B is initialised to zeros, A with random Gaussian
# 5. Scale output by alpha/r
# 6. Replace all Linear layers in a GPT-2 with LoRALinear
# 7. Verify that parameter count is dramatically reduced
```

**Challenge C — Paper Benchmarking**
```python
# Pick any 2 models from HuggingFace and reproduce their benchmark comparison:
# 1. Choose a task (e.g., MMLU, HellaSwag, or code generation)
# 2. Load the eval dataset
# 3. Run both models on 100 examples
# 4. Compute the same metric used in the original paper
# 5. Write a brief analysis: do your results match the paper's?
#    If not: why might they differ? (quantisation, context length, prompt format)
```

---

### 4. Mini Project

**Paper Implementation Project**: Pick any one of these and implement it:
- Simplified BPE tokeniser (from the GPT-2 tokeniser paper) from scratch
- Simplified attention mechanism (single head, then multi-head) from "Attention Is All You Need"
- Simplified LoRA layer and apply it to GPT-2 fine-tuning

Document: background, core math, your implementation, verification tests, comparison to the original, what you learned.

---

### 5. Capstone Project

**"Your AI Research Portfolio"**: The culmination of the entire roadmap.

Publish a writeup that includes:
1. **Paper implementation**: implement one technique from scratch and explain the math
2. **Benchmark reproduction**: reproduce one result from a published paper
3. **Production lesson**: connect the research to something you built in Phase 9
4. **Open source PR**: one real contribution to a public AI project

Publish on: GitHub (code) + Medium/Substack/personal site (writeup) + LinkedIn. This is your public evidence of being a serious AI engineer.

---

### 6. Interview Questions

**Beginner**

1. **Q: What is arXiv and why is it important for AI?**
   A: arXiv is a free preprint server for academic papers. AI researchers publish there immediately after writing, before formal peer review. This means you can read the latest research the day it's written. The cs.CL (computation and language) and cs.LG (machine learning) sections are the most relevant for LLMs.

2. **Q: How do you read an academic paper efficiently?**
   A: 5-pass approach: (1) Skim — title, abstract, headings, conclusion. Goal: decide if worth reading. (2) Structure — identify the claim, method, and evaluation. (3) Detailed — read methods and results carefully, write equations by hand. (4) Critical — what are the limitations? (5) Connect — how does this relate to what I've built?

3. **Q: What is "Attention Is All You Need" and why does it matter?**
   A: The 2017 paper introducing the Transformer architecture. Before it: RNNs/LSTMs handled sequences sequentially. The Transformer replaced recurrence with multi-head self-attention, enabling massive parallelism and scaling. Every LLM today is built on this architecture.

4. **Q: What is the difference between a preprint and a published paper?**
   A: A preprint (e.g., arXiv) is not peer reviewed — the authors post it directly. A published paper went through peer review (experts checked the methodology). In AI, preprints often become the primary reference even before or instead of formal publication because the field moves too fast.

5. **Q: What makes a good open source contribution?**
   A: (1) Solves a real problem others have reported. (2) Follows the project's coding conventions. (3) Includes tests if the project has tests. (4) Has a clear PR description explaining the problem, solution, and how to test it. (5) Starts small: documentation fix or bug → then features.

6. **Q: How do you find your first open source issue to work on?**
   A: Filter GitHub issues by label: "good first issue", "help wanted", "documentation". On large projects: look for issues with clear descriptions and no active PR already assigned. Read the existing code first. Comment "I'd like to work on this" before spending days on it.

7. **Q: What is "Papers With Code"?**
   A: A website that links research papers to their open-source implementations. Also tracks state-of-the-art benchmarks for ML tasks. Useful for: finding implementations to study, checking if a paper beat the SOTA, and finding reproducible experiments.

**Intermediate**

8. **Q: Walk me through the Transformer architecture and explain each component's role.**
   A: (1) Tokeniser: text → token IDs. (2) Embedding layer: IDs → dense vectors. (3) Positional encoding: adds position information (Transformers have no inherent order). (4) Multi-head attention: each token attends to all others, different heads capture different patterns. (5) Feed-forward: independent per-token MLP, adds capacity. (6) Layer norm + residual: training stability. (7) Output projection: hidden states → vocabulary logits → next token probabilities.

9. **Q: What is the LoRA paper's key insight and why does it work?**
   A: Key insight: weight updates during fine-tuning have low intrinsic rank — they don't need to update all d×k parameters. Instead, ΔW = B@A where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r << d. This works because: (1) pre-trained models already encode rich representations; (2) fine-tuning only needs to shift a small subspace; (3) this has been verified empirically across dozens of tasks.

10. **Q: What is the RLHF paper (InstructGPT) and what problem does it solve?**
    A: InstructGPT (2022) solved "alignment" — making LLMs follow instructions and be helpful rather than just predicting text. Method: (1) SFT on human-written examples; (2) train a reward model on human preference pairs; (3) PPO to optimise the LLM to maximise the reward model. This is what made ChatGPT possible.

11. **Q: What is a Mixture of Experts (MoE) architecture?**
    A: MoE replaces each FFN layer with N "experts" (separate FFNs) plus a router. Each token is routed to top-k experts (typically k=2). Benefits: model has N× more parameters but uses only k/N of them per token → same inference cost as a smaller dense model but much more capacity. Mixtral 8×7B, GPT-4 (rumoured) use this.

12. **Q: How would you critically evaluate a paper's claims?**
    A: (1) Baselines: does it compare against all relevant prior work, or cherry-picked weak ones? (2) Test set: is the test set standard and not contaminated? (3) Significance: is the improvement statistically significant? (4) Ablation: does it isolate what's actually causing improvement? (5) Reproducibility: is code and data released? (6) Compute: what resources were used; is it reproducible by normal researchers?

13. **Q: What is Constitutional AI (Claude's training approach)?**
    A: Constitutional AI (Anthropic) replaced human feedback for harmful content classification with AI self-critique. (1) Generate response; (2) ask the model to critique it against a constitution (set of principles); (3) revise the response; (4) use the revised response as training data. Reduces reliance on human annotators for harmful content labelling.

**Advanced**

14. **Q: Explain FlashAttention's contribution and why it matters for training scale.**
    A: Standard attention materialises the O(n²) attention matrix in HBM (GPU memory). FlashAttention uses tiled computation: (1) load Q,K,V in tiles from HBM to SRAM; (2) compute attention in tiles without materialising the full matrix; (3) write output back. Result: 2-4× speedup and O(n) memory instead of O(n²). This enabled longer context windows and faster training — critical for 100k+ context models.

15. **Q: What is the "bitter lesson" in AI research and what does it imply for engineers?**
    A: Richard Sutton's 2019 essay: general methods that leverage computation win over methods that encode human knowledge. Examples: hand-crafted chess features → self-play; hand-crafted NLP features → transformers. Implication for engineers: invest in scalable architectures and data pipelines, not clever domain-specific tricks. Scale wins.

16. **Q: What is Chain-of-Thought prompting and what does the paper reveal about LLM reasoning?**
    A: CoT (Wei et al. 2022): asking the model to "think step by step" dramatically improves multi-step reasoning. Key findings: (1) CoT is an emergent ability — only appears in large models (>100B); (2) zero-shot CoT ("think step by step") often works as well as few-shot CoT; (3) CoT helps most on tasks requiring intermediate computations. Implication: what looks like "reasoning" may actually be learned pattern completion.

17. **Q: What is the difference between a Tier 1 and Tier 3 paper and how do you prioritise reading?**
    A: Tier 1: foundational — changed the entire field (Attention Is All You Need, BERT, GPT-3). Must-read before anything else. Tier 3: current — published last year, improves on prior work incrementally. Nice to read to stay current but lower return on investment. Prioritise: read all Tier 1 first (10-15 papers); then skim Tier 3 titles/abstracts to know what exists; deep-read only when directly relevant to your work.

18. **Q: How would you contribute a feature (not just a bug fix) to LangChain?**
    A: (1) Open an issue describing the feature and wait for maintainer feedback before building. (2) Fork and create a feature branch. (3) Follow existing code patterns for similar components. (4) Write tests matching the project's test framework. (5) Update documentation. (6) Submit PR with: motivation, design decision rationale, test evidence. (7) Respond promptly to review comments. (8) Be prepared to iterate 3-5 times.

19. **Q: What is the "sparks of AGI" paper claiming and what is the counter-argument?**
    A: The Microsoft Research 2023 paper argued GPT-4 shows early signs of general intelligence: broad capability across diverse tasks, emergent reasoning, tool use. Counter-arguments: (1) the paper cherry-picks impressive examples; (2) GPT-4 still fails simple consistency tests; (3) "AGI" has no agreed definition — this is a marketing framing. The debate illustrates why critical reading of research is important.

20. **Q: If you were hiring an ML engineer at a senior level, what would you look for in their research literacy?**
    A: (1) Can explain 3-5 foundational papers from memory. (2) Has read papers outside their comfort zone. (3) Can distinguish what a paper claims vs. proves. (4) Has implemented at least one technique from scratch. (5) Stays current (knows papers from the last 6 months). (6) Can bridge paper → production: "this technique would improve our system because…"

---

### 7. Self-Assessment Quiz

- [ ] Name the 5 passes of the paper reading method.
- [ ] What is arXiv and what sections do you follow?
- [ ] What year was "Attention Is All You Need" published?
- [ ] What is the role of positional encoding in Transformers?
- [ ] What problem does LoRA solve?
- [ ] What is the rank r in LoRA and what does it control?
- [ ] What does InstructGPT do differently from GPT-3?
- [ ] What is a Mixture of Experts architecture?
- [ ] What is FlashAttention's key optimisation?
- [ ] What is the "bitter lesson" in AI?
- [ ] What is Chain-of-Thought prompting?
- [ ] What is Constitutional AI?
- [ ] Name 3 places to find daily AI research news.
- [ ] What is "Papers With Code"?
- [ ] What makes a good "good first issue" on GitHub?
- [ ] What should a good PR description contain?
- [ ] What is the difference between a preprint and a published paper?
- [ ] How do you evaluate whether a paper's baseline comparisons are fair?
- [ ] What is the difference between Tier 1 and Tier 3 papers?
- [ ] Name 3 papers from Tier 1 of the roadmap.
- [ ] What is HuggingFace Daily Papers?
- [ ] How do you write a paper summary?
- [ ] What is "emergent ability" in LLMs?
- [ ] What is an ablation study and why is it important?
- [ ] What does it mean to "reproduce a paper's results"?

**Scoring**: 22–25 ✅ = Ready to contribute at the frontier. 17–21 = Review weak areas. Below 17 = More deep reading needed.

---

### 8. Common Mistakes

| Mistake | Why It Happens | How to Avoid |
|---------|---------------|--------------|
| Reading papers but not implementing | Passive reading feels like learning | For every 3 papers read, implement one concept from scratch |
| Trying to read every new paper | FOMO; the field is fast | Daily skim (5 min), weekly deep read (1 paper), monthly foundational read |
| Contributing code before reading CONTRIBUTING.md | Eager to help | Always read CONTRIBUTING.md and open an issue before coding |
| Equating paper accuracy with production accuracy | Trust in published benchmarks | Benchmarks are narrow; always validate on your own data |
| Writing PR without tests | "Tests slow me down" | No tests = your PR will be rejected; tests are non-negotiable |
| Reading only recent papers | Recency bias | Foundational papers from 2017-2022 are still the most important; read them first |

---

### 9. Readiness Criteria

This is the final assessment — there is no next phase. You are operating at the frontier when **all** of the following are true:

- [ ] I have read all 5 Tier 1 papers and can explain each from memory
- [ ] I have implemented at least one major technique from scratch (attention, LoRA, BPE)
- [ ] I have submitted at least one PR to an open-source AI project
- [ ] I have a daily habit for staying current with AI research
- [ ] I maintain a personal paper reading log with summaries
- [ ] I have completed the Phase 10 Capstone (public writeup + code + PR)
- [ ] I scored 22/25 or higher on the Self-Assessment Quiz
- [ ] I can answer at least 16/20 Interview Questions correctly

---

### 10. Revision Summary

```
THE 5-PASS METHOD
─────────────────────────────────────────────────────
Pass 1: Skim         → worth reading? (5 min)
Pass 2: Structure    → claim, method, eval (20 min)
Pass 3: Deep read    → write out equations (1-2 hr)
Pass 4: Critical     → limitations, weak baselines? (30 min)
Pass 5: Connect      → how would this improve my system? (15 min)

TIER 1 PAPERS (read these first)
─────────────────────────────────────────────────────
Attention Is All You Need (2017)     → Transformer architecture
BERT (2018)                          → bidirectional pretraining
GPT-3 (2020)                         → few-shot learning at scale
InstructGPT (2022)                   → RLHF alignment
LoRA (2021)                          → efficient fine-tuning

OPEN SOURCE CONTRIBUTION FLOW
─────────────────────────────────────────────────────
fork → clone → branch → code → test → PR
PR description: problem / solution / test evidence
Always: read CONTRIBUTING.md first, open issue before coding

STAYING CURRENT
─────────────────────────────────────────────────────
Daily (5 min):    arXiv new, HF daily papers
Weekly (30 min):  Ahead of AI, The Batch, 1 paper abstract
Monthly (2 hr):   1 foundational paper, deep read
```

---

### 11. Next Phase Prerequisites

**There is no Phase 11.** This is where the formal roadmap ends. From here, your growth depends on:

| Practice | Why It Sustains Growth |
|----------|----------------------|
| Daily paper reading habit | The field changes every month; staying current is mandatory |
| Regular open source contribution | Forces production-quality code; builds reputation |
| Public writing (blog, talks) | Clarifies thinking; creates career opportunities |
| Connecting research to production | The rare engineer who can do both; highest value in the market |

**The compound loop that sustains elite AI engineers:**

```
Read research → Implement it → Deploy it → Measure the gap from the paper → 
Report findings → Contribute improvements → Read more research
```

This loop is the career. There is no destination — only the quality of the loop.

---

*Phase 10 | Part of the [GenAI Engineer Roadmap](./00_README.md)*  
*[← Back to README](./00_README.md) | [← Phase 9](./10_Phase9_Production_AI_Engineering.md)*
