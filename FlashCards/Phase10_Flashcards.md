# Phase 10 Flashcards — Research & Open Source

[← Flashcards Index](./README.md) | [Phase 10 File](../11_Phase10_Research_and_OpenSource.md) | [Cheat Sheet](../CheatSheets/Phase10_Research.md)

---

**ID**: P10-001
**Front**: What is the 3-pass method for reading research papers? How long does each pass take?
**Back**: Pass 1 (10 min): Read title, abstract, introduction, conclusions, headings, scan figures. Decide: worth full read? Pass 2 (1–2 hrs): Read all except proofs. Note key equations, tables, experimental setup. Understand mechanism and results. Pass 3 (4+ hrs): Deep dive, verify claims, reproduce results, check references. For most papers: Pass 1 + 2 is sufficient. Pass 3 for papers you want to build on or cite in your own work.
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase10 #paper-reading #research

---

**ID**: P10-002
**Front**: What did "Attention Is All You Need" (2017) contribute?
**Back**: Vaswani et al. 2017: introduced the Transformer architecture for sequence-to-sequence tasks. Key contributions: (1) Replaced RNNs with self-attention — fully parallelizable training. (2) Multi-head attention. (3) Positional encoding. (4) Pre-norm/post-norm variants. (5) Scaled dot-product attention with √d_k. Set SOTA on WMT translation. Foundation for: GPT, BERT, LLaMA, and virtually all modern NLP. Most cited paper in AI history.
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase10 #attention #transformer #seminal-paper

---

**ID**: P10-003
**Front**: What did InstructGPT (2022) prove about LLM training?
**Back**: Ouyang et al. 2022 (OpenAI): RLHF fine-tuning makes LLMs dramatically more useful. Key findings: (1) 1.3B InstructGPT preferred by humans over 175B GPT-3 (smaller + aligned > bigger + unaligned). (2) RLHF reduces hallucination, improves instruction following, improves truthfulness. (3) Alignment tax: small degradation in NLP benchmarks for large gains in helpfulness. Impact: became the blueprint for ChatGPT and all aligned LLMs.
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase10 #InstructGPT #RLHF #alignment

---

**ID**: P10-004
**Front**: What did the Chinchilla scaling law paper (2022) change about LLM training?
**Back**: Hoffmann et al. (DeepMind) 2022: optimal compute allocation ≈ equal parameters and training tokens. Prior: researchers trained models like GPT-3 (175B params on 300B tokens — undertraining). Chinchilla: 70B model on 1.4T tokens (same compute as Gopher 280B on 300B tokens) → better than Gopher. Impact: LLaMA series deliberately overtrained smaller models (LLaMA-3-8B on 15T tokens) for better inference-time efficiency. Now everyone trains smaller models on more data.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase10 #scaling-laws #chinchilla

---

**ID**: P10-005
**Front**: What did LoRA (Hu et al., 2021) establish?
**Back**: LoRA: Low-Rank Adaptation of Large Language Models. Key findings: (1) Fine-tuning has low intrinsic rank — most of the update lives in a low-dimensional subspace. (2) Low-rank decomposition of weight updates (r=4-64) achieves near full fine-tuning performance. (3) No inference latency (adapters merged at deployment). (4) 10,000× fewer trainable parameters for GPT-3 scale. Impact: democratized fine-tuning — you don't need to train all 175B parameters to specialize a model.
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase10 #LoRA #fine-tuning

---

**ID**: P10-006
**Front**: What are the "good first issue" labels on GitHub and how do you use them?
**Back**: GitHub label for beginner-friendly issues that don't require deep codebase knowledge. Filter: `is:open is:issue label:"good first issue"` in GitHub search. How to approach: (1) Read the issue + all comments. (2) Comment "I'd like to work on this" (avoid duplicate work). (3) Fork → clone → create branch `fix/issue-description`. (4) Make focused change. (5) Write/run tests. (6) Submit PR. (7) Address review comments. Start with documentation → then tests → then small bug fixes → then features.
**Difficulty**: Beginner
**Category**: Research
**Tags**: #phase10 #open-source #github #contributing

---

**ID**: P10-007
**Front**: What are the key OSS projects worth contributing to for an LLM engineer?
**Back**: High impact: (1) LangGraph/LangChain: most widely used agentic framework. (2) vLLM: fastest growing LLM serving project. (3) RAGAS: RAG evaluation (smaller, more approachable). (4) instructor: structured output library. (5) LiteLLM: universal LLM API. (6) HuggingFace Transformers: core library (harder, higher bar). Strategy: start with smaller projects (RAGAS, instructor, LiteLLM), build track record, then tackle larger (vLLM, LangGraph).
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase10 #open-source #contributing

---

**ID**: P10-008
**Front**: How do you write a good PR description for an open source contribution?
**Back**: Good PR: (1) Title: `fix: resolve KeyError when empty messages list passed to chat()`. (2) Description: what problem does this fix? link to issue. (3) How you fixed it: brief technical explanation. (4) Tests added/changed. (5) Any breaking changes. (6) Screenshots if UI. (7) How reviewer can test it. Bad: "Fixed bug" with no context. Rules: one logical change per PR. Small focused PRs get merged faster. Respect project style guide. Run CI locally before pushing.
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase10 #open-source #PR #contributing

---

**ID**: P10-009
**Front**: What is arXiv and what is the paper submission process?
**Back**: arXiv (arxiv.org): preprint server — publish before peer review. Free, immediate public access. Sections: cs.CL (NLP), cs.AI (AI), cs.LG (ML), stat.ML. Reading: search by keyword, author, abstract. New papers: Hugging Face Daily Papers (curated), `@_akhaliq` Twitter for AI papers. Submitting: create account, submit LaTeX/PDF, select primary category. Revisions: v2, v3 etc. Peer review: separate, submitted to workshops/conferences (NeurIPS, ICML, ICLR, ACL). arXiv: not peer-reviewed but community-validated.
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase10 #arXiv #research #papers

---

**ID**: P10-010
**Front**: What is the peer review process at major AI conferences (NeurIPS, ICML, ICLR)?
**Back**: Process: (1) Submission deadline (usually Feb/Apr/Sept). (2) 3–4 reviewers assigned. (3) Review period: 6–8 weeks. (4) Author rebuttal: 1–2 weeks to respond. (5) Discussion + final decision. (6) Accept rate: NeurIPS ~25%, ICML ~30%, ICLR ~32% (2024). Review criteria: novelty, correctness, clarity, significance. Strategy: submit to workshops first (lower bar, faster feedback). ACL/EMNLP for NLP-specific work. Double-blind: reviewers and authors anonymous to each other.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase10 #conferences #NeurIPS #peer-review

---

**ID**: P10-011
**Front**: What is the difference between connected papers and semantic scholar for paper research?
**Back**: Connected Papers: visualizes a "graph" of papers related to a seed paper — shows citation connections, clusters of related work. Best for: understanding the field around one paper. Semantic Scholar: academic search engine with AI-powered features — citation counts, influential citations, semantic search, paper summaries. Best for: finding papers on a topic. Use both: Semantic Scholar for discovery, Connected Papers to explore the neighborhood of key papers.
**Difficulty**: Beginner
**Category**: Research
**Tags**: #phase10 #research-tools #papers

---

**ID**: P10-012
**Front**: What does it mean for a paper to be "statistically significant" and why do many AI papers fail this bar?
**Back**: Statistical significance: the probability that the observed improvement is due to chance is below α (usually 0.05). Requires: multiple runs, reporting mean ± std, hypothesis testing. AI papers fail because: (1) Single-run results (no confidence intervals). (2) Small test sets where noise is large. (3) Comparing on benchmarks the model may have seen during training. (4) Picking best checkpoint after many tries. Red flags: "our method achieves X% improvement" with no error bars, evaluated on same test used for tuning.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase10 #statistics #reproducibility

---

**ID**: P10-013
**Front**: How do you implement a paper's method? What are the key steps?
**Back**: (1) Read the paper at least twice — note all details. (2) Check for official code (usually in paper or author's GitHub). (3) Find any re-implementations (Papers with Code). (4) Start with simplest version: get one example working. (5) Scale up to their experimental setup. (6) Reproduce one key result from Table 1/2. (7) Ablate: remove components to understand what matters. (8) If can't reproduce: check with 5% tolerance (floating point, random seed). Write a blog post about what you found.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase10 #paper-implementation #reproducibility

---

**ID**: P10-014
**Front**: What is a "seminal paper" vs a "technical report" in AI research?
**Back**: Seminal paper: introduces a fundamentally new idea, method, or empirical finding. Peer-reviewed (or extremely widely cited preprint). Long-lasting impact. Examples: Attention Is All You Need, LoRA, BERT. Technical report: describes a model or system without necessarily introducing new science. Often from large labs. Not peer-reviewed. Examples: GPT-4 Technical Report, Gemini Technical Report. Both are valuable: technical reports reveal model capabilities and training details. Seminal papers advance the field.
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase10 #papers #research

---

**ID**: P10-015
**Front**: What is Papers with Code and how do you use it effectively?
**Back**: paperswithcode.com: links papers to GitHub implementations + benchmark leaderboards. Features: (1) Method pages (e.g., LoRA) with all related papers and implementations. (2) Task pages (e.g., "Language Modeling") with SOTA leaderboard. (3) Dataset pages. (4) Trending papers (community upvotes). Use for: (1) Finding implementations before writing from scratch. (2) Checking if your idea has been tried. (3) Understanding SOTA for a task. (4) Finding reproducible papers (code available filter).
**Difficulty**: Beginner
**Category**: Research
**Tags**: #phase10 #papers-with-code #research-tools

---

**ID**: P10-016
**Front**: What is the best way to build a public reputation in the AI research community?
**Back**: (1) Twitter/X: share paper summaries (2-3/week), code snippets, findings. (2) Blog posts: deep dives on papers or techniques you implement (medium, substack, github.io). (3) Open source: merge PRs to well-known repos — your name in git history. (4) GitHub: public repos with well-documented projects (README, requirements.txt, demo). (5) arXiv preprints: even workshop papers count. (6) Conference presence: attend (virtually) NeurIPS, ICML, ICLR. Compound over time: consistent > sporadic.
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase10 #community #reputation

---

**ID**: P10-017
**Front**: What is "mechanistic interpretability" in AI research?
**Back**: Mechanistic interpretability: understand the actual computations inside neural networks — not just behavior but implementation. Key findings: attention heads have interpretable roles (induction heads, name mover heads). Circuits: small subgraphs of the network that perform identifiable tasks. Tools: TransformerLens (Neel Nanda). Activation patching: replace activations from one input with another to find which neurons matter. Goal: understand how transformer "knows" facts, does reasoning. Related: superposition hypothesis (neurons encode multiple features).
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase10 #interpretability #mechanistic

---

**ID**: P10-018
**Front**: What is an AI safety evaluation and why does it matter?
**Back**: AI safety eval: systematically test for dangerous capabilities (CBRN knowledge, cyberattacks, deception). Organizations: ARC Evals (now METR), UK AISI, Anthropic RSP. Types: (1) Uplift testing: does the model help bad actors more than Google? (2) Autonomous replication: can it acquire resources, replicate itself? (3) Deception: does it behave differently when it thinks it's being evaluated? Why matters: Anthropic/OpenAI commit to evaluating frontier models before release. Responsible scaling policy.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase10 #AI-safety #evaluation

---

**ID**: P10-019
**Front**: What is the "bitter lesson" (Sutton, 2019) in AI research?
**Back**: Richard Sutton's essay: the two methods that consistently triumph long-term are search and learning (general compute). Not domain knowledge, not clever engineering tricks. Historically: every AI subfield where researchers added human knowledge was eventually surpassed by general methods + more compute. Examples: chess (Deep Blue's handcrafted eval → AlphaZero's self-play), speech recognition (HMMs with phoneme rules → end-to-end deep learning). Implication: invest in scalable methods, not clever heuristics.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase10 #bitter-lesson #AI-history

---

**ID**: P10-020
**Front**: What is the Hugging Face Hub and what can you do with it?
**Back**: Hugging Face Hub: central repository for models, datasets, and spaces. Upload: `model.push_to_hub("username/my-lora-adapter")`. Download: `AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")`. Datasets: `load_dataset("username/my-dataset")`. Spaces: deploy Gradio/Streamlit demos for free. Model cards: document training data, eval results, intended use. For the community: share fine-tuned models, evaluation results, datasets. Gated access: some models (LLaMA) require approval.
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase10 #huggingface #open-source #models

---

**ID**: P10-021
**Front**: What is AlphaCode/Codex and what benchmark does it evaluate on?
**Back**: Codex (OpenAI 2021): GPT-3 fine-tuned on GitHub code — powers GitHub Copilot. AlphaCode (DeepMind 2022): competitive programming-grade code generation. Benchmark: HumanEval (Chen et al. 2021): 164 Python programming problems, functional correctness via unit tests. Metric: pass@k — probability that at least 1 of k samples passes all tests. GPT-4: ~85% pass@1. Latest: DeepSeek-Coder, Claude 3.5 Sonnet competing at 90%+. Code models trained on: GitHub, StackOverflow, documentation.
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase10 #code-generation #HumanEval #AlphaCode

---

**ID**: P10-022
**Front**: What is the difference between a workshop paper and a full conference paper?
**Back**: Full paper: 8–9 pages, main conference (NeurIPS, ICML, ICLR, ACL), peer-reviewed, higher bar, more prestigious. Workshop paper: 4 pages, co-located workshop, less rigorous review, accept rate ~50-60%, faster turnaround. Workshop strategy: (1) Get feedback on in-progress work. (2) Build community connections. (3) First publication credit. (4) Lower bar to start your research career. Many full papers started as workshop papers. Both appear on arXiv.
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase10 #conferences #workshops #research

---

**ID**: P10-023
**Front**: What is the DeepSeek-R1 paper's key contribution?
**Back**: DeepSeek-R1 (2025): open-source chain-of-thought reasoning model matching or exceeding o1. Key contributions: (1) GRPO training (Group Relative Policy Optimization) — PPO without a critic model. (2) "Aha moment" emergence: model spontaneously developed self-reflection during RL training. (3) Showed: pure RL (without SFT first) can develop reasoning. (4) Fully open: model weights + training details released. Impact: democratized reasoning model training; showed chain-of-thought reasoning is learnable via RL.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase10 #DeepSeek #reasoning #GRPO

---

**ID**: P10-024
**Front**: What is an ablation study in ML research and why is it important?
**Back**: Ablation: systematically remove/disable components to understand their individual contribution. Example: "We tested our RAG system with and without: (1) reranker, (2) hybrid search, (3) query expansion. Table 3 shows each component's contribution." Why important: (1) Proves each component adds value (no free riders). (2) Helps practitioners know what to implement. (3) Validates the paper's design decisions. (4) Reviewer requirement for good papers. Red flag: no ablations = can't trust claimed improvements.
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase10 #ablation #research-methodology

---

**ID**: P10-025
**Front**: What does "reproducibility crisis" mean in AI research and how do you address it?
**Back**: Many AI papers can't be reproduced: different results with different GPUs, seeds, library versions. Causes: (1) Unlisted hyperparameters. (2) Different random seeds. (3) Library version differences. (4) Underreporting of failed experiments. (5) Cherry-picked results. How to ensure your work is reproducible: (1) Pin all dependencies (`requirements.txt` with exact versions). (2) Set all random seeds. (3) Report mean ± std across 3+ runs. (4) Provide code (mandatory for NeurIPS/ICML since 2022). (5) Use `reproducibility.md` in repo.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase10 #reproducibility #research
