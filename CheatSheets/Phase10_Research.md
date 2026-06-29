# Phase 10 — Research & Open Source Cheat Sheet

[← CheatSheets Index](./README.md) | [Full Phase File](../11_Phase10_Research_and_OpenSource.md)

---

## Paper Reading Framework (3-Pass Method)

### Pass 1 — Skim (10 minutes)
Read: title, abstract, intro, conclusion, headings, figures
Answer: What problem? What approach? What result? Should I read fully?

### Pass 2 — Understand (1–2 hours)
Read everything except proofs. Note: key equations, experimental setup, main tables.
Answer: How does the method work? What are the assumptions? Are results convincing?

### Pass 3 — Critique (2–4 hours)
Read proofs, check references, try to reproduce key result.
Answer: What are the limitations? What would you do differently? What follow-on work is possible?

---

## Must-Read Papers by Phase

| Paper | Year | Why Essential |
|-------|:----:|--------------|
| Attention Is All You Need | 2017 | Original transformer |
| BERT | 2018 | Bidirectional pretraining |
| GPT-3 | 2020 | Scaling laws, few-shot |
| InstructGPT | 2022 | RLHF, alignment |
| LoRA | 2021 | Efficient fine-tuning |
| QLoRA | 2023 | 4-bit fine-tuning |
| Constitutional AI | 2022 | RLHF alternative |
| Toolformer | 2023 | LLMs learning to use tools |
| RAG | 2020 | Retrieval-augmented generation |
| Direct Preference Optimization | 2023 | DPO |
| Chain-of-Thought Prompting | 2022 | Reasoning |
| Scaling Laws | 2020 | How LLMs scale |

---

## How to Find Papers

| Source | URL | Best For |
|--------|-----|---------|
| arXiv | arxiv.org/list/cs.AI/recent | Preprints (first access) |
| Papers with Code | paperswithcode.com | Code + benchmarks |
| Semantic Scholar | semanticscholar.org | Citation graphs |
| Hugging Face Papers | huggingface.co/papers | Curated daily digest |
| Connected Papers | connectedpapers.com | Related work discovery |

---

## Open Source Contribution Path

```
1. Use the library (identify friction points)
    ↓
2. Open issues (bugs you find, features you want)
    ↓
3. Fix documentation (typos, outdated examples)
    ↓
4. Fix small bugs (good first issue label)
    ↓
5. Implement features (after discussion in issue)
    ↓
6. Review others' PRs
    ↓
7. Become a regular contributor → maintainer
```

---

## Key Open Source Projects to Contribute To

| Project | Language | Difficulty | Impact |
|---------|----------|:----------:|--------|
| LangChain / LangGraph | Python | Medium | Very High |
| vLLM | Python/C++ | Hard | Very High |
| HuggingFace Transformers | Python | Medium | Very High |
| RAGAS | Python | Easy–Medium | High |
| LiteLLM | Python | Easy | High |
| Ollama | Go | Hard | High |
| instructor | Python | Easy | Medium |

---

## Research Frontier (2025–2026)

| Area | Key Question | Papers to Track |
|------|-------------|----------------|
| Reasoning models | Chain-of-thought vs. GRPO (DeepSeek-R1) | DeepSeek-R1, o1 report |
| Long context | Scaling to 1M+ tokens efficiently | Gemini 1.5 report |
| Multimodal | Beyond image+text (video, audio, robot) | GPT-4o system card |
| Agentic | Autonomous long-horizon tasks | AGENTBENCH |
| Alignment | Scalable oversight, interpretability | Anthropic alignment research |
| Efficiency | MoE, SSMs, linear attention | Mamba2, RWKV-6 |

---

## Paper Critique Checklist

- [ ] Are the baselines fair? (same compute budget, same data)
- [ ] Is the evaluation dataset held out or used for tuning?
- [ ] What are the failure modes? (the paper will hide them)
- [ ] Can results be explained by a simpler hypothesis?
- [ ] Is variance reported? (single run results are unreliable)
- [ ] Are ablations comprehensive? (what component actually matters)
- [ ] Can I reproduce the key result?

---

## Building a Research Reputation

| Action | Frequency | Platform |
|--------|:---------:|---------|
| Paper summaries | 2/week | Twitter/X, LinkedIn |
| Blog posts | 1/month | Medium, personal site |
| Code reproductions | 1/quarter | GitHub |
| OSS contributions | ongoing | GitHub |
| Conference papers | 1/year (stretch) | arXiv, NeurIPS |

---

## Interview Quick-Hits

**Q: How do you stay current with AI research?**  
A: Hugging Face Papers daily digest (curated top papers). Papers with Code for reproducible results. Twitter/X following key researchers. Reading 2 papers/week deeply rather than skimming 20.

**Q: What is your process for reading a new paper?**  
A: 3-pass method: (1) 10-min skim to decide relevance, (2) 1-2 hour full read except proofs, (3) implement/reproduce key result. I take notes in a research journal.

**Q: Why contribute to open source if you're at a company?**  
A: (1) Reputation and visibility in the community. (2) Forces you to deeply understand libraries. (3) Builds relationships with maintainers. (4) Companies trust engineers who can navigate complex open source codebases.

**Q: What's the most important AI paper of the last 5 years?**  
A: Strong answers: InstructGPT (2022) — demonstrated that RLHF makes LLMs actually useful; LoRA (2021) — democratized fine-tuning. Weak answer: "Attention is All You Need" — too old and obvious a pick.
