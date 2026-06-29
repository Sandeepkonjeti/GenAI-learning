# Roadmap — Future Development

> Planned improvements for this curriculum. Contributions welcome — see [CONTRIBUTING.md](./CONTRIBUTING.md).

---

## Status Key

- 🔴 High priority — significantly improves learning outcomes
- 🟡 Medium priority — adds value but not blocking
- 🟢 Low priority / nice to have
- ✅ Done

---

## Near-Term (Next 3 Months)

### Content Gaps (from AI_Curriculum_Quality_Audit.md)

| Priority | Gap | Phase | Notes |
|----------|-----|-------|-------|
| 🔴 | Constitutional AI / RLAIF | 7b | Anthropic's alternative to RLHF from human feedback; increasingly common |
| 🔴 | KTO (Kahneman-Tversky Optimization) | 7b | Alternative to DPO, requires only binary feedback |
| 🔴 | Prefix caching / prompt caching | 9 | Anthropic and OpenAI both offer this; significant cost reduction |
| 🔴 | AI safety evaluation (OWASP LLM Top 10) | 9 | LLM01–LLM10 walkthrough with mitigations |
| 🟡 | Mixture of Experts deep dive | 8 | Mistral-8x7B, GPT-4 speculation; MoE is the dominant architecture |
| 🟡 | State Space Models (Mamba) | 8 | Linear attention alternative; gaining traction |
| 🟡 | Multimodal RAG patterns | 8 | Images + text retrieval; ColPali, LLaVA for document parsing |
| 🟡 | Structured generation (outlines, guidance) | 5a | Grammar-constrained generation for guaranteed output format |
| 🟡 | Tool call parallelism | 6 | Parallel tool execution in agent loops |
| 🟡 | Batch inference patterns | 9 | Async batching for throughput; vLLM batch API |
| 🟢 | GRPO (Group Relative Policy Optimization) | 7b | DeepSeek-R1 training method |
| 🟢 | Long-context techniques (LongRoPE, YaRN) | 4b | Extending context beyond training length |
| 🟢 | Flash Attention 3 details | 4b | Hardware-aware attention; IO complexity |

---

## Medium-Term (3–6 Months)

### Phase 11: AI Safety and Interpretability (Proposed)

A new phase covering:
- Mechanistic interpretability (circuits, attention heads, superposition)
- Activation steering and representation engineering
- Jailbreak taxonomy and red-teaming methodology
- Constitutional AI and debate as scalable oversight
- AI risk frameworks (ARC, METR evaluations)
- Interpretability tools: TransformerLens, Baukit

**Prerequisite**: Phase 9 complete  
**Estimated scope**: 6 weeks | 15–20 pages

---

### `AI_Projects_Roadmap.md` — 100 Projects (Planned)

A comprehensive project catalog from beginner to research-grade:
- 100 projects total (current: 24)
- Each with 11 fields: Phase, Difficulty, Time, Technologies, Skills, Resume Value, Interview Value, GitHub Starter Template, Learning Objectives, Stretch Goals, Similar Industry Applications
- Organized by specialization: App Engineer, MLOps, Agent, Fine-Tuning, Research, Multimodal

**Status**: Scoped, not yet created  
**Priority**: 🟡 — valuable but not blocking core learning

---

### Interactive Exercises

Convert "try it yourself" sections in phase files into Jupyter notebooks:
- `exercises/Phase4b_Transformer.ipynb` — implement attention from scratch with unit tests
- `exercises/Phase5b_RAG_Eval.ipynb` — build RAGAS evaluation pipeline step by step
- `exercises/Phase7a_LoRA.ipynb` — implement LoRA adapter layer from scratch

---

## Long-Term (6+ Months)

### Community Features

- Discussion templates for each phase (GitHub Discussions)
- "Study buddy" matching system for learners at the same phase
- Monthly "build challenge" prompts with example implementations

### Automation

- GitHub Action to validate all code blocks compile/run
- Anki deck auto-export from `Flashcards/` markdown files
- Progress tracker web UI (simple static site reading PROGRESS_TRACKER.md)

### Curriculum Expansion

| Proposed Module | Target | Priority |
|----------------|--------|----------|
| AI for Data Engineers (Spark + LLMs) | Databricks users | 🔴 |
| LLM Evaluation Deep Dive | Production engineers | 🟡 |
| Enterprise AI Patterns | Senior engineers | 🟡 |
| AI Security Specialist Track | Security engineers | 🟢 |

---

## Known Issues

| Issue | File | Priority |
|-------|------|----------|
| Phase 8 multimodal section lacks CLIP implementation | `09_Phase8_Advanced_Topics.md` | 🟡 |
| Phase 3b CNN section references torchvision but import not shown | `04_Phase3_Part2_PyTorch.md` | 🟢 |
| Resources/Courses.md fast.ai course URL should be verified | `Resources/Courses.md` | 🟢 |

---

*[← Index](./INDEX.md) | [Changelog](./CHANGELOG.md) | [Contributing](./CONTRIBUTING.md)*
