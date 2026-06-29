# Changelog

All significant changes to this curriculum are documented here.

Format: `## [version] — YYYY-MM-DD` | Added / Changed / Fixed / Removed

---

## [1.3.0] — 2026-06

### Added
- `INDEX.md` — Master navigation hub with all file links, learning paths, and quick reference tables
- `PROGRESS_TRACKER.md` — Comprehensive learning dashboard with weekly/monthly logs, habit tracker, confidence ratings, quiz scores, mistakes log, interview prep, and final readiness dashboard
- `PROJECT_INDEX.md` — Complete catalog of all 24 projects with resume value, interview value, time estimates, and portfolio strategy
- `LEARNING_GRAPH.md` — Mermaid dependency diagrams for core path and 5 alternative learning paths (LLM App Engineer, MLOps, Agent Specialist, Fine-tuning Specialist, Research)
- `CONTRIBUTING.md` — Style guide, quality bar, code conventions, flashcard format, resource entry format, PR process
- `CheatSheets/` folder — 11 phase cheat sheets for 15–30 minute revision sessions
- `Flashcards/` folder — 11 phase flashcard files in Anki-compatible format with Python export script
- `Resources/` folder — 9 curated resource guides (Books, Courses, YouTube, GitHub, Papers, Blogs, Newsletters, Communities, Podcasts)
- `ROADMAP.md` — Future curriculum development plan
- `AI_Curriculum_Quality_Audit.md` — Full audit document scoring all 15 phases with gap analysis

### Changed
- Phase 4b: Added GQA/MQA section with PyTorch implementation (modern LLMs — LLaMA-3, Mistral — use GQA; was completely missing)
- Phase 5a: Added "Production Structured Outputs with instructor" section (instructor library, Pydantic validation, retry logic)
- Phase 5b: Added 3 advanced RAG patterns (Corrective RAG, Contextual Retrieval, Retrieval Quality Metrics with MRR/Precision@k)
- Phase 6: Added Part F — Multi-Agent Systems (supervisor-worker pattern, loop detection, parallel execution, handoff protocol)
- Phase 7a: Added "Distributed and Multi-GPU Training" section (DDP, DeepSpeed ZeRO, Databricks TorchDistributor)
- Phase 9: Added Topic 6 — Databricks AI Engineering (Lakehouse AI architecture, Model Serving, Mosaic AI Vector Search, MLflow LLM tracking, AI Gateway)
- Phase 9: Added comprehensive LLM Benchmark Map (MMLU, HumanEval, BIG-bench, HELM, MT-Bench, LMSYS, GSM8K, TruthfulQA)
- Phase 9: Added FastAPI production deployment section (streaming response, auth middleware, rate limiting, TTFT/TPOT metrics)

### Fixed
- Phase 6: Replaced `eval()` with `simpleeval` in calculator tool (security — eval exploitable via prompt injection)
- Phase 7b: Corrected RLHF label from 🟡 "Learn Later" to 🔴 "Must Learn" (RLHF is foundational for understanding InstructGPT/ChatGPT)
- Phase 6: Added Text-to-SQL Agent (Project 17b) with error-correction loop

---

## [1.2.0] — 2026-06

### Added
- Quality audit performed across all 15 phases
- Identified 6 critical gaps, 7 high-priority gaps, 7 medium gaps, 6 low-priority gaps
- Scored curriculum: Technical Depth 85%, Practical 72%, Production 68%, Interview 80%
- All critical and high-priority fixes applied (see v1.3.0 Changed/Fixed above)

---

## [1.1.0] — 2026-06

### Added
- Phase Completion & Readiness Assessments appended to all 15 phase files
- Each assessment includes: 25-question self-assessment quiz, readiness checklist, prerequisite gate (must score ≥22/25 to proceed), key skills validation, and what-to-review-if-stuck guide

---

## [1.0.0] — 2026-06

### Added
- Initial 18-month GenAI Engineer curriculum (Phases 0–10, 15 phase files)
- `01_Phase0_CS_Fundamentals_and_Python.md`
- `02_Phase1_Mathematics_for_ML.md`
- `03_Phase2_Machine_Learning.md`
- `04_Phase3_Part1_Deep_Learning_Theory.md`
- `04_Phase3_Part2_PyTorch.md`
- `05_Phase4_Part1_NLP_Fundamentals.md`
- `05_Phase4_Part2_Transformers.md`
- `06_Phase5_Part1_LLMs_and_Prompt_Engineering.md`
- `06_Phase5_Part2_Embeddings_VectorDB_RAG.md`
- `07_Phase6_AI_Agents_and_MCP.md`
- `08_Phase7_Part1_FineTuning_and_LoRA.md`
- `08_Phase7_Part2_RLHF_and_DPO.md`
- `09_Phase8_Advanced_Topics.md`
- `10_Phase9_Production_AI_Engineering.md`
- `11_Phase10_Research_and_OpenSource.md`
- `12_Projects_Beginner_to_Expert.md` — 22-project portfolio guide
- `13_Study_Schedule_18_Months.md` — Day-by-day/week-by-week schedule
- `14_Interview_Preparation.md` — Comprehensive interview prep guide
- `15_Career_Roadmap.md` — Career progression guide

---

*[← Index](./INDEX.md)*
