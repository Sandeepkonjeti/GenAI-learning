# Research Papers

[← Back to INDEX](../INDEX.md) | [Books](./Books.md) | [GitHub Repos](./GitHubRepositories.md) | [Blogs](./Blogs.md)

Must-read research papers for the GenAI Engineer learning path, organized by phase.

**Reading tip:** Use the 3-pass method. Read abstract + conclusion first to decide if worth your time.  
**Finding papers:** [Papers with Code](https://paperswithcode.com) links papers to implementations.

---

## Phase 4 — Transformers

### Attention Is All You Need (2017)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/1706.03762 |
| **Why useful** | Foundation of modern AI — introduced transformers, self-attention, positional encoding; you cannot call yourself an AI engineer without reading this |
| **Difficulty** | Intermediate |
| **Prerequisites** | Basic RNN understanding |
| **Best time to use** | Phase 4 |
| **Estimated time** | 3–5 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2017) |

### BERT: Pre-training of Deep Bidirectional Transformers (2018)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/1810.04805 |
| **Why useful** | Introduced masked language modeling (MLM), bidirectional context; the paper that made transformers the default for NLP; still referenced constantly |
| **Difficulty** | Intermediate |
| **Prerequisites** | Attention Is All You Need |
| **Best time to use** | Phase 4 |
| **Estimated time** | 2–3 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2018) |

### Flash Attention (2022) & Flash Attention 2 (2023)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/2205.14135 |
| **Why useful** | IO-aware attention — O(n) memory instead of O(n²); standard in all modern LLMs; read to understand GPU memory hierarchy and why it matters for long context |
| **Difficulty** | Advanced |
| **Prerequisites** | Transformer architecture, CUDA basics helpful |
| **Best time to use** | Phases 4, 8 |
| **Estimated time** | 3–4 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (paper) |

---

## Phase 5 — LLMs & RAG

### GPT-3: Language Models Are Few-Shot Learners (2020)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/2005.14165 |
| **Why useful** | Demonstrated emergent few-shot learning at scale; established prompting as a paradigm; scale → capability, not just parameter → task fine-tuning |
| **Difficulty** | Intermediate |
| **Prerequisites** | Transformer basics |
| **Best time to use** | Phase 5 |
| **Estimated time** | 2–3 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2020) |

### Chain-of-Thought Prompting Elicits Reasoning (2022)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/2201.11903 |
| **Why useful** | Shows that "think step by step" dramatically improves LLM reasoning on math and logic; CoT is now a standard technique in every LLM system |
| **Difficulty** | Beginner |
| **Prerequisites** | LLM basics |
| **Best time to use** | Phase 5 |
| **Estimated time** | 1–2 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2022) |

### RAG: Retrieval-Augmented Generation (2020)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/2005.11401 |
| **Why useful** | Introduced the RAG paradigm — combine retrieval with generative models; the original paper behind the most widely deployed LLM architecture pattern |
| **Difficulty** | Intermediate |
| **Prerequisites** | Transformer basics |
| **Best time to use** | Phase 5 |
| **Estimated time** | 2–3 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2020) |

### Scaling Laws for Neural Language Models (2020)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/2001.08361 |
| **Why useful** | Shows compute, data, parameters scale predictably — power laws; foundation for understanding LLM training costs and why Chinchilla changed everything |
| **Difficulty** | Advanced |
| **Prerequisites** | ML fundamentals |
| **Best time to use** | Phase 5, 10 |
| **Estimated time** | 2–3 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2020) |

---

## Phase 6 — Agents

### Toolformer: Language Models Can Teach Themselves to Use Tools (2023)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/2302.04761 |
| **Why useful** | Showed LLMs can learn to call APIs via self-supervised training; conceptual foundation for modern tool-using agents |
| **Difficulty** | Intermediate |
| **Prerequisites** | GPT-3 basics |
| **Best time to use** | Phase 6 |
| **Estimated time** | 2–3 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2023) |

### ReAct: Synergizing Reasoning and Acting in Language Models (2022)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/2210.03629 |
| **Why useful** | Introduced the ReAct pattern (Thought → Action → Observation) that is the basis for most AI agents today; short and highly readable |
| **Difficulty** | Intermediate |
| **Prerequisites** | CoT prompting |
| **Best time to use** | Phase 6 |
| **Estimated time** | 1–2 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2022) |

---

## Phase 7 — Fine-Tuning & Alignment

### InstructGPT: Training Language Models to Follow Instructions with Human Feedback (2022)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/2203.02155 |
| **Why useful** | RLHF for alignment — shows 1.3B InstructGPT beats 175B GPT-3 on helpfulness; blueprint for ChatGPT; essential reading for anyone doing alignment |
| **Difficulty** | Intermediate |
| **Prerequisites** | GPT-3 paper |
| **Best time to use** | Phase 7 |
| **Estimated time** | 2–3 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2022) |

### LoRA: Low-Rank Adaptation of Large Language Models (2021)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/2106.09685 |
| **Why useful** | Enables fine-tuning 175B+ models with < 1% of parameters; read the math — it's clean and well-explained; used in virtually every fine-tuning workflow today |
| **Difficulty** | Intermediate |
| **Prerequisites** | Matrix algebra, transformer fine-tuning concepts |
| **Best time to use** | Phase 7 |
| **Estimated time** | 2–3 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2021) |

### QLoRA: Efficient Finetuning of Quantized LLMs (2023)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/2305.14314 |
| **Why useful** | Fine-tune 65B parameter models on a single 48GB GPU via 4-bit quantization + LoRA; introduced NF4, double quantization, paged Adam; changed what's possible on consumer hardware |
| **Difficulty** | Advanced |
| **Prerequisites** | LoRA paper |
| **Best time to use** | Phase 7 |
| **Estimated time** | 3–4 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2023) |

### Direct Preference Optimization (DPO) (2023)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/2305.18290 |
| **Why useful** | Alignment without a reward model — directly optimize on preference pairs; cleaner than RLHF/PPO; increasingly the standard for alignment |
| **Difficulty** | Advanced |
| **Prerequisites** | InstructGPT, RL basics |
| **Best time to use** | Phase 7 |
| **Estimated time** | 3–4 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2023) |

### Constitutional AI: Harmlessness from AI Feedback (2022)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/2212.08073 |
| **Why useful** | RLAIF — use AI feedback instead of human labeling for alignment; Claude's training technique; shows AI can critique and improve its own outputs |
| **Difficulty** | Advanced |
| **Prerequisites** | InstructGPT |
| **Best time to use** | Phase 7, 10 |
| **Estimated time** | 2–3 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2022) |

---

## Phase 8 — Advanced Topics

### Mixtral of Experts (2024)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/2401.04088 |
| **Why useful** | Open weights MoE model — outperforms LLaMA-2-70B at lower inference cost; the technical report explains MoE routing and load balancing clearly |
| **Difficulty** | Advanced |
| **Prerequisites** | Transformer architecture |
| **Best time to use** | Phase 8 |
| **Estimated time** | 2 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2024) |

### Mamba: Linear-Time Sequence Modeling with Selective State Spaces (2023)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/2312.00752 |
| **Why useful** | O(n) alternative to transformers — selective state space model; may matter if context windows get very large; important for Phase 10 research frontier |
| **Difficulty** | Advanced |
| **Prerequisites** | RNN/SSM basics, transformer architecture |
| **Best time to use** | Phase 8, 10 |
| **Estimated time** | 4–5 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2023) |

### LLaVA: Visual Instruction Tuning (2023)
| Field | Value |
|-------|-------|
| **URL** | https://arxiv.org/abs/2304.08485 |
| **Why useful** | Connects CLIP vision encoder to LLaMA via projection layer; shows multimodal instruction tuning with GPT-4 generated data; foundational for multimodal agents |
| **Difficulty** | Advanced |
| **Prerequisites** | LoRA, CLIP basics |
| **Best time to use** | Phase 8 |
| **Estimated time** | 2–3 hours |
| **Free/Paid** | Free |
| **Actively maintained** | N/A (2023) |
