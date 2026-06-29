# GitHub Repositories

[← Back to INDEX](../INDEX.md) | [Books](./Books.md) | [Papers](./ResearchPapers.md) | [Communities](./Communities.md)

Curated GitHub repositories for the GenAI Engineer learning path — for study, reference, and contribution.

---

## Study From Scratch

### micrograd (Karpathy)
| Field | Value |
|-------|-------|
| **URL** | https://github.com/karpathy/micrograd |
| **Why useful** | 100-line autograd engine — read the full code to genuinely understand backpropagation; every GenAI engineer should read this once |
| **Difficulty** | Intermediate |
| **Prerequisites** | Python, calculus |
| **Best time to use** | Phase 3 |
| **Estimated time** | 3–5 hours |
| **Free/Paid** | Free |
| **Actively maintained** | No (stable reference, intentionally minimal) |

### nanoGPT (Karpathy)
| Field | Value |
|-------|-------|
| **URL** | https://github.com/karpathy/nanoGPT |
| **Why useful** | Cleanest GPT-2 implementation (~300 lines of core code); read before studying larger transformers; used as Project 10 in this curriculum |
| **Difficulty** | Advanced |
| **Prerequisites** | PyTorch, Phase 3–4 content |
| **Best time to use** | Phase 4 |
| **Estimated time** | 5–8 hours to read and run |
| **Free/Paid** | Free |
| **Actively maintained** | No (stable reference) |

---

## Production Libraries (Use & Study)

### HuggingFace Transformers
| Field | Value |
|-------|-------|
| **URL** | https://github.com/huggingface/transformers |
| **Why useful** | The standard library for working with LLMs — 100,000+ stars; study the source to understand how AutoModel, pipeline, and generation actually work |
| **Difficulty** | Advanced |
| **Prerequisites** | PyTorch, Phase 4 content |
| **Best time to use** | Phases 4–7 |
| **Estimated time** | Reference (ongoing) |
| **Free/Paid** | Free |
| **Actively maintained** | Yes (multiple commits/day) |

### PEFT (HuggingFace)
| Field | Value |
|-------|-------|
| **URL** | https://github.com/huggingface/peft |
| **Why useful** | Official LoRA, QLoRA, prefix tuning, prompt tuning implementation — read the LoRA source to understand how adapters hook into the model |
| **Difficulty** | Advanced |
| **Prerequisites** | HuggingFace Transformers |
| **Best time to use** | Phase 7 |
| **Estimated time** | Reference |
| **Free/Paid** | Free |
| **Actively maintained** | Yes |

### TRL (HuggingFace)
| Field | Value |
|-------|-------|
| **URL** | https://github.com/huggingface/trl |
| **Why useful** | SFTTrainer, DPOTrainer, PPOTrainer — the standard RLHF/alignment library; study examples for Phase 7 projects |
| **Difficulty** | Advanced |
| **Prerequisites** | PEFT, HuggingFace Transformers |
| **Best time to use** | Phase 7 |
| **Estimated time** | Reference |
| **Free/Paid** | Free |
| **Actively maintained** | Yes |

### vLLM
| Field | Value |
|-------|-------|
| **URL** | https://github.com/vllm-project/vllm |
| **Why useful** | Fastest LLM inference server — PagedAttention, continuous batching; study the engine architecture; 25,000+ stars |
| **Difficulty** | Advanced |
| **Prerequisites** | PyTorch, CUDA basics |
| **Best time to use** | Phases 8–9 |
| **Estimated time** | Reference |
| **Free/Paid** | Free |
| **Actively maintained** | Yes (daily commits) |

---

## LLM Applications (Build & Study)

### LangChain
| Field | Value |
|-------|-------|
| **URL** | https://github.com/langchain-ai/langchain |
| **Why useful** | Most widely used LLM application framework — chains, memory, tools, agents; study `langchain_core` for clean abstractions |
| **Difficulty** | Intermediate |
| **Prerequisites** | Python, LLM APIs |
| **Best time to use** | Phases 5–6 |
| **Estimated time** | Reference |
| **Free/Paid** | Free |
| **Actively maintained** | Yes |

### LangGraph
| Field | Value |
|-------|-------|
| **URL** | https://github.com/langchain-ai/langgraph |
| **Why useful** | State machine framework for AI agents — used in Phase 6 projects; study examples to understand stateful agent patterns |
| **Difficulty** | Advanced |
| **Prerequisites** | LangChain |
| **Best time to use** | Phase 6 |
| **Estimated time** | 5–10 hours to study examples |
| **Free/Paid** | Free |
| **Actively maintained** | Yes |

### instructor
| Field | Value |
|-------|-------|
| **URL** | https://github.com/instructor-ai/instructor |
| **Why useful** | Best structured output library — Pydantic + LLM APIs; compact codebase (~3000 lines), great to understand for contributing |
| **Difficulty** | Intermediate |
| **Prerequisites** | Pydantic, OpenAI API |
| **Best time to use** | Phase 5 |
| **Estimated time** | 2–3 hours to study |
| **Free/Paid** | Free |
| **Actively maintained** | Yes |

### RAGAS
| Field | Value |
|-------|-------|
| **URL** | https://github.com/explodinggradients/ragas |
| **Why useful** | RAG evaluation framework — faithfulness, relevance, context recall; relatively small codebase, excellent for first OSS contribution |
| **Difficulty** | Intermediate |
| **Prerequisites** | Python, RAG basics |
| **Best time to use** | Phases 5, 9 |
| **Estimated time** | Reference |
| **Free/Paid** | Free |
| **Actively maintained** | Yes |

### LiteLLM
| Field | Value |
|-------|-------|
| **URL** | https://github.com/BerriAI/litellm |
| **Why useful** | Unified interface for 100+ LLM providers — swap models without code changes; study for fallback and model routing patterns |
| **Difficulty** | Intermediate |
| **Prerequisites** | LLM API basics |
| **Best time to use** | Phase 9 |
| **Estimated time** | Reference |
| **Free/Paid** | Free |
| **Actively maintained** | Yes |

---

## Data & Infrastructure

### Qdrant
| Field | Value |
|-------|-------|
| **URL** | https://github.com/qdrant/qdrant |
| **Why useful** | Production-ready vector database — well-documented, actively maintained, clean Python client; used as primary vector DB in this curriculum |
| **Difficulty** | Intermediate |
| **Prerequisites** | Embeddings basics |
| **Best time to use** | Phases 5–9 |
| **Estimated time** | Reference |
| **Free/Paid** | Free (open source) |
| **Actively maintained** | Yes |

### MLflow
| Field | Value |
|-------|-------|
| **URL** | https://github.com/mlflow/mlflow |
| **Why useful** | Standard MLops platform — tracking, model registry, serving; built into Databricks; study LLM evaluation and pyfunc modules |
| **Difficulty** | Intermediate |
| **Prerequisites** | ML basics |
| **Best time to use** | Phase 9 |
| **Estimated time** | Reference |
| **Free/Paid** | Free |
| **Actively maintained** | Yes |
