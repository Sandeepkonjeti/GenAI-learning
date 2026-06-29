# Phase 7 Flashcards — Fine-Tuning, RLHF & DPO

[← Flashcards Index](./README.md) | [Phase 7 Files](../08_Phase7_Part1_FineTuning_and_LoRA.md) | [Cheat Sheet](../CheatSheets/Phase7_FineTuning.md)

---

**ID**: P7-001
**Front**: Write the LoRA update rule mathematically. What are r, α, B, and A?
**Back**: W' = W₀ + ΔW = W₀ + (α/r) · B · A. W₀ ∈ R^(d×k): frozen pretrained weight. B ∈ R^(d×r): learned low-rank matrix. A ∈ R^(r×k): learned low-rank matrix. r: rank (typically 4–64). α: scaling factor (often = r; final scale = α/r = 1). Trainable params: r(d+k) vs dk — e.g., r=16, d=k=4096 → 131K vs 16.7M — 128× smaller.
**Difficulty**: Advanced
**Category**: Math
**Tags**: #phase7 #LoRA #math #fine-tuning

---

**ID**: P7-002
**Front**: What is QLoRA and how does it differ from LoRA?
**Back**: QLoRA (Dettmers et al. 2023): combine 4-bit quantization with LoRA. Quantize base model to NF4 (Normal Float 4-bit) — reduces memory from 16GB to ~4GB for 7B model. LoRA adapters trained in bfloat16. Key techniques: (1) NF4 quantization (information-optimal for Gaussian-distributed weights). (2) Double quantization (quantize the quantization constants). (3) Paged optimizer (handle GPU memory spikes). Enables: 70B model on 2×A100, 7B on consumer GPU.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase7 #QLoRA #quantization #fine-tuning

---

**ID**: P7-003
**Front**: Which target modules do you typically use for LoRA in LLaMA models?
**Back**: LLaMA attention projections: `q_proj, v_proj` (minimum — most impactful). More comprehensive: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (attention + FFN). How to check: `model.named_modules()` to find all Linear layers. Rule: apply LoRA to the largest Linear layers. LLaMA-3-8B: d_model=4096, so each projection is 4096×4096=16.7M params. With r=16: only 131K trainable per layer.
**Difficulty**: Advanced
**Category**: Coding
**Tags**: #phase7 #LoRA #target-modules #LLaMA

---

**ID**: P7-004
**Front**: What is the RLHF pipeline? Describe all three phases.
**Back**: Phase 1 — SFT (Supervised Fine-Tuning): fine-tune pretrained LLM on (prompt, high-quality response) pairs → SFT model. Phase 2 — Reward Model: train a classifier on (prompt, chosen response, rejected response) pairs to predict human preference → RM. Phase 3 — PPO: optimize SFT model using RM signals. Objective: max_π E[r(x,y)] - β·KL(π||π_SFT). β: KL penalty to prevent reward hacking. This produced InstructGPT/ChatGPT.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase7 #RLHF #PPO #alignment

---

**ID**: P7-005
**Front**: What is the KL penalty in RLHF and what happens if β is too high or too low?
**Back**: KL penalty: β·D_KL(π_θ||π_ref) added to reward objective. Forces fine-tuned model to stay close to SFT base. Too high β: model barely changes from SFT → poor instruction following. Too low β: reward hacking — model finds adversarial ways to maximize reward without being truly helpful (e.g., very long repetitive responses). β typically: 0.01–0.1. Increasing β during training if reward hacking is detected.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase7 #RLHF #KL-penalty

---

**ID**: P7-006
**Front**: What is DPO (Direct Preference Optimization) and what is its key innovation?
**Back**: DPO (Rafailov et al. 2023): eliminates the explicit reward model. Directly optimizes on preference pairs. Loss: -E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]. The model implicitly learns a reward function during training. Why revolutionary: (1) No RM training step. (2) More stable than PPO. (3) Simpler pipeline. (4) Same quality as RLHF. Preferred for production fine-tuning.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase7 #DPO #alignment #preference-learning

---

**ID**: P7-007
**Front**: What does a preference dataset look like for DPO training?
**Back**: Format: `{"prompt": str, "chosen": str, "rejected": str}`. Prompt: the input. Chosen: human-preferred response. Rejected: less preferred response. Sources: (1) Human labelers rank two model outputs. (2) Use strong model (GPT-4) to grade and pick preferred. (3) Use rule-based selection (e.g., correct code chosen, buggy code rejected). Example datasets: Anthropic HH-RLHF, UltraFeedback, OpenAssistant. Quality > quantity for preference data.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase7 #DPO #dataset #preference

---

**ID**: P7-008
**Front**: What is the `SFTTrainer` from TRL and what dataset format does it expect?
**Back**: TRL (Transformer Reinforcement Learning) library from HuggingFace. SFTTrainer: supervised fine-tuning with automatic chat template formatting. Expected format: (1) Raw text: `{"text": "..."}`. (2) Instruction format: `{"instruction": "...", "response": "..."}`. (3) Conversations: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`. `SFTTrainer(model, args, train_dataset, tokenizer, peft_config=lora_config)`.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase7 #TRL #SFTTrainer #fine-tuning

---

**ID**: P7-009
**Front**: What is catastrophic forgetting in fine-tuning and how does LoRA mitigate it?
**Back**: Catastrophic forgetting: when fine-tuning on task B, model "forgets" task A knowledge (weights overwritten). LoRA mitigates by: (1) Freezing original weights — they can't be overwritten. (2) Only the small adapters are trained. (3) Original knowledge stays in frozen W₀; new task in ΔW = BA. Trade-off: LoRA can still shift behavior via adapter interactions. Full fine-tuning with low LR also mitigates forgetting but less than LoRA.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase7 #catastrophic-forgetting #LoRA

---

**ID**: P7-010
**Front**: What is the difference between full fine-tuning, LoRA, and frozen base + LoRA in terms of what parameters are updated?
**Back**: Full fine-tuning: ALL parameters updated (billions). Expensive, risks catastrophic forgetting. LoRA (default): frozen base + trainable A,B adapters. Only adapters updated. Frozen base + LoRA: explicitly frozen base weights (they have no gradient). Usually LoRA already freezes base, so these are equivalent. QLoRA: frozen quantized base + LoRA adapters in higher precision. PEFT comparison: LoRA < prefix-tuning < adapter layers in parameter count; LoRA generally best quality/param ratio.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase7 #fine-tuning #LoRA #PEFT

---

**ID**: P7-011
**Front**: What is `gradient_checkpointing` in HuggingFace and when should you enable it?
**Back**: `model.gradient_checkpointing_enable()`. Trades compute for memory: instead of caching all activations in forward pass for backward, recompute them during backward. Memory: ~√n of layers (significant). Compute: ~33% slower. Enable when: training on long sequences (>1024 tokens), large batch sizes cause OOM, fine-tuning 70B models. Must set `model.enable_input_require_grads()` when using with PEFT. Default: disabled (memory-hungry but faster).
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase7 #gradient-checkpointing #memory #training

---

**ID**: P7-012
**Front**: What is instruction tuning and how does it differ from pretraining?
**Back**: Pretraining: next-token prediction on raw web text — learns language, facts, reasoning. No specific behavior. Instruction tuning (SFT): fine-tune on (instruction, response) pairs — teaches the model to follow instructions, answer helpfully, use a specific format. Dataset: Alpaca, FLAN, OpenAssistant. Effect: transforms raw LLM into assistant. Without instruction tuning: model completes your prompt (continues text). With: model answers your question.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase7 #instruction-tuning #SFT #pretraining

---

**ID**: P7-013
**Front**: What is the Unsloth library and why use it for fine-tuning?
**Back**: Unsloth: optimized fine-tuning library (LoRA/QLoRA). Claims: 2× faster, 60% less memory vs vanilla PEFT + TRL. Key optimizations: fused kernels, memory-efficient attention, optimized backward pass. `from unsloth import FastLanguageModel`. Supports: LLaMA, Mistral, Phi, Gemma. Free tier: CPU/consumer GPU. Pro: multi-GPU. Use when: budget-constrained, single GPU fine-tuning. TRL is more flexible for custom training loops; Unsloth for standard fine-tuning.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase7 #unsloth #fine-tuning #optimization

---

**ID**: P7-014
**Front**: How do you evaluate a fine-tuned model? What metrics matter?
**Back**: Task-specific: (1) Perplexity on held-out set (lower = better next-token prediction). (2) Task metrics: ROUGE for summarization, exact match for QA, pass@k for code. (3) Human evaluation (gold standard, expensive). (4) LLM-as-judge: GPT-4 rates fine-tuned vs base on helpfulness, accuracy. (5) Benchmark regression: check MMLU, HumanEval didn't drop too much. Key: compare to SFT base AND original base. Side-effect check: did fine-tuning break other capabilities?
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase7 #evaluation #fine-tuning

---

**ID**: P7-015
**Front**: What is the relationship between fine-tuning rank r and quality?
**Back**: Higher r: more expressiveness, more parameters. Diminishing returns after r=64 for most tasks. Lower r: less memory, less overfit risk, sometimes better generalization. Recommended: r=8 for simple style tasks, r=16 for most tasks (default), r=32–64 for complex domain adaptation. Also tune: `lora_alpha = r` (so α/r=1 scale). More important than rank: WHICH layers to apply LoRA to (query+value minimum, all attention+FFN for harder tasks).
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase7 #LoRA #rank #hyperparameters

---

**ID**: P7-016
**Front**: What is reward hacking in RLHF and give a concrete example?
**Back**: Reward hacking: model finds unintended ways to maximize the reward model's score without being genuinely helpful. Example: if reward model is biased toward verbose responses → model generates very long repetitive answers. If biased toward confident tone → model becomes overconfident and wrong. Famous example: OpenAI boat racing game agent learned to spin in circles collecting bonuses instead of finishing the race. Mitigation: (1) Diverse reward modeling. (2) KL penalty. (3) Red-teaming. (4) Ensemble reward models.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase7 #reward-hacking #RLHF

---

**ID**: P7-017
**Front**: What is the chat template in HuggingFace and why does it matter for fine-tuning?
**Back**: Chat template: formats (system, user, assistant) messages into model-specific string. LLaMA-3: `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n...<|eot_id|><|start_header_id|>user<|end_header_id|>...`. Mistral: `[INST] ... [/INST]`. Fine-tuning: MUST use the same template as pretraining — else model doesn't learn to respond correctly. `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)`.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase7 #chat-template #fine-tuning #tokenizer

---

**ID**: P7-018
**Front**: What is DDP (Distributed Data Parallel) and when do you use it?
**Back**: DDP: each GPU gets a full model copy + fraction of the batch. Gradients synced via all-reduce after each backward pass. `model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])`. Use when: model fits in 1 GPU but training is slow — scale to 4/8 GPUs. NOT for: model too large for 1 GPU (use DeepSpeed ZeRO instead). Overhead: gradient sync (AllReduce) communication. Usually scales to ~80% efficiency per GPU.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase7 #DDP #distributed-training

---

**ID**: P7-019
**Front**: What are the three ZeRO stages in DeepSpeed and what does each shard?
**Back**: ZeRO (Zero Redundancy Optimizer): partition model state across GPUs. ZeRO-1: shard optimizer states. ZeRO-2: shard optimizer states + gradients. ZeRO-3: shard optimizer states + gradients + model parameters. ZeRO-3: can train models 10× larger than single GPU can fit. Cost: communication overhead. Typically: ZeRO-2 for ≤30B, ZeRO-3 for 65B+. Databricks TorchDistributor wraps this: `TorchDistributor(num_processes=4, local_mode=False)`.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase7 #DeepSpeed #ZeRO #distributed-training

---

**ID**: P7-020
**Front**: What is PEFT and name three methods it includes?
**Back**: PEFT (Parameter-Efficient Fine-Tuning): HuggingFace library for fine-tuning large models with very few trainable parameters. Methods: (1) LoRA: low-rank decomposition of weight updates. (2) Prefix tuning: prepend learnable "prefix" tokens to each layer's K,V. (3) Prompt tuning: add learnable "soft" prompt tokens (only first layer). (4) IA³: learn element-wise scaling vectors. (5) AdaLoRA: adaptive rank allocation. LoRA is the most widely used; PEFT provides unified API for all.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase7 #PEFT #LoRA #fine-tuning

---

**ID**: P7-021
**Front**: What is Constitutional AI (CAI) and how does it differ from RLHF?
**Back**: Constitutional AI (Anthropic 2022): align models using a set of written principles ("constitution") instead of human labelers. Process: (1) Generate outputs. (2) Ask model to critique outputs against the constitution. (3) Revise based on critique (self-improvement). (4) Use revised outputs for RLAIF (Reinforcement Learning from AI Feedback — AI instead of humans labels preference pairs). Benefits: (1) Scales without human labeling cost. (2) Transparent (published constitution). (3) More consistent than human labelers.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase7 #constitutional-AI #RLAIF #alignment

---

**ID**: P7-022
**Front**: When should you fine-tune vs use RAG vs prompt engineering?
**Back**: Prompt engineering first: cheapest, fastest. RAG for: factual knowledge, private data, updatable content. Fine-tune for: (1) Consistent output format/style. (2) Domain vocabulary (model doesn't know the jargon). (3) Latency (no retrieval step). (4) Confidentiality (can't share KB with API provider). Combine: fine-tune for style/behavior + RAG for knowledge. Rule: prompt engineering → RAG → fine-tuning (increasing effort and cost).
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase7 #fine-tuning #RAG #decision

---

**ID**: P7-023
**Front**: What is KTO (Kahneman-Tversky Optimization)?
**Back**: KTO (Ethayarajh et al. 2024): alignment method requiring only BINARY feedback (good/bad), not pairwise preferences. Based on Kahneman-Tversky prospect theory: humans feel losses more strongly than gains (loss aversion). KTO loss: penalizes when model prefers rejected over chosen, weighted asymmetrically (losses penalized more). Advantage over DPO: much easier to collect binary labels than preference pairs. Fewer annotations needed.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase7 #KTO #alignment #preference

---

**ID**: P7-024
**Front**: How do you merge LoRA adapters into the base model for deployment?
**Back**: `from peft import PeftModel`. `model = PeftModel.from_pretrained(base_model, adapter_path)`. `merged = model.merge_and_unload()`. `merged.save_pretrained("merged_model_path")`. Result: standard HuggingFace model with adapters baked in — no PEFT dependency at inference. Benefits: (1) No adapter overhead at inference. (2) Can be quantized with GPTQ/AWQ. (3) Deploy with vLLM. Do NOT merge if: you need to switch adapters (e.g., different LoRA for different users).
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase7 #LoRA #merge #deployment

---

**ID**: P7-025
**Front**: What is the Databricks `TorchDistributor` and how does it differ from DeepSpeed?
**Back**: `from pyspark.ml.torch.distributor import TorchDistributor`. TorchDistributor: runs PyTorch training distributed across Databricks cluster using Spark. Handles: worker coordination, NCCL setup, checkpoint management. DeepSpeed: optimizer/memory efficiency technique (ZeRO stages). They're complementary: TorchDistributor handles cluster distribution; DeepSpeed handles per-node memory optimization. Use both together for large-scale fine-tuning on Databricks. Key advantage: native Databricks integration (Unity Catalog, MLflow, Delta).
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase7 #databricks #TorchDistributor #distributed

---

**ID**: P7-026
**Front**: What learning rate should you use for LoRA fine-tuning?
**Back**: LoRA LR: higher than full fine-tuning (only adapters trained, not entire model). Typical: 1e-4 to 3e-4. Full fine-tuning transformer: 1e-5 to 5e-5. Why higher: adapters start from zero (random init for A, zero for B) — need to learn faster. With warmup: start at 0, linearly increase to target LR over 100-500 steps. Cosine decay: decay from max LR to 0 over training. Monitor: if loss doesn't decrease in first 100 steps → LR too low; if NaN → LR too high.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase7 #LoRA #learning-rate #fine-tuning

---

**ID**: P7-027
**Front**: What is the Alpaca dataset and why is it famous?
**Back**: Alpaca (Stanford, 2023): 52K instruction-following examples generated using GPT-3 self-instruct method. Used to fine-tune LLaMA-7B → Alpaca-7B. Showed: fine-tuning a 7B model on 52K synthetic examples makes it competitive with early GPT-3.5. Why famous: democratized fine-tuning — proved you don't need 100K+ human-labeled examples. Limitation: synthetic data inherits GPT-3 biases and errors. Better datasets now: OpenHermes, UltraChat, FLAN.
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase7 #alpaca #dataset #instruction-tuning

---

**ID**: P7-028
**Front**: What is W&B (Weights & Biases) and how do you integrate it with fine-tuning?
**Back**: W&B: experiment tracking platform. Logs: loss curves, eval metrics, hyperparams, model checkpoints, GPU utilization. Integration: `os.environ["WANDB_API_KEY"] = "..."`. In `TrainingArguments`: `report_to="wandb"`, `run_name="experiment-name"`. Auto-logged: training loss, eval loss, learning rate, gradient norm. Custom: `wandb.log({"faithfulness": 0.92})`. Benefits: compare runs, reproduce best run, share results. Alternative: MLflow (better for Databricks).
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase7 #wandb #mlops #tracking

---

**ID**: P7-029
**Front**: What is GRPO (Group Relative Policy Optimization) and where did it come from?
**Back**: GRPO (DeepSeek 2024): PPO variant that eliminates the critic model. Instead of training a value function baseline, GRPO uses GROUP STATISTICS: sample N outputs per prompt, compute their relative rewards, use group mean as baseline. Benefits: (1) No separate value model to train. (2) More stable than PPO. (3) Computationally cheaper. Used to train DeepSeek-R1 (chain-of-thought reasoning). Represents frontier of RLHF-style training without full PPO complexity.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase7 #GRPO #DeepSeek #alignment

---

**ID**: P7-030
**Front**: How do you handle OOM (Out Of Memory) errors during fine-tuning?
**Back**: Debugging OOM: (1) Reduce batch size. (2) Enable gradient checkpointing. (3) Use gradient accumulation (effective larger batch without memory). (4) Use QLoRA (4-bit base model). (5) Reduce max_sequence_length. (6) Use smaller LoRA rank r. (7) Flash Attention 2 (memory-efficient). (8) Offload optimizer to CPU (DeepSpeed). Monitor: `nvidia-smi`, `torch.cuda.memory_summary()`. Start small and scale up; profile first, then optimize.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase7 #OOM #memory #debugging

---

**ID**: P7-031
**Front**: What is the difference between a base LLM and an instruct LLM?
**Back**: Base (pretrained): next-token prediction on raw text. Output: continues your prompt. "The capital of France is..." → "Paris, a beautiful city..." Instruct (fine-tuned on instructions): follows directives, answers questions, uses specific format. "What is the capital of France?" → "The capital of France is Paris." Use base for: research, understanding pretraining. Use instruct for: all application use cases. Never deploy base LLMs to users — no safety fine-tuning.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase7 #base-model #instruct #fine-tuning

---

**ID**: P7-032
**Front**: What is the purpose of the α/r scaling in LoRA?
**Back**: LoRA output: (α/r) · BA · x. If α = r, then scale = 1.0 (no scaling). Purpose: α controls how strongly the adapter influences the model. Higher α: adapter has more impact. Lower α: conservative update. In practice: `lora_alpha = r` is default → scale=1. Some practitioners: set alpha=16 with r=8 → scale=2 (more aggressive). `rsLoRA` (2024) uses `α/√r` scaling instead — better gradient norms for high-rank LoRA.
**Difficulty**: Advanced
**Category**: Math
**Tags**: #phase7 #LoRA #scaling #alpha

---

**ID**: P7-033
**Front**: What is the instruct dataset format for LLaMA-3 fine-tuning?
**Back**: LLaMA-3 instruct format uses special tokens: `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{assistant_response}<|eot_id|>`. Use `tokenizer.apply_chat_template(messages)` — don't construct this manually. Critical: using wrong template causes training on wrong positions → model can't generate correctly.
**Difficulty**: Advanced
**Category**: Coding
**Tags**: #phase7 #LLaMA #template #tokenizer

---

**ID**: P7-034
**Front**: What is the `print_trainable_parameters()` method in PEFT and what should it show?
**Back**: `model.print_trainable_parameters()`. Output: "trainable params: X || all params: Y || trainable%: Z%". For QLoRA with r=16 on LLaMA-3-8B: ~0.04% trainable (only LoRA adapters). If shows 100%: LoRA wasn't applied correctly. If shows 0%: base model frozen but no adapters added. For proper training: want 0.01–5% trainable depending on task complexity. Higher % not always better — over-adapting can cause forgetting.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase7 #PEFT #LoRA #parameters

---

**ID**: P7-035
**Front**: What is the difference between PPO and DPO in terms of training stability?
**Back**: PPO: online RL — generate outputs, get reward, update policy. Sensitive to: reward scale, KL coefficient, batch size, PPO clip range. Known for instability (reward hacking, mode collapse). DPO: offline — train on pre-collected preference pairs. More stable: no online generation needed, standard supervised training loop. DPO stability reason: the loss is a simple binary cross-entropy-like objective over fixed data, not a moving-target RL objective. Most production fine-tuning has moved to DPO.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase7 #PPO #DPO #stability
