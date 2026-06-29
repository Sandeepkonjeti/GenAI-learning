# Phase 7 — Fine-Tuning, RLHF & DPO Cheat Sheet

[← CheatSheets Index](./README.md) | [Full Phase Files](../08_Phase7_Part1_FineTuning_and_LoRA.md)

---

## When to Fine-Tune (Decision Tree)

```
Is your task solvable with prompt engineering?
  YES → Don't fine-tune (save cost/time)
  NO ↓
Is your task solvable with RAG?
  YES → Use RAG + prompt engineering
  NO ↓
Do you need consistent format/style/domain?
  YES → Fine-tune with LoRA/QLoRA
Do you need aligned behavior (safety/helpfulness)?
  YES → RLHF or DPO
```

---

## LoRA (Low-Rank Adaptation)

### Math

$$W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r} \cdot B \cdot A$$

Where:
- $W_0 \in \mathbb{R}^{d \times k}$: frozen pretrained weights
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$: trainable low-rank matrices
- $r$: rank (4–64; default 16)
- $\alpha$: scaling factor (often = r)
- Trainable params: $r \times (d + k)$ vs $d \times k$ — massive reduction

**QLoRA**: quantize $W_0$ to 4-bit (NF4), train B,A in bfloat16. Enables 70B models on 2× A100s.

---

## Fine-Tuning Code Pattern

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# 1. Load with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

# 2. Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # attention projections
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 3.4M || all params: 8B || 0.04%

# 3. Train with SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(output_dir="./output", num_train_epochs=3, per_device_train_batch_size=4),
)
trainer.train()
```

---

## RLHF Pipeline

```
1. SFT (Supervised Fine-Tuning)
   Pretrained LLM → train on (prompt, gold response) → SFT model

2. Reward Model
   SFT model → train on (prompt, chosen, rejected) pairs → RM

3. PPO (Proximal Policy Optimization)
   SFT model → optimize with RM signal + KL penalty → Aligned model

Objective: max E[r(x,y)] - β·KL(π||π_ref)
             ↑ reward        ↑ don't drift too far from SFT model
```

**KL penalty β**: keeps fine-tuned model from exploiting reward model (reward hacking).

---

## DPO (Direct Preference Optimization)

**Key insight**: skip the explicit reward model. Optimize directly on preference pairs.

$$\mathcal{L}_{DPO} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

```python
from trl import DPOTrainer, DPOConfig

# Dataset format: {"prompt": str, "chosen": str, "rejected": str}
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # frozen SFT model
    args=DPOConfig(beta=0.1, output_dir="./dpo_output"),
    train_dataset=preference_dataset,
)
trainer.train()
```

**DPO vs RLHF**:
| | RLHF | DPO |
|--|------|-----|
| Reward model | Explicit, trained separately | Implicit in loss |
| Stability | Can be unstable (PPO) | More stable |
| Compute | Higher (RM + PPO) | Lower |
| Quality | Typically better | Close to RLHF |

---

## Distributed Training

| Method | GPU Memory | When |
|--------|:----------:|------|
| Single GPU | Full model | ≤7B |
| DDP | Full model × N (grads synced) | Data parallelism |
| DeepSpeed ZeRO-2 | Split optimizer + grad | ≤30B |
| DeepSpeed ZeRO-3 | Split params too | 65B+ |
| FSDP | PyTorch-native ZeRO-3 equivalent | Preferred in PyTorch 2.x |

```python
# Databricks TorchDistributor
from pyspark.ml.torch.distributor import TorchDistributor

def train_fn():
    # your training code here
    pass

distributor = TorchDistributor(num_processes=4, local_mode=False, use_gpu=True)
distributor.run(train_fn)
```

---

## Common Mistakes

- **Wrong target modules for LoRA** — check model architecture; use `q_proj, v_proj` for LLaMA
- **Too high rank (r=64+)** — diminishing returns; r=16 usually sufficient
- **Not using `gradient_checkpointing`** — OOM on long sequences; enable it
- **Reward hacking in RLHF** — model finds loopholes to maximize reward without being helpful; KL penalty + careful reward design mitigates
- **DPO beta too high** — model barely deviates from reference; too low → reward hacking

---

## Interview Quick-Hits

**Q: Explain LoRA mathematically.**  
A: LoRA freezes pretrained weights $W_0$ and learns a low-rank update $\Delta W = BA$ where $B \in \mathbb{R}^{d×r}$ and $A \in \mathbb{R}^{r×k}$ with rank $r \ll \min(d,k)$. This reduces trainable params from $d×k$ to $r(d+k)$, e.g., 0.04% of parameters for LLaMA-3-8B with r=16.

**Q: What is RLHF and why does it matter?**  
A: RLHF (Reinforcement Learning from Human Feedback) is how InstructGPT/ChatGPT was aligned to follow instructions helpfully and safely. Without it, LLMs just predict next tokens without following user intent. The KL penalty prevents the model from gaming the reward model.

**Q: DPO vs RLHF — when would you choose each?**  
A: DPO for production (simpler, more stable, cheaper — no RM training). RLHF when maximum quality is needed and you have resources for a separate RM + PPO loop. DPO quality has closed the gap significantly.

**Q: What is QLoRA's key contribution?**  
A: Quantize the frozen base model to 4-bit NF4 format, train LoRA adapters in bfloat16. This allows fine-tuning 65B models on a single A100 (80GB) — democratizing fine-tuning.
