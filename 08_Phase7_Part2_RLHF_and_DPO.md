# Phase 7b: RLHF & DPO
**Months 15–16 | Difficulty: 9/10 | Label: � Must Learn**

> **Previous Phase**: [Phase 7a — Fine-Tuning & LoRA](./08_Phase7_Part1_FineTuning_and_LoRA.md)  
> **Next Phase**: [Phase 8 — Advanced Topics](./09_Phase8_Advanced_Topics.md)

---

## Table of Contents

- [Phase Overview](#phase-overview)
- [The Alignment Problem](#the-alignment-problem)
- [Part A: RLHF (Reinforcement Learning from Human Feedback)](#part-a-rlhf)
  - [The RLHF Pipeline](#the-rlhf-pipeline)
  - [Step 1: Supervised Fine-Tuning (SFT)](#step-1-supervised-fine-tuning-sft)
  - [Step 2: Reward Model Training](#step-2-reward-model-training)
  - [Step 3: PPO Training](#step-3-ppo-training)
  - [The KL Divergence Penalty](#the-kl-divergence-penalty)
- [Part B: DPO (Direct Preference Optimization)](#part-b-dpo)
  - [Why DPO Replaces PPO](#why-dpo-replaces-ppo)
  - [The DPO Objective](#the-dpo-objective)
  - [DPO Loss Derivation](#dpo-loss-derivation)
  - [Implementing DPO with TRL](#implementing-dpo-with-trl)
- [Part C: Preference Datasets](#part-c-preference-datasets)
  - [Dataset Format](#dataset-format)
  - [Human vs. AI-Generated Preferences](#human-vs-ai-generated-preferences)
- [Part D: Constitutional AI](#part-d-constitutional-ai)
- [Resources](#resources)
- [Projects](#projects)
- [Common Mistakes](#common-mistakes)
- [Mastery Checklist](#mastery-checklist)

---

## Phase Overview

| Attribute | Details |
|-----------|---------|
| **Duration** | Months 15–16 (3 weeks) |
| **Daily Time** | 1–1.5 hours |
| **Difficulty** | 9/10 |
| **Label** | � Must Learn |
| **Prerequisites** | Phase 7a (SFT, LoRA), Phase 3a (reinforcement learning concepts helpful) |
| **Outcome** | Deep understanding of alignment techniques; can run DPO training; can reason about AI safety |

---

## The Alignment Problem

After pretraining and SFT, models are capable but not necessarily aligned with human values. The **alignment problem** asks:

> "How do we make an AI system do what humans actually want — not just what we explicitly specified?"

```
The three A's of alignment:
1. Ability: Can the model do the task? (pretraining solves this)
2. Alignment: Does it do the task in the way we want? (RLHF/DPO solve this)
3. Avoidance: Does it avoid things we don't want? (RLHF, safety training, constitutional AI)

Example of the gap:
  Task: "Write a persuasive essay about climate policy."
  
  SFT model: Writes a good essay (helpful).
  
  Unaligned risks:
  - Could be maximally persuasive by using manipulation techniques
  - Could present biased, one-sided information
  - Could hallucinate statistics to strengthen the argument
  
  Aligned model: Writes a balanced, factually accurate essay that is persuasive
                 through legitimate reasoning.
```

---

## Part A: RLHF

### The RLHF Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                     RLHF PIPELINE                              │
│                                                                │
│  Step 1: SFT                                                   │
│  ──────────                                                    │
│  Base LLM → Fine-tune on instruction data → SFT Model         │
│                                                                │
│  Step 2: Reward Model                                          │
│  ────────────────────                                          │
│  For each prompt: generate 2 responses A and B                 │
│  Human labels: "A is better" or "B is better"                  │
│  Train: Reward Model to assign higher score to preferred response│
│                                                                │
│  Step 3: RL Training (PPO)                                     │
│  ─────────────────────────                                     │
│  Policy (LLM) generates response                               │
│  Reward Model assigns score                                    │
│  PPO updates LLM to maximize reward                            │
│  KL penalty prevents too much deviation from SFT model         │
└────────────────────────────────────────────────────────────────┘
```

---

### Step 1: Supervised Fine-Tuning (SFT)

(Covered in Phase 7a — this is a prerequisite for RLHF)

The SFT model is the starting point. Without it, RL training is unstable — the model has no basis for producing coherent responses that the reward model can evaluate.

---

### Step 2: Reward Model Training

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class RewardModel(nn.Module):
    """
    A reward model predicts how "good" a response is.
    
    Architecture: Take a pre-trained LLM (or SFT model),
    replace the language modeling head with a scalar regression head.
    
    Input: (prompt + response) concatenated
    Output: scalar score (higher = better)
    """
    def __init__(self, base_model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        d_model = self.model.config.hidden_size
        
        # Replace LM head with scalar regression head
        self.reward_head = nn.Linear(d_model, 1, bias=False)
        
        # Remove the original LM head (saves memory during RM training)
        self.model.lm_head = nn.Identity()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use the last token's hidden state as the representation
        # (for decoder models, this captures the full sequence)
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # (batch, d_model)
        reward = self.reward_head(last_hidden)  # (batch, 1)
        
        return reward.squeeze(-1)  # (batch,)


class PreferenceDataset(Dataset):
    """
    Pairs of (prompt, chosen_response, rejected_response).
    
    Chosen: human preferred response
    Rejected: human non-preferred response
    """
    def __init__(self, data: list[dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # Format: "Prompt: {prompt}\n\nResponse: {response}"
        def encode(response: str) -> dict:
            text = f"Prompt: {prompt}\n\nResponse: {response}"
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            return {k: v.squeeze(0) for k, v in encoded.items()}
        
        return {
            "chosen": encode(chosen),
            "rejected": encode(rejected)
        }


def train_reward_model(model: RewardModel, dataset: PreferenceDataset, epochs: int = 3):
    """
    Train reward model with Bradley-Terry preference model.
    
    The loss is:
    L = -E[log σ(r_chosen - r_rejected)]
    
    where σ is sigmoid. This pushes chosen rewards higher than rejected.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            chosen_ids = batch["chosen"]["input_ids"]
            chosen_mask = batch["chosen"]["attention_mask"]
            rejected_ids = batch["rejected"]["input_ids"]
            rejected_mask = batch["rejected"]["attention_mask"]
            
            # Compute rewards
            r_chosen = model(chosen_ids, chosen_mask)     # (batch,)
            r_rejected = model(rejected_ids, rejected_mask) # (batch,)
            
            # Bradley-Terry loss: maximize probability that chosen > rejected
            # -log σ(r_chosen - r_rejected)
            loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Accuracy: fraction of examples where model prefers chosen > rejected
        accuracy = (r_chosen > r_rejected).float().mean().item()
        print(f"Epoch {epoch+1}: loss={total_loss/len(dataloader):.4f}, accuracy={accuracy:.3f}")
```

---

### Step 3: PPO Training

```python
# PPO (Proximal Policy Optimization) in the context of RLHF

# The key components:
# Policy:         The LLM (what we're training)
# Reference:      The SFT model (frozen — defines "normal" behavior)
# Reward:         Reward model score + KL penalty
# Value function: Another head that predicts expected future reward

# The optimization:
# Maximize:  R(response) - β * KL(policy || reference)
# 
# Where:
# R(response) = reward model score
# KL(policy || reference) = divergence from SFT model (prevents mode collapse)
# β = KL coefficient (typically 0.1-0.5)

# Key insight: the KL penalty is what prevents reward hacking
# Without it, the policy would exploit the reward model (find inputs that get high
# reward but are actually terrible — reward models are imperfect)

# In practice, PPO for RLHF is complex and unstable:
# - Requires 4 models simultaneously (policy, reference, reward, value)
# - Requires significant VRAM (4 × LLM size)
# - Training is sensitive to hyperparameters
# - Can take days to converge
# This is why DPO was developed.
```

---

### The KL Divergence Penalty

```python
import torch
import torch.nn.functional as F

def compute_kl_divergence(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute per-token KL divergence between policy and reference model.
    
    KL(policy || reference) = Σ p(token) * log(p(token) / q(token))
    
    For RLHF, this is computed per-token and summed over the response.
    """
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    reference_log_probs = F.log_softmax(reference_logits, dim=-1)
    
    # KL(p || q) = p * (log_p - log_q)
    policy_probs = F.softmax(policy_logits, dim=-1)
    kl = (policy_probs * (policy_log_probs - reference_log_probs)).sum(dim=-1)
    
    return kl  # Per-token KL divergence

# Why KL matters:
kl_effects = {
    "β too high (KL penalty too strong)": """
    Policy barely deviates from SFT model.
    Almost no improvement in alignment.
    Model still behaves like SFT model.
    """,
    "β too low (KL penalty too weak)": """
    Policy diverges heavily from SFT model.
    Reward hacking: model finds nonsense outputs that fool the reward model.
    Model loses coherence and quality.
    """,
    "β well-tuned": """
    Policy improves on alignment metrics.
    Maintains SFT model's quality.
    Doesn't exploit reward model.
    """
}
```

---

## Part B: DPO

### Why DPO Replaces PPO

DPO (Direct Preference Optimization, Rafailov et al., 2023) is now the standard alignment technique, largely replacing PPO because:

| | PPO | DPO |
|--|:--:|:--:|
| Models needed simultaneously | 4 (policy, reference, reward, value) | 2 (policy, reference) |
| Training stability | Low (RL is notoriously unstable) | High (supervised loss) |
| Hyperparameter sensitivity | Very high | Moderate |
| Implementation complexity | Very high | Low |
| Quality | Slightly better on some tasks | Competitive |
| Memory requirement | 4× LLM size | 2× LLM size |
| Time to train | Days–weeks | Hours–days |

---

### The DPO Objective

**Key insight of DPO**: The reward maximization problem has a closed-form solution. You don't need to train a separate reward model and run RL. Instead, you can directly optimize the policy using preference pairs.

The DPO loss is:

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right]$$

Where:
- $y_w$: **winning** (preferred) response
- $y_l$: **losing** (rejected) response
- $\pi_\theta$: policy (trainable model)
- $\pi_{ref}$: reference (frozen SFT model)
- $\beta$: temperature controlling KL constraint strength

**Intuition**: DPO increases the probability of preferred responses and decreases the probability of rejected responses, **relative** to the reference model. The reference model acts as an implicit KL constraint.

---

### DPO Loss Derivation

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logprobs: torch.Tensor,    # log π_θ(y_w | x)
    policy_rejected_logprobs: torch.Tensor,  # log π_θ(y_l | x)
    reference_chosen_logprobs: torch.Tensor, # log π_ref(y_w | x)
    reference_rejected_logprobs: torch.Tensor, # log π_ref(y_l | x)
    beta: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DPO loss from Rafailov et al. 2023.
    
    The key quantities:
    - chosen_rewards = β * (log π_θ(y_w|x) - log π_ref(y_w|x))
    - rejected_rewards = β * (log π_θ(y_l|x) - log π_ref(y_l|x))
    
    These measure "how much MORE the policy prefers y_w/y_l compared to reference"
    
    Loss = -log σ(chosen_rewards - rejected_rewards)
    This maximizes the margin between chosen and rejected, relative to reference.
    """
    
    # Compute log ratios (implicit reward model in DPO)
    chosen_rewards = beta * (policy_chosen_logprobs - reference_chosen_logprobs)
    rejected_rewards = beta * (policy_rejected_logprobs - reference_rejected_logprobs)
    
    # DPO loss
    reward_margins = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(reward_margins).mean()
    
    # Monitoring metrics
    chosen_reward_mean = chosen_rewards.mean()
    rejected_reward_mean = rejected_rewards.mean()
    accuracy = (reward_margins > 0).float().mean()  # Fraction where chosen > rejected
    
    return loss, chosen_reward_mean, rejected_reward_mean


def get_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute log probabilities for a sequence given the model.
    This is the sum of log p(token_i | token_1, ..., token_{i-1}) over response tokens.
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Shift: labels are input_ids shifted by 1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Log probabilities for each token
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log prob for the actual token
    token_log_probs = log_probs.gather(
        -1, shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Sum over response tokens only (mask out prompt tokens with -1)
    response_mask = (shift_labels != -100).float()
    sequence_log_probs = (token_log_probs * response_mask).sum(-1)
    
    return sequence_log_probs
```

---

### Implementing DPO with TRL

```python
from trl import DPOTrainer, DPOConfig
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch

# ── Step 1: Load SFT Model (starting point for DPO) ──
model_name = "your-sft-model"  # Fine-tuned model from Phase 7a
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA (optional but recommended for memory efficiency)
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)

# ── Step 2: Load Reference Model (frozen SFT model) ──
reference_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
# No LoRA on reference model — it stays frozen

# ── Step 3: Prepare Preference Dataset ──
# DPO dataset format: prompt, chosen (preferred), rejected (not preferred)
preference_data = [
    {
        "prompt": "How do I handle NULL values in PySpark?",
        "chosen": "Use df.fillna(value) to replace NULLs, df.dropna() to remove rows, or df.na.fill({'col': value}) for column-specific filling.",
        "rejected": "You can use various methods to handle nulls in PySpark."  # vague, unhelpful
    },
    {
        "prompt": "Explain Delta Lake ACID transactions.",
        "chosen": "Delta Lake provides ACID transactions through: 1) Atomicity via transaction log, 2) Consistency via schema enforcement, 3) Isolation via optimistic concurrency control, 4) Durability via cloud storage. The delta_log/ directory stores JSON/Parquet files recording every change.",
        "rejected": "Delta Lake supports ACID transactions for data reliability."  # too brief
    }
    # ... more examples
]

dataset = Dataset.from_list(preference_data)

# ── Step 4: Configure and Run DPO ──
dpo_config = DPOConfig(
    output_dir="./dpo-finetuned",
    
    # Core DPO parameter
    beta=0.1,              # KL penalty coefficient (0.01-0.5)
    
    # Training
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,    # Lower than SFT (already at a good starting point)
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    
    # Memory
    bf16=True,
    gradient_checkpointing=True,
    
    # DPO-specific
    max_prompt_length=256,
    max_length=512,
    
    # Logging
    logging_steps=10,
    eval_strategy="epoch",
)

trainer = DPOTrainer(
    model=model,
    ref_model=reference_model,   # Frozen reference model
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

# ── Monitoring DPO Training ──
# Good signs:
# - chosen_rewards increasing
# - rejected_rewards decreasing (or staying flat)
# - accuracy > 0.7 (model prefers chosen over rejected 70%+ of the time)
# - Loss decreasing steadily

# Warning signs:
# - chosen_rewards AND rejected_rewards both increasing equally (reward hacking)
# - Very high loss (dataset may have incorrect labels)
# - accuracy < 0.5 (model is getting confused)
```

---

## Part C: Preference Datasets

### Dataset Format

```python
# Standard format for DPO/preference training
preference_example = {
    "prompt": "What is the best way to optimize a slow Spark job?",
    "chosen": """There are several key optimizations to consider:

1. **Partitioning**: Ensure data is well-partitioned (200 partitions for large datasets, ~128MB per partition)
2. **Broadcast joins**: For tables < 10MB, use broadcast hint: df.join(broadcast(small_df), ...)
3. **Caching**: Cache DataFrames reused multiple times: df.cache()
4. **Avoid UDFs**: Pandas UDFs (vectorized) are 10-100x faster than Python UDFs
5. **Predicate pushdown**: Filter early, before joins and aggregations
6. **AQE (Adaptive Query Execution)**: Enable spark.sql.adaptive.enabled=true in Spark 3.x

Profile first with Spark UI, then optimize the most expensive stage.""",
    
    "rejected": """You can optimize Spark jobs by looking at the Spark UI and trying different 
configurations. Some things to try include caching your data and using broadcast joins when 
you have small tables. Also make sure your partitioning is good."""
}

# What makes a good preference pair?
preference_quality = {
    "chosen characteristics": [
        "Comprehensive and accurate",
        "Well-structured with specific details",
        "Actionable (user can implement immediately)",
        "Appropriately detailed for the question",
        "Factually correct and verifiable",
    ],
    "rejected characteristics": [
        "Vague or generic ('try different things')",
        "Incomplete or missing critical steps",
        "Inaccurate or hallucinated facts",
        "Wrong format (no code when code expected)",
        "Too brief for a complex question",
    ]
}
```

---

### Human vs. AI-Generated Preferences

```python
# Option 1: Human preferences (gold standard, expensive)
# - ~$0.10-1.00 per preference pair from annotators
# - Subjective and inconsistent between annotators
# - Required for safety-critical applications

# Option 2: GPT-4 as judge (practical, scalable)
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def gpt4_judge(prompt: str, response_a: str, response_b: str) -> dict:
    """
    Use GPT-4 to create preference pairs.
    This is common practice (Alpaca-Farm, UltraFeedback datasets).
    """
    judge_prompt = f"""You are an expert AI evaluator. Compare two responses to the same question.

Question: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Evaluate which response is better based on:
1. Accuracy and factual correctness
2. Completeness and depth
3. Clarity and structure
4. Practical usefulness

Output JSON only:
{{"winner": "A" or "B" or "tie", "reasoning": "brief explanation", "confidence": 0.0-1.0}}"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"}
    )
    
    import json
    result = json.loads(response.choices[0].message.content)
    return result

# Existing preference datasets to start with:
public_datasets = {
    "HuggingFaceH4/ultrafeedback_binarized": "280K pairs, GPT-4 judged, high quality",
    "argilla/dpo-mix-7k":                    "7K curated pairs, mixed sources",
    "Intel/orca_dpo_pairs":                  "13K coding-focused pairs",
    "lmsys/chatbot_arena_conversations":     "33K human preference pairs, battle format",
    "anthropic/hh-rlhf":                     "170K human-written pairs, Anthropic",
}
```

---

## Part D: Constitutional AI

Constitutional AI (Anthropic, 2022) is an alternative approach that avoids human preference labeling entirely:

```python
# Constitutional AI procedure:
# 1. Generate response to a potentially harmful prompt
# 2. Have the model critique its own response based on a "constitution"
# 3. Have the model revise the response based on the critique
# 4. Use revised responses as "chosen" and original as "rejected" for DPO

CONSTITUTION = [
    "Does this response reflect someone who is helpful, harmless, and honest?",
    "Does this response avoid providing content that could be used to harm others?",
    "Does this response avoid deception or manipulation?",
    "Is this response appropriately qualified when uncertainty exists?",
    "Does this response treat all people with equal respect?",
]

def apply_constitution(prompt: str, initial_response: str, client) -> dict:
    """Apply Constitutional AI to improve a response."""
    
    # Step 1: Critique
    critique_prompt = f"""Here is a conversation:
Human: {prompt}
AI: {initial_response}

Review the AI's response against this criterion: {CONSTITUTION[0]}
Write a brief critique."""
    
    critique = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": critique_prompt}]
    ).choices[0].message.content
    
    # Step 2: Revision
    revision_prompt = f"""Human: {prompt}
AI: {initial_response}

Critique: {critique}

Please rewrite the AI response to address the critique."""
    
    revised_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": revision_prompt}]
    ).choices[0].message.content
    
    return {
        "prompt": prompt,
        "rejected": initial_response,     # Original (may be harmful)
        "chosen": revised_response,        # Revised (better aligned)
    }

# This generates preference data automatically, without human annotators!
# Used by Anthropic to train Claude at scale.
```

---

## Resources

| Rank | Resource | Type | Cost | Why |
|------|----------|------|------|-----|
| 1 | [DPO Paper (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) | Paper | Free | Short (10 pages), clear math. Read this. |
| 2 | [TRL DPO Tutorial](https://huggingface.co/docs/trl/dpo_trainer) | Docs | Free | Best hands-on DPO guide |
| 3 | [RLHF Paper (Stiennon et al., 2020)](https://arxiv.org/abs/2009.01325) | Paper | Free | Original RLHF from human feedback paper |
| 4 | [Constitutional AI (Bai et al., 2022)](https://arxiv.org/abs/2212.08073) | Paper | Free | Anthropic's approach to scalable alignment |
| 5 | [Lilian Weng "RLHF: Reinforcement Learning from Human Feedback"](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) | Blog | Free | Best blog overview of alignment techniques |

---

## Projects

### Project 19: DPO Alignment Fine-Tune
**Difficulty**: 8/10 | **Time**: 2 weeks

Start from your fine-tuned model from Phase 7a and apply DPO:

1. Generate 200 responses from your SFT model to domain-specific prompts
2. Use GPT-4 to generate preferred alternatives for bad responses
3. Create (prompt, chosen, rejected) triples
4. Train with DPO using TRL
5. Compare: SFT vs. DPO on 30 test prompts

**Key evaluation**: Does the DPO model give more complete, better-structured answers?

---

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Low-quality preference pairs | Model learns wrong preferences | Ensure clear quality gap between chosen and rejected |
| β too high | Model barely changes from SFT | Lower β to 0.05–0.1 |
| β too low | Reward hacking, incoherent outputs | Raise β; add more validation examples |
| Preference pairs with same quality | Model can't learn meaningful differences | Only use pairs with clear preference margin |
| Not starting from SFT model | Unstable training, poor results | Always use a fine-tuned SFT model, not the raw base |
| Too few preference pairs | Overfitting | Minimum 500 pairs; prefer 2000+ |

---

## Mastery Checklist

- [ ] Can explain the 3-step RLHF pipeline (SFT → RM → PPO)
- [ ] Understands why KL divergence is necessary in RLHF
- [ ] Can explain DPO and why it replaces PPO in practice
- [ ] Can derive the DPO loss intuitively
- [ ] Ran a DPO fine-tuning job with TRL
- [ ] Can explain Constitutional AI and how it generates preference data without humans
- [ ] Read the DPO paper

---

## Moving to Phase 8

**Before proceeding to [Phase 8: Advanced Topics](./09_Phase8_Advanced_Topics.md), confirm:**

- [ ] Can explain when to use SFT vs. DPO (they're complementary, not alternatives)
- [ ] Have run both SFT and DPO fine-tuning on a model
- [ ] Understand the tradeoffs between RLHF and DPO

**Why Phase 8 comes next**: You've mastered the core of GenAI engineering. Phase 8 covers advanced topics — multimodal AI, diffusion models, GPU optimization, and inference optimization — that will differentiate you as an expert.

---

## Phase Completion & Readiness Assessment

> Complete this assessment before Phase 8. RLHF and DPO are how frontier labs produce helpful, harmless, honest models — understanding them makes you a complete AI engineer.

---

### 1. Knowledge Checklist

**Alignment Problem**
- [ ] Why aligned ≠ capable: difference between "can do" and "will do"
- [ ] RLHF: 3-stage pipeline — SFT → reward model → PPO
- [ ] Why base models generate harmful, unhelpful, or biased content without alignment

**Reward Model**
- [ ] Bradley-Terry model: probability of A preferred over B
- [ ] Reward model training: pairwise ranking loss
- [ ] Why the reward model is separate from the policy model

**PPO for RLHF**
- [ ] PPO: policy gradient method with clipped surrogate objective
- [ ] KL divergence penalty: why it prevents "reward hacking"
- [ ] Reference model: why it's kept frozen during PPO
- [ ] Actor-critic: value model role in PPO
- [ ] PPO instability: common failure modes

**DPO**
- [ ] DPO mathematical derivation: closed-form solution from RLHF objective
- [ ] DPO loss: L_DPO(θ) formula
- [ ] How DPO eliminates the need for a reward model
- [ ] DPO vs. PPO: tradeoffs (simplicity vs. flexibility)
- [ ] Beta (β) hyperparameter: effect on alignment strength
- [ ] Preference dataset format: (prompt, chosen, rejected) triplets
- [ ] IPO (Identity Preference Optimisation): when DPO can overfit

---

### 2. Practical Skills Checklist

- [ ] Implement the Bradley-Terry reward model training from scratch
- [ ] Implement the DPO loss function from scratch in PyTorch
- [ ] Build a preference dataset from LLM outputs (using GPT-4 as judge)
- [ ] Run DPO training using TRL DPOTrainer
- [ ] Evaluate SFT vs. DPO using LLM-as-judge with a rubric
- [ ] Explain the KL divergence penalty and why removing it causes reward hacking
- [ ] Derive the DPO objective from first principles (on paper)

---

### 3. Coding Challenges

**Challenge A — DPO Loss from Scratch**
```python
# Implement the DPO loss without using the TRL library:
def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # log P_θ(y_w | x)
    policy_rejected_logps: torch.Tensor,   # log P_θ(y_l | x)
    reference_chosen_logps: torch.Tensor,  # log P_ref(y_w | x)
    reference_rejected_logps: torch.Tensor,# log P_ref(y_l | x)
    beta: float = 0.1,
) -> torch.Tensor:
    """DPO loss: -E[log sigmoid(beta * (log ratio chosen - log ratio rejected))]"""
    pass

# Verify: loss decreases when policy assigns higher probability to chosen vs. rejected
# Verify: with beta=0, loss is constant (no preference learning)
```

**Challenge B — Preference Dataset Builder**
```python
# Build a pipeline that creates a DPO preference dataset:
# 1. Take 200 prompts from your fine-tuning domain
# 2. Generate 3 responses per prompt using your SFT model (different temperatures)
# 3. Use GPT-4 to rank the 3 responses with justification
# 4. Extract the best (chosen) and worst (rejected) response
# 5. Quality filter: discard pairs where GPT-4 is uncertain (score difference < 2)
# Output: HuggingFace Dataset in DPO format (prompt, chosen, rejected)
```

**Challenge C — Constitutional AI Lite**
```python
# Implement a simplified Constitutional AI pipeline:
# Principles: ["Be helpful", "Be honest", "Avoid harmful content"]
# For each response your SFT model generates:
# 1. Ask the LLM: "Does this response violate any of these principles? Why?"
# 2. If violations found: "Rewrite the response to comply with all principles"
# 3. Use (original_response, revised_response) as a preference pair
# Build a 100-example preference dataset this way
# Evaluate: do responses improve on a human quality rubric?
```

---

### 4. Mini Project

**DPO Alignment Fine-Tune** (Project 19 from the roadmap):
- Start from your Phase 7a SFT model
- Build a 300+ example preference dataset using Challenge B
- Run DPO training with TRL DPOTrainer (beta=0.1, lr=1e-6)
- Compare: SFT model vs. DPO model on 30 test prompts
- Evaluation: LLM-as-judge scores for helpfulness, harmlessness, correctness
- Report: did alignment improve without losing capability?

---

### 5. Capstone Project

**Alignment Study**: Compare 3 alignment methods on the same base SFT model:
1. SFT only (no alignment)
2. DPO (beta=0.1 and beta=0.3)
3. Constitutional AI Lite (Challenge C approach)

Evaluation on 50 test prompts: helpfulness, harmlessness, format quality.
Report: Which method wins? Where does each fail?
What is the capability-alignment tradeoff for each?

---

### 6. Interview Questions

**Beginner**

1. **Q: What is the alignment problem in LLMs?**
   A: The alignment problem is the difficulty of getting LLMs to behave as intended — being helpful, honest, and avoiding harmful outputs. Base models optimised for next-token prediction generate whatever is statistically likely, not what is safe or helpful. Alignment training explicitly teaches the model what humans prefer.

2. **Q: What are the three stages of RLHF?**
   A: (1) SFT: fine-tune on high-quality demonstrations. (2) Reward Model: train a model to predict human preferences from pairwise comparisons. (3) PPO: optimise the policy to maximise the reward model's score while staying close to the SFT model (KL penalty).

3. **Q: What is the reward model in RLHF?**
   A: A model that takes a prompt and a response and outputs a scalar score representing how much a human would prefer that response. Trained on pairwise human preference data (which of two responses is better). Used in PPO to provide a reward signal.

4. **Q: What is DPO and why is it simpler than RLHF?**
   A: Direct Preference Optimisation eliminates the reward model and PPO loop. It derives a closed-form objective that directly optimises for human preferences from pairwise data. Same data format as RLHF but ~10× simpler to implement and more stable to train.

5. **Q: What is the KL divergence penalty in RLHF and why is it needed?**
   A: Without the KL penalty, PPO will find responses that fool the reward model but are meaningless — "reward hacking". The KL penalty KL(π_θ || π_ref) penalises the policy for deviating too far from the SFT model, ensuring the model doesn't collapse into degenerate outputs.

6. **Q: What is a preference dataset?**
   A: A dataset of triplets: (prompt, chosen, rejected). For each prompt, "chosen" is the human-preferred response and "rejected" is the less-preferred response. Used to train reward models (RLHF) or directly for DPO.

7. **Q: What does the beta parameter in DPO control?**
   A: Beta controls the strength of alignment. High beta (>0.3): stronger push toward preferences but more risk of forgetting general capabilities. Low beta (<0.05): weaker alignment, closer to SFT. Typical range: 0.1–0.3. It corresponds to the KL penalty weight in the RLHF objective.

**Intermediate**

8. **Q: Derive the DPO loss function conceptually (not full math).**
   A: Start from the RLHF objective: maximise reward minus KL penalty. The optimal policy has a closed form in terms of the reward function. Substituting, you can express the reward difference in terms of log ratios of policy to reference. DPO loss: minimise -log sigmoid(β × (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x))). No reward model needed.

9. **Q: What is the Bradley-Terry model used for in reward model training?**
   A: Bradley-Terry models pairwise comparisons: P(A > B) = σ(r_A - r_B). The reward model outputs r_A and r_B; training maximises the probability of the human-preferred response being ranked higher. Loss: -log σ(r_chosen - r_rejected).

10. **Q: What are the failure modes of PPO for LLM alignment?**
    A: (1) Reward hacking: model finds loopholes in the reward model (e.g., verbose responses score higher). (2) Training instability: PPO requires careful hyperparameter tuning. (3) Mode collapse: model stops generating diverse responses. (4) KL explosion: policy deviates too far, generating gibberish. Mitigation: clip KL penalty, monitor reward model score distribution.

11. **Q: How does DPO handle the case where the SFT model assigns equal probability to chosen and rejected?**
    A: If log π_SFT(y_w) = log π_SFT(y_l), the DPO gradient is zero — no learning signal. This "data difficulty" issue occurs when the SFT model is well-calibrated already. Solution: use a weaker reference model, or use a temperature-scaled reference.

12. **Q: What is the difference between online and offline RLHF?**
    A: Offline: preference data is fixed before training (DPO is offline). Online: generate responses during training, collect new preferences, update. Online RLHF (PPO) uses fresh data each iteration. Online is generally better but requires a deployed reward model and is much more complex.

13. **Q: What is Constitutional AI and how does it reduce the need for human annotation?**
    A: Constitutional AI (Anthropic) trains a model to critique and revise its own outputs according to a set of principles ("the constitution"). The model generates both the original and revised responses, creating synthetic preference data. Dramatically reduces human annotation cost while producing well-aligned models.

**Advanced**

14. **Q: Write the DPO loss in mathematical notation.**
    A: L_DPO(θ) = -E_{(x,y_w,y_l)~D}[log σ(β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x)))]

15. **Q: Why can DPO sometimes harm model capabilities (over-refusal)?**
    A: DPO pushes the model to decrease probability of "rejected" responses. If the preference dataset disproportionately labels substantive (but potentially risky) responses as rejected, the model learns to refuse broadly rather than selectively. Mitigation: balanced preference data, careful rejected response curation, using IPO instead of DPO.

16. **Q: How does IPO (Identity Preference Optimisation) differ from DPO?**
    A: DPO can overfit to the preference dataset when the model is trained for many steps — the log ratios grow unbounded. IPO adds an identity constraint that prevents this: it penalises large log ratio differences, keeping the policy from over-optimising. Better for large preference datasets.

17. **Q: What is reward over-optimisation and how does the KL penalty mitigate it?**
    A: As PPO training proceeds, the policy maximises the reward model score — but the reward model is imperfect. The model eventually finds outputs that score very high on the reward model but are low quality (gibberish that scores well due to reward model errors). KL penalty: R(x,y) - β·KL(π_θ||π_ref) prevents straying too far from the reference, limiting exploitation.

18. **Q: How would you decide between DPO and PPO for a production fine-tuning project?**
    A: Use DPO when: (1) you have a fixed preference dataset; (2) simplicity matters; (3) offline training is acceptable; (4) resources are limited. Use PPO when: (1) you can collect dynamic preference feedback; (2) task is complex enough that fixed datasets aren't sufficient; (3) you have infrastructure for running the reward model during training.

19. **Q: What is RLAIF (Reinforcement Learning from AI Feedback) and when is it used?**
    A: Instead of human feedback, use a stronger LLM (e.g., GPT-4) to generate preference judgements. RLAIF: Constitutional AI, Self-Instruct, Alpaca, many open models. Benefits: scale human annotation budget, consistency. Risks: biases from the judge model propagate, may miss human-specific preferences.

20. **Q: How would you evaluate whether your DPO training improved alignment?**
    A: (1) LLM-as-judge: compare SFT vs. DPO on 50 test prompts (GPT-4 scores 1-5 for helpfulness, harmlessness, honesty). (2) MT-Bench: standard benchmark for chat models. (3) Human evaluation: blind A/B test on 20 diverse prompts. (4) Check capability preservation: eval on MMLU/HumanEval to ensure fine-tuning didn't harm general abilities. (5) Safety benchmark: ToxiGen, AdvBench.

---

### 7. Self-Assessment Quiz

- [ ] Write the DPO loss formula from memory.
- [ ] What is the Bradley-Terry model?
- [ ] What are the 3 stages of RLHF?
- [ ] What is reward hacking and how does KL penalty prevent it?
- [ ] What is the beta parameter in DPO?
- [ ] What is the difference between DPO and PPO?
- [ ] What is the preference dataset format?
- [ ] What does the reference model do in DPO training?
- [ ] Why is DPO simpler to implement than RLHF?
- [ ] What is Constitutional AI?
- [ ] What is the difference between online and offline RLHF?
- [ ] What is reward over-optimisation?
- [ ] What is IPO and when is it better than DPO?
- [ ] What is RLAIF?
- [ ] What does "alignment tax" mean?
- [ ] How do you evaluate DPO improvement?
- [ ] What is the actor-critic setup in PPO for RLHF?
- [ ] Why must you compute log probabilities for both chosen and rejected?
- [ ] What is the difference between DPO and supervised fine-tuning?
- [ ] What dataset size is typically needed for DPO?
- [ ] What is the reference policy in DPO?
- [ ] What happens if all your preference pairs have the same reward difference?
- [ ] How do you handle ties in preference annotation?
- [ ] What is the "forbidden zone" in KL-reward tradeoff space?
- [ ] What is reinforcement learning from human feedback at a conceptual level?

**Scoring**: 22–25 ✅ = Ready. 17–21 = Review weak areas. Below 17 = Spend more time on Phase 7b.

---

### 8. Common Mistakes

| Mistake | Why It Happens | How to Avoid |
|---------|---------------|--------------|
| Using too high a beta | Higher seems more aligned | High beta degrades capabilities; start with 0.1, increase cautiously |
| Preference data where chosen/rejected are too similar | Fast labelling | Only use pairs where the quality gap is clear; ambiguous pairs hurt training |
| Not monitoring capability metrics | Only track alignment | Always evaluate on MMLU or domain benchmarks alongside alignment metrics |
| Using the same model for reference and policy | Saves memory | DPO requires two separate models; the reference must stay frozen |
| Training too long | More training = better alignment | DPO can overfit; monitor val loss and stop early |
| Using GPT-4 for preferences without checking its biases | GPT-4 is a reliable judge | GPT-4 may prefer verbose or sycophantic responses; use structured rubrics, not open-ended preference |

---

### 9. Readiness Criteria

You are ready for Phase 8 when **all** of the following are true:

- [ ] I implemented the DPO loss from scratch (Challenge A) and verified it works
- [ ] I built a preference dataset using GPT-4 as judge (Challenge B)
- [ ] I ran DPO training with TRL and my fine-tuned model improved on alignment metrics
- [ ] I can write the DPO loss formula from memory
- [ ] I scored 22/25 or higher on the Self-Assessment Quiz
- [ ] I can answer at least 16/20 Interview Questions correctly
- [ ] I can explain why DPO doesn't need a reward model (conceptually and mathematically)

---

### 10. Revision Summary

```
RLHF PIPELINE (3 stages)
─────────────────────────────────────────────────────
Stage 1:  SFT on demonstrations (quality data, few examples)
Stage 2:  Train reward model on preference pairs (Bradley-Terry loss)
Stage 3:  PPO: maximise r(x,y) - β·KL(π_θ||π_SFT)

DPO
─────────────────────────────────────────────────────
Loss: -E[log σ(β·(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
β:    KL strength  (typical: 0.1–0.3)
Data: (prompt, chosen, rejected) triplets
No:   reward model, PPO, value model, online generation
Yes:  two model copies (policy + frozen reference), offline training

COMPARISON
─────────────────────────────────────────────────────
PPO:  complex, online, powerful, unstable → frontier labs
DPO:  simple, offline, stable, slightly weaker → practical choice
IPO:  DPO + regularisation → better for large preference datasets
RLAIF: use GPT-4 instead of humans for preferences → scalable
```

---

### 11. Next Phase Prerequisites

**What Phase 8 (Advanced Topics) requires from Phase 7b:**

| Phase 7b Concept | How Phase 8 Uses It |
|-----------------|----------------------|
| Model training stability | Advanced training (diffusion, multimodal) requires same stability knowledge |
| RLHF reward concept | Constitutional AI in Phase 9 builds on alignment understanding |
| Gradient flow in fine-tuning | Diffusion model training uses same gradient principles |
| DPO data generation | Understanding how to generate synthetic training data (used in distillation) |
| Evaluation methodology | Advanced topics all require rigorous evaluation |

**The critical dependency**: Phase 7 (fine-tuning + alignment) represents the complete picture of how LLMs are created and refined. Phases 8-10 build on this: Phase 8 assumes you understand model training deeply, Phase 9 assumes you can evaluate models rigorously.

---

*Phase 7b | Part of the [GenAI Engineer Roadmap](./00_README.md)*
