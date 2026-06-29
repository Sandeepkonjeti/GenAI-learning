# Phase 3 Flashcards — Deep Learning & PyTorch

[← Flashcards Index](./README.md) | [Phase 3 Files](../04_Phase3_Part1_Deep_Learning_Theory.md) | [Cheat Sheet](../CheatSheets/Phase3_DL_PyTorch.md)

---

**ID**: P3-001
**Front**: Walk through backpropagation for a single linear layer z = Wx + b. What are dL/dW, dL/dx, dL/db?
**Back**: Given dL/dz (gradient from next layer): dL/dW = dL/dz · x^T (outer product, shape same as W). dL/dx = W^T · dL/dz (for chain rule to previous layer). dL/db = sum(dL/dz, axis=0) (sum over batch). Key insight: gradients flow backward through transposed weight matrices.
**Difficulty**: Advanced
**Category**: Math
**Tags**: #phase3 #backprop #gradients #linear-layer

---

**ID**: P3-002
**Front**: What is the vanishing gradient problem and how is it solved?
**Back**: Deep networks: gradients become tiny (→0) as they propagate backward through many layers with sigmoid/tanh activations (derivatives ≤ 0.25). Solutions: (1) ReLU activation — gradient = 1 for x>0. (2) Residual connections — gradient "highway" bypasses layers. (3) Batch/layer normalization — keeps activations in good range. (4) Gradient clipping. (5) Better initialization (Xavier/He).
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase3 #vanishing-gradient #backprop

---

**ID**: P3-003
**Front**: What is Xavier (Glorot) initialization and when do you use it?
**Back**: Xavier: W ~ Uniform(-√(6/(n_in+n_out)), √(6/(n_in+n_out))). Keeps variance of activations roughly constant across layers (with tanh/sigmoid). He initialization: W ~ N(0, 2/n_in). Better for ReLU (ReLU kills half the variance, so start with 2× more). PyTorch: `nn.init.xavier_uniform_(layer.weight)` or `nn.init.kaiming_normal_`.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase3 #initialization #xavier #he

---

**ID**: P3-004
**Front**: What does `optimizer.zero_grad()` do and why must you call it before each backward pass?
**Back**: Clears the `.grad` attribute of all parameters to zero. PyTorch ACCUMULATES gradients by default — each `.backward()` ADDS to existing gradients. If you don't zero_grad: gradients from multiple batches accumulate → incorrect updates. Intentional accumulation use case: gradient accumulation for large effective batch sizes (call zero_grad every N steps instead of every step).
**Difficulty**: Beginner
**Category**: Coding
**Tags**: #phase3 #pytorch #training-loop #gradients

---

**ID**: P3-005
**Front**: What is `torch.no_grad()` and when do you use it?
**Back**: Context manager that disables gradient computation — saves memory and speeds up computation. Use for: (1) Inference/evaluation — don't need gradients. (2) Any computation using model that shouldn't be part of backward graph. Use `model.eval()` separately (affects dropout/batchnorm behavior). `@torch.inference_mode()` is the faster version in PyTorch 1.9+.
**Difficulty**: Beginner
**Category**: Coding
**Tags**: #phase3 #pytorch #inference #no_grad

---

**ID**: P3-006
**Front**: What is the difference between `model.train()` and `model.eval()`?
**Back**: `model.train()`: enables Dropout (randomly drops neurons) and BatchNorm (uses batch statistics). `model.eval()`: disables Dropout (all neurons active) and BatchNorm uses running statistics (not batch stats). Always: call `model.eval()` before inference/validation; call `model.train()` before each training epoch. Forgetting eval() → validation metrics are artificially noisy.
**Difficulty**: Beginner
**Category**: Coding
**Tags**: #phase3 #pytorch #train-eval

---

**ID**: P3-007
**Front**: Why is AdamW preferred over Adam for training transformers?
**Back**: Adam applies L2 regularization through the gradient update scaling — this is mathematically incorrect because adaptive scaling causes weight decay to depend on gradient history. AdamW separates weight decay from gradient update: θ ← θ·(1-λ) - α·grad_update. Empirically: AdamW consistently outperforms Adam for transformers (Loshchilov & Hutter 2019). Always use AdamW with `weight_decay=0.01`.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase3 #pytorch #adamw #optimizer

---

**ID**: P3-008
**Front**: What is gradient clipping and what value is typically used for transformers?
**Back**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`. Clips the global gradient norm to max_norm. Why: prevents "exploding gradients" where one large gradient step destabilizes training. Transformers: typical value is 1.0. Signs of exploding gradients: loss suddenly spikes, NaN in loss. Call AFTER `.backward()` and BEFORE `.step()`.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase3 #pytorch #gradient-clipping #training

---

**ID**: P3-009
**Front**: What is batch normalization? Write the formula and explain the learnable parameters.
**Back**: BN: normalize across batch dimension for each feature: x̂ = (x - μ_batch) / √(σ²_batch + ε). Then: y = γ·x̂ + β. γ (scale) and β (shift) are learnable parameters — allow the network to undo normalization if needed. At inference: use running mean/var (computed during training). Use LayerNorm for transformers (normalizes across features, not batch).
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase3 #batch-norm #normalization

---

**ID**: P3-010
**Front**: What is the difference between BatchNorm and LayerNorm?
**Back**: BatchNorm: normalizes across BATCH dimension — statistics computed per-feature across samples. Depends on batch size; fails with batch=1. LayerNorm: normalizes across FEATURE dimension — statistics computed per-sample across features. Batch-size independent. Use BatchNorm for: CNNs (large batches). Use LayerNorm for: transformers, RNNs, any variable batch size. PyTorch: `nn.BatchNorm1d(d)` vs `nn.LayerNorm(d)`.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase3 #batch-norm #layer-norm

---

**ID**: P3-011
**Front**: What is dropout regularization? What probability p means?
**Back**: During training: randomly zero out each neuron with probability p. Scales remaining activations by 1/(1-p) to maintain expected value. At evaluation: no dropout (all neurons active). `nn.Dropout(p=0.1)` in PyTorch — 10% of neurons dropped. Standard values: 0.1 for transformers, 0.5 for dense layers in vision. Intuition: forces redundancy — can't rely on any single neuron.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase3 #dropout #regularization

---

**ID**: P3-012
**Front**: Write a minimal PyTorch training loop. What are the 5 essential steps?
**Back**: (1) `optimizer.zero_grad()` — clear old gradients. (2) `outputs = model(X)` — forward pass. (3) `loss = criterion(outputs, y)` — compute loss. (4) `loss.backward()` — backprop. (5) `optimizer.step()` — update parameters. Optional but important: `clip_grad_norm_` between (4) and (5). `scheduler.step()` after optimizer.step().
**Difficulty**: Beginner
**Category**: Coding
**Tags**: #phase3 #pytorch #training-loop

---

**ID**: P3-013
**Front**: What is `view()` vs `reshape()` in PyTorch?
**Back**: Both change tensor shape. `view()`: requires contiguous memory — fails if not contiguous (after transpose, permute). Returns view (no copy). `reshape()`: works on non-contiguous tensors — may copy if needed. In practice: prefer `reshape()` for safety. Use `.contiguous()` before `view()` if needed: `t.contiguous().view(...)`. `.shape` to check dimensions.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase3 #pytorch #tensors

---

**ID**: P3-014
**Front**: What is a DataLoader and what are its key parameters?
**Back**: `DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)`. `shuffle=True` for training, `False` for val/test. `num_workers`: parallel data loading processes (4-8 typical). `pin_memory=True`: faster CPU→GPU transfer. `collate_fn`: custom batching (needed for variable-length sequences). `drop_last=True`: discard incomplete final batch (helpful for batch norm).
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase3 #pytorch #dataloader #dataset

---

**ID**: P3-015
**Front**: What is a residual (skip) connection and why does it help training?
**Back**: Residual connection: output = x + F(x) instead of just F(x). From ResNet paper (He et al. 2015). Why it helps: (1) Gradient highway — gradient flows directly to early layers through +, bypassing F(x). (2) Easy to learn identity (F(x)≈0). (3) Depth becomes "free" — adding more layers with residuals can only help (worst case: F(x)≈0). All modern transformers use residual connections.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase3 #residual #skip-connection #resnet

---

**ID**: P3-016
**Front**: What is autograd in PyTorch and how does it work?
**Back**: Autograd: automatic differentiation engine. During forward pass: builds a directed acyclic computational graph (DAG) of operations. Each tensor with `requires_grad=True` records its operation and inputs. During `.backward()`: traverses graph in reverse, applying chain rule at each node. `detach()`: removes tensor from graph. `with torch.no_grad()`: disables graph building entirely.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase3 #pytorch #autograd #computational-graph

---

**ID**: P3-017
**Front**: What is weight tying and why is it used in language models?
**Back**: Weight tying: share the token embedding matrix with the output projection (language model head). `lm_head.weight = token_embeddings.weight`. Benefits: (1) Reduces parameters by ~embedding_dim × vocab_size (≈ 125M for GPT-2). (2) Forces consistency: tokens used similarly in input appear similarly in output. Used in: GPT-2, LLaMA, most language models.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase3 #pytorch #language-model #weight-tying

---

**ID**: P3-018
**Front**: What is mixed precision training (float16/bfloat16)?
**Back**: Train with 16-bit floats instead of 32-bit: 2× less memory, 2× faster on modern GPUs. PyTorch: `torch.cuda.amp.autocast()`. Problem: fp16 has smaller range → NaN in gradients. Solution: GradScaler — scale loss up before backward, scale down before optimizer step. BF16 (bfloat16): same range as fp32 but less precision — preferred for transformers (no GradScaler needed). `torch.set_float32_matmul_precision('high')` for TF32.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase3 #pytorch #mixed-precision #bf16

---

**ID**: P3-019
**Front**: What is a convolutional neural network? What are the key parameters of Conv2d?
**Back**: CNN: uses shared sliding filter (kernel) to detect local patterns — translation invariant. `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`. kernel_size: filter size (3×3 common). stride: step size. padding: adds zeros to preserve spatial dimensions. `padding='same'` for input=output size. Output size: (W - K + 2P)/S + 1. CNNs: hierarchical features (edges → textures → objects).
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase3 #pytorch #CNN #conv2d

---

**ID**: P3-020
**Front**: Explain the purpose of the learning rate scheduler. Name two common types.
**Back**: Adjusts LR during training to improve convergence. Too high → diverge. Too low → slow. Schedulers: (1) CosineAnnealingLR: LR decreases following cosine curve to η_min — smooth, widely used for transformers. (2) Linear warmup + decay: LR increases linearly for N steps (warmup), then decreases — standard for BERT/LLM fine-tuning. (3) ReduceLROnPlateau: reduce LR when val loss plateaus. Call `scheduler.step()` after each epoch.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase3 #pytorch #learning-rate #scheduler

---

**ID**: P3-021
**Front**: What is the `nn.Embedding` layer and how does it work?
**Back**: `nn.Embedding(num_embeddings, embedding_dim)`. Learnable lookup table: integer token ID → dense vector. Internally: weight matrix of shape (vocab_size, d_model). Forward: `emb_layer(token_ids)` performs an indexed lookup (equivalent to one-hot @ weight, but efficient). Output: (batch, seq_len, d_model). Initialized randomly; learned during training. Language models + transformers start here.
**Difficulty**: Beginner
**Category**: Coding
**Tags**: #phase3 #pytorch #embedding #nlp

---

**ID**: P3-022
**Front**: What is gradient accumulation and when is it useful?
**Back**: Update weights every N mini-batches instead of every batch. Simulates large batch size without requiring large GPU memory. Pattern: `loss = loss / N; loss.backward()` — accumulate N times, then `optimizer.step(); optimizer.zero_grad()`. Use when: effective batch size needed > what fits in GPU memory. Common in LLM fine-tuning: batch_size=1, accumulate_steps=8 → effective batch=8.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase3 #pytorch #gradient-accumulation #training

---

**ID**: P3-023
**Front**: What is the difference between `loss.backward()` and `torch.autograd.grad()`?
**Back**: `loss.backward()`: computes gradients for ALL leaf tensors with requires_grad=True; stores in `.grad` attribute; accumulates. `torch.autograd.grad(loss, params)`: computes gradients for SPECIFIC tensors; returns tuple (no side effect on .grad). Use `autograd.grad` for: second-order gradients, meta-learning (MAML), custom gradient flows.
**Difficulty**: Advanced
**Category**: Coding
**Tags**: #phase3 #pytorch #autograd #advanced

---

**ID**: P3-024
**Front**: What is the purpose of `torch.nn.utils.clip_grad_norm_` and when should you call it?
**Back**: Computes the global L2 norm of all gradients, then scales all gradients down proportionally if norm > max_norm. Call: AFTER `loss.backward()` and BEFORE `optimizer.step()`. Typical value for transformers: 1.0. If gradients explode (loss NaN), lower to 0.1. Also useful diagnostic: log the gradient norm — sudden spikes indicate instability. Returns the gradient norm before clipping.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase3 #pytorch #gradient-clipping

---

**ID**: P3-025
**Front**: What is `@torch.compile` (PyTorch 2.0) and what speedup does it provide?
**Back**: JIT compiler for PyTorch models using TorchDynamo + TorchInductor backends. Traces the computation graph and applies kernel fusion, memory optimization. `model = torch.compile(model)`. Typical speedup: 1.5–3× for training, up to 4× for inference. Works best on repeated computation patterns. May increase compilation time on first call. Not all ops supported — check output for graph breaks.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase3 #pytorch #compile #optimization

---

**ID**: P3-026
**Front**: Why does using `model.parameters()` matter vs iterating over `model.state_dict()`?
**Back**: `model.parameters()`: returns only TRAINABLE parameters (leaf tensors with requires_grad=True) — used for optimizer. `model.state_dict()`: returns ALL tensors (trainable + frozen + buffers like running_mean in BatchNorm) — used for saving/loading. `model.named_parameters()`: name + param pairs — useful for filtering (e.g., freeze certain layers).
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase3 #pytorch #parameters

---

**ID**: P3-027
**Front**: What are the four main steps of LSTM that differ from vanilla RNN?
**Back**: LSTM has 4 gates: Forget gate (f): which past state to forget. Input gate (i): what new info to add. Cell update (g): candidate new cell values. Output gate (o): what to expose from cell. Cell state c_t = f⊙c_{t-1} + i⊙g (protected from vanishing gradient). h_t = o⊙tanh(c_t). The cell state is the "long-term memory highway." LSTMs largely replaced by transformers for NLP.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase3 #LSTM #RNN

---

**ID**: P3-028
**Front**: How does max pooling work in CNNs and what does it achieve?
**Back**: MaxPool2d(kernel_size=2, stride=2): takes maximum value in each 2×2 window → halves spatial dimensions. Benefits: (1) Spatial invariance — small translations don't affect output. (2) Reduces computation. (3) Increases receptive field. Alternative: GlobalAveragePooling — average entire feature map to a single value per channel. Transformers use attention instead of pooling.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase3 #CNN #pooling

---

**ID**: P3-029
**Front**: What is `nn.Module.register_buffer` and when do you use it?
**Back**: Registers a tensor that is part of model state but NOT a trainable parameter. Example: causal mask in transformers, running statistics in BatchNorm. Saved/loaded with `state_dict()`, moved to correct device with `model.to(device)`. Usage: `self.register_buffer('causal_mask', torch.tril(torch.ones(T, T)))`. Without register_buffer: manual device management required.
**Difficulty**: Advanced
**Category**: Coding
**Tags**: #phase3 #pytorch #module #buffers

---

**ID**: P3-030
**Front**: What is the receptive field of a CNN and how does it grow with depth?
**Back**: Receptive field: the area of the input that influences a given output neuron. Single conv (3×3, stride=1): receptive field = 3×3. After k conv layers: receptive field ≈ 2k+1 (linear growth). After pooling/stride=2: receptive field doubles. Why it matters: deep features need large receptive fields to capture global context. Alternative: dilated convolutions grow receptive field exponentially.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase3 #CNN #receptive-field

---

**ID**: P3-031
**Front**: What is transfer learning and how does fine-tuning differ from feature extraction?
**Back**: Transfer learning: use pretrained model's representations for a new task. Feature extraction: freeze pretrained layers, train only a new head (fast, less data needed). Fine-tuning: unfreeze some/all layers and train with small LR (better performance, needs more data). Best practice: always start with feature extraction to verify task-model compatibility; then fine-tune upper layers; then full fine-tuning if data allows.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase3 #transfer-learning #fine-tuning

---

**ID**: P3-032
**Front**: What is the exploding gradient problem and how does it differ from vanishing gradients?
**Back**: Exploding: gradients become very large → parameter updates too large → loss diverges (NaN). Vanishing: gradients become very small → early layers don't learn. Exploding common in: RNNs on long sequences, deep networks without proper init. Solutions for exploding: gradient clipping. Solutions for both: residual connections, layer norm, careful initialization, shorter sequences, LSTM.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase3 #exploding-gradient #vanishing-gradient

---

**ID**: P3-033
**Front**: What does `torch.Tensor.detach()` do and when is it needed?
**Back**: Creates a new tensor sharing same storage but with `requires_grad=False` and no autograd history. Use cases: (1) Stop gradient flow for part of computation (e.g., target network in RL, stop-gradient in BYOL). (2) Convert to NumPy: `t.detach().cpu().numpy()`. (3) Logging intermediate values without storing graph. Note: `detach()` ≠ `clone()` — detach shares memory, clone copies.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase3 #pytorch #detach #gradients

---

**ID**: P3-034
**Front**: What is the difference between `torch.save(model, path)` and `torch.save(model.state_dict(), path)`?
**Back**: `save(model, path)`: saves entire model including architecture (uses pickle). Brittle: breaks if class definition changes. `save(model.state_dict(), path)`: saves only weights dict. Portable: load into any compatible architecture. ALWAYS prefer state_dict. Load: `model.load_state_dict(torch.load(path, map_location='cpu'))`. Add `weights_only=True` in PyTorch 2.x for security.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase3 #pytorch #saving #checkpointing

---

**ID**: P3-035
**Front**: How does the attention mechanism address the limitation of fixed-size context in RNNs?
**Back**: RNNs: compress entire input into fixed-size hidden state → information bottleneck. Attention (Bahdanau 2015): at each decode step, compute weighted sum over ALL encoder hidden states. Weight = similarity(decoder_state, encoder_state_i) → softmax. Model can "look back" at any input position. Transformers extend this: every token attends to every other token in parallel — O(n²) but parallelizable.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase3 #attention #RNN #transformers

---

**ID**: P3-036
**Front**: What is gradient checkpointing and when should you use it?
**Back**: Trade compute for memory: instead of caching all intermediate activations for backward pass, recompute them during backward. Reduces memory by ~√n (n = layers). Cost: ~33% more compute. Use when: training with long sequences or large batch sizes causes OOM. PyTorch: `torch.utils.checkpoint.checkpoint(function, *inputs)`. HuggingFace: `model.gradient_checkpointing_enable()`.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase3 #pytorch #gradient-checkpointing #memory

---

**ID**: P3-037
**Front**: What are the three types of activation functions most commonly used today and in which architectures?
**Back**: (1) ReLU: CNNs, MLPs. Simple, fast, but dying ReLU. (2) GELU: transformers (BERT, GPT-2/3). Smooth, probabilistic — `x·Φ(x)`. (3) SiLU/Swish: LLaMA, Mistral. `x·σ(x)`. Often paired with GLU (Gated Linear Units) as SwiGLU in LLaMA FFN. Modern trend: gated activations outperform ungated; GELU/SiLU preferred for LLMs.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase3 #activation #GELU #SiLU #ReLU

---

**ID**: P3-038
**Front**: What is the difference between a shallow network and a deep network? Why go deep?
**Back**: Shallow (1–2 hidden layers): universal approximation theorem says it can represent any function — but may need exponentially many neurons. Deep (many layers): computes hierarchical features; each layer builds on previous (edges → shapes → objects). Empirically: deep networks generalize better with fewer total parameters. Residual connections make depth "free." Depth >> width for same parameter count.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase3 #deep-learning #depth #universal-approximation

---

**ID**: P3-039
**Front**: How do you freeze layers in PyTorch for transfer learning?
**Back**: `for param in model.base_model.parameters(): param.requires_grad = False`. The optimizer only updates parameters with `requires_grad=True`. Verify: `model.print_trainable_parameters()` (HuggingFace) or `sum(p.numel() for p in model.parameters() if p.requires_grad)`. Unfreeze later: `param.requires_grad = True`. For gradual unfreezing: unfreeze one layer group at a time.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase3 #pytorch #freeze #transfer-learning

---

**ID**: P3-040
**Front**: What is the purpose of the warmup phase in LR scheduling for transformers?
**Back**: LR starts at 0, increases linearly to target LR over first N steps (warmup), then decays. Without warmup: early parameter updates with high LR and random gradients → unstable training. With warmup: gradients stabilize first (model finds reasonable region), then LR increases. Standard for BERT: warmup over first 10% of training steps. LLM fine-tuning: 100–500 warmup steps typical.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase3 #pytorch #warmup #learning-rate #transformer
