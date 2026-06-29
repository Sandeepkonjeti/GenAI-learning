# Phase 8 Flashcards — Advanced Topics

[← Flashcards Index](./README.md) | [Phase 8 File](../09_Phase8_Advanced_Topics.md) | [Cheat Sheet](../CheatSheets/Phase8_Advanced.md)

---

**ID**: P8-001
**Front**: What is Mixture of Experts (MoE) and how does it scale models efficiently?
**Back**: MoE: replace each dense FFN layer with N expert FFNs + a router. Router (softmax over experts) selects top-K experts per token. Only K out of N experts are activated per token — compute = K × FFN, parameters = N × FFN. Example: Mixtral-8x7B has 8 experts, uses top-2: compute = 2 × 7B ≈ 12.6B active params but total = 47B params (more knowledge). Achieves: large model (knowledge) + small model (inference cost).
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase8 #MoE #mixture-of-experts

---

**ID**: P8-002
**Front**: What is PagedAttention in vLLM and what problem does it solve?
**Back**: Problem: variable-length sequences mean KV cache has different sizes. Standard allocation: reserve max_seq_len × memory per sequence → massive fragmentation (up to 80% wasted). PagedAttention: inspired by OS virtual memory. KV cache divided into fixed-size pages. Non-contiguous pages mapped to logical sequence positions. Result: near-zero fragmentation, much higher batch sizes. vLLM = 5-10× more throughput than naïve HuggingFace generate.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase8 #vLLM #PagedAttention #inference

---

**ID**: P8-003
**Front**: What is the difference between post-training quantization (PTQ) and quantization-aware training (QAT)?
**Back**: PTQ: quantize a pretrained model after training. Fast, no training needed. Quality loss depends on how well weights/activations follow quantization-friendly distributions. Examples: GPTQ, AWQ, llama.cpp (GGUF). QAT: train (or fine-tune) with quantization simulation in the loop — model learns to be robust to quantization. Better quality but requires retraining. For LLMs: PTQ (GPTQ/AWQ) is standard — QAT too expensive for 7B+ models.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase8 #quantization #PTQ #QAT

---

**ID**: P8-004
**Front**: What is GPTQ quantization and how does it work?
**Back**: GPTQ (Frantar et al. 2022): group-wise PTQ. Algorithm: for each weight matrix, use a calibration dataset to compute optimal quantization error. Solve: minimize ||WX - ˆWX||² where ˆW is quantized. Uses Hessian information (second-order optimization). Result: 4-bit weights with minimal quality loss. Faster and better than naive round-to-nearest. AWQ (Lin et al. 2023): activation-aware weight quantization — identifies important weights (those multiplied by large activations) and quantizes them less aggressively.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase8 #GPTQ #AWQ #quantization

---

**ID**: P8-005
**Front**: What is speculative decoding and what speedup does it provide?
**Back**: LLM generates 1 token per forward pass — bottleneck. Speculative decoding: (1) Small draft model generates K candidate tokens quickly. (2) Large target model verifies all K in ONE parallel forward pass. (3) Accept tokens until first disagreement; reject rest. Result: if draft model is often right → 2-3× throughput for same output quality. Requirements: same vocabulary, draft must be faster (~10-100×). Draft model: smaller version of target (LLaMA-3-1B draft for LLaMA-3-70B).
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase8 #speculative-decoding #inference

---

**ID**: P8-006
**Front**: How do you serve a model with vLLM? What are the key configuration parameters?
**Back**: `from vllm import LLM, SamplingParams`. `llm = LLM(model="...", tensor_parallel_size=2, gpu_memory_utilization=0.9, dtype="bfloat16", max_model_len=4096)`. `params = SamplingParams(temperature=0.0, max_tokens=512)`. `outputs = llm.generate(prompts, params)`. REST API: `python -m vllm.entrypoints.openai.api_server --model ... --port 8000`. OpenAI-compatible — swap base_url to point to vLLM server. `tensor_parallel_size`: GPUs to shard across. `gpu_memory_utilization`: fraction for KV cache.
**Difficulty**: Advanced
**Category**: Coding
**Tags**: #phase8 #vLLM #serving #inference

---

**ID**: P8-007
**Front**: What is continuous batching in inference servers?
**Back**: Traditional batching: server waits until batch is full, processes together, returns all. Long requests block short ones. Continuous batching: new requests "join" the batch mid-generation. Finished sequences immediately release their slot. Result: GPU is always busy, tail latency much lower. vLLM uses continuous batching. Critical for production: without it, one long request stalls all concurrent users. Related: iteration-level scheduling vs request-level scheduling.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase8 #continuous-batching #inference #vLLM

---

**ID**: P8-008
**Front**: What is tensor parallelism and pipeline parallelism for distributed inference?
**Back**: Tensor parallelism: split individual matrices across GPUs (row/column parallel). Each GPU holds part of each layer. Communication: all-reduce after each matmul. vLLM `tensor_parallel_size=4`: splits across 4 GPUs. Pipeline parallelism: split model layers across GPUs. GPU0: layers 0-8, GPU1: 9-16, etc. Communication: only activations between pipeline stages. Tensor parallelism: more communication but lower latency. Pipeline: less communication but higher latency (bubble problem).
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase8 #tensor-parallelism #pipeline-parallelism

---

**ID**: P8-009
**Front**: What is the CLIP model and how does it enable zero-shot image classification?
**Back**: CLIP (Contrastive Language-Image Pretraining, OpenAI 2021): train image encoder + text encoder jointly with contrastive loss — similar (image, text) pairs → close in embedding space. Zero-shot: embed class names as text (e.g., "a photo of a cat"). Embed image. Find closest class text embedding → that's the prediction. No task-specific training needed. Used as: vision encoder in LLaVA, GPT-4V. Also: image retrieval with text queries (same embedding space).
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase8 #CLIP #multimodal #vision

---

**ID**: P8-010
**Front**: What are State Space Models (SSMs) / Mamba and how do they differ from transformers?
**Back**: SSMs (Mamba, 2023): model sequences as continuous-time state space: h'(t) = Ah(t) + Bx(t), y(t) = Ch(t). Discretized for sequences. Key property: O(n) time AND memory (vs O(n²) for attention). Selective SSM (Mamba): input-dependent state transitions — can "forget" irrelevant tokens. Trade-offs: less expressive than full attention for in-context learning; better at very long sequences. Hybrid models (Jamba, Mamba-2 + attention): best of both worlds.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase8 #Mamba #SSM #state-space

---

**ID**: P8-011
**Front**: What is the NF4 data type used in QLoRA?
**Back**: NF4 (Normal Float 4-bit): quantization format designed for normally-distributed weights. Instead of uniform 4-bit quantization: compute the 16 values that minimize expected quantization error under a Gaussian distribution. Theoretically optimal for normally distributed weights (which LLM weights approximately are). vs INT4: uniform levels, not optimized for Gaussian. NF4 + double quantization + paged adamw = QLoRA. Available in `bitsandbytes` library.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase8 #NF4 #quantization #QLoRA

---

**ID**: P8-012
**Front**: What is the purpose of `tensor_parallel_size` in vLLM and what's the maximum useful value?
**Back**: `tensor_parallel_size=N`: shard model weights across N GPUs. Each GPU holds 1/N of each weight matrix. Useful up to: 8 (single node). Beyond 8: network bandwidth between nodes becomes bottleneck (NVLink: 600GB/s; InfiniBand: 200GB/s for cross-node). For 8B model: fits on 1 GPU (A100 80GB). For 70B: need 4 GPUs (4×20B = 80GB activations + weights). For 405B: 8 GPUs minimum. Rule: use minimum GPUs needed to fit model.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase8 #vLLM #tensor-parallel #GPU

---

**ID**: P8-013
**Front**: What is LLaVA and how does it connect vision and language?
**Back**: LLaVA (Large Language and Vision Assistant): CLIP vision encoder → projection layer → LLM decoder. Architecture: (1) CLIP ViT-L/14 encodes image → 256 patch embeddings. (2) MLP projection layer → d_model vectors (same space as text tokens). (3) These visual tokens are prepended to text tokens → LLM. Training: (1) Pretrain projection layer on image-caption pairs (freeze both encoders). (2) Fine-tune on visual instruction data. Modern: LLaVA-NeXT uses higher-res tiles.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase8 #LLaVA #multimodal #vision

---

**ID**: P8-014
**Front**: What is prefix caching and how does it reduce LLM costs?
**Back**: Prefix caching: cache the computed KV tensors for a repeated system prompt prefix. On subsequent requests with same prefix: skip computing prefix KV (just load from cache). Savings: if system prompt is 1000 tokens and context is 100 tokens → 90% of compute cached. APIs: Anthropic prompt caching (up to 90% discount for cached prefix), OpenAI caching (50% discount). Implementation: hash prefix → store KV in Redis/disk. Essential when: long system prompts, same prompt for many users.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase8 #prefix-caching #cost #KV-cache

---

**ID**: P8-015
**Front**: What is the Whisper model and what task does it solve?
**Back**: Whisper (OpenAI 2022): encoder-decoder transformer for automatic speech recognition (ASR). Trained on 680K hours of multilingual audio. Encoder: process mel spectrogram audio. Decoder: generate text transcription autoregressively. Features: multilingual, translation, timestamps, punctuation. Use: `whisper.transcribe("audio.mp3")`. Or HuggingFace: `pipeline("automatic-speech-recognition", "openai/whisper-large-v3")`. State-of-art for ASR without fine-tuning on most languages.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase8 #whisper #ASR #multimodal

---

**ID**: P8-016
**Front**: What is the load balancing loss in MoE models?
**Back**: Without load balancing: router learns to always send tokens to the same expert (expert collapse). Load balancing auxiliary loss: L_aux = α Σ_e (f_e × P_e) where f_e = fraction of tokens routed to expert e, P_e = average routing probability for expert e. Minimizing L_aux encourages uniform expert utilization. Added to main language modeling loss: L_total = L_LM + α_aux × L_aux. Typical α_aux = 0.01–0.001. During inference: not needed (no training).
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase8 #MoE #load-balancing #training

---

**ID**: P8-017
**Front**: What is Flash Attention 2 and what improvements does it make over Flash Attention 1?
**Back**: Flash Attention 2 (Dao 2023): improved IO-aware attention. Changes: (1) Better parallelism — distributes work better across CUDA thread blocks. (2) Reduced non-matmul FLOPs. (3) Better support for different head dimensions. (4) Supports causal + non-causal attention. Speedup: 2× over FA1 on A100. Memory: still O(n) — doesn't materialize full attention matrix. Now standard: most frameworks use FA2 by default. `pip install flash-attn`. LLaMA: `attn_implementation="flash_attention_2"`.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase8 #flash-attention #optimization #GPU

---

**ID**: P8-018
**Front**: What is GPT-4V / multimodal LLM architecture at a high level?
**Back**: Multimodal LLM: extends text-only LLM to process images (and sometimes audio, video). Architecture variants: (1) Adapter approach (LLaVA): visual encoder → projection → LLM text. (2) Cross-attention (Flamingo): visual features as cross-attention K,V in LLM layers. (3) Early fusion: process image patches as tokens alongside text from the start. GPT-4o: end-to-end native multimodal (architecture not fully published). Quality progression: GPT-4V < LLaVA-NeXT < Gemini 1.5 < GPT-4o for vision tasks.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase8 #multimodal #GPT-4V #vision

---

**ID**: P8-019
**Front**: What is GGUF format (formerly GGML) and when do you use it?
**Back**: GGUF (GPT-Generated Unified Format): model format for llama.cpp inference. Supports: 2-bit to 8-bit quantization. Runs on: CPU (slow but works), Apple Silicon (Metal), CUDA. Tool: `llama.cpp`, Ollama. Use when: (1) No GPU — CPU inference. (2) Local privacy (no API calls). (3) Embed in desktop app. Not for: production serving (use vLLM with GPTQ/AWQ). Convert: `llama_cpp.convert_hf_to_gguf("model_path", outtype="q4_k_m")`. Q4_K_M: 4-bit mixed quantization, good quality/size balance.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase8 #GGUF #quantization #llama-cpp

---

**ID**: P8-020
**Front**: What is Retrieval-Augmented Fine-Tuning (RAFT)?
**Back**: RAFT (2024): fine-tuning technique that trains model to be a domain expert that can use retrieved context. Training: for each (question, golden document, distractor documents), train model to identify and use the golden document while ignoring distractors. Also train on some examples without relevant documents (teaches graceful abstention). Combines: RAG's knowledge injection + fine-tuning's consistent behavior. Better than: RAG alone (no task adaptation) or fine-tuning alone (no retrieval).
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase8 #RAFT #fine-tuning #RAG

---

**ID**: P8-021
**Front**: What is the difference between model throughput and model latency?
**Back**: Latency: time for one request to complete (seconds). Throughput: total tokens generated per second across all concurrent requests (tokens/s). Trade-off: large batch → high throughput, high latency. Small batch → low latency, low throughput. Metrics: TTFT (time to first token) ← latency. TPOT (time per output token). Tokens/s ← throughput. Continuous batching optimizes throughput without hurting tail latency. Choose: user-facing → optimize latency. Batch jobs → optimize throughput.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase8 #latency #throughput #inference

---

**ID**: P8-022
**Front**: What is RLHF from AI feedback (RLAIF) and how does it scale?
**Back**: RLAIF: use AI (usually stronger LLM) as preference judge instead of human labelers. Process: generate response pairs → ask Claude/GPT-4 to rate which is better → use AI ratings to train reward model → PPO or DPO. Why it scales: (1) AI can rate millions of pairs cheaply. (2) No annotation latency. (3) More consistent than humans. Quality: RLAIF competitive with RLHF (Bai et al. 2022, Lee et al. 2023). Risk: inherits biases of the judge model. Constitutional AI uses RLAIF.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase8 #RLAIF #alignment #scaling

---

**ID**: P8-023
**Front**: What is the memory cost of a 7B parameter model at different precisions?
**Back**: Rule: 1B params ≈ 2GB (fp16/bf16) or 1GB (int8) or 0.5GB (int4). LLaMA-3-7B: ~14GB (bf16), ~7GB (int8), ~3.5GB (int4/NF4). Plus: activations (~2-4GB for inference), KV cache (variable). A100-40GB: can run 7B bf16 + some batch. A100-80GB: can run 30B bf16 or 70B int4. Consumer GPU (RTX 3090 24GB): 7B bf16, or 13B int4. Use QLoRA: base model in int4 + adapters in bf16.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase8 #memory #GPU #model-size

---

**ID**: P8-024
**Front**: What is YaRN and why does it matter for long-context LLMs?
**Back**: YaRN (Yet another RoPE extensioN, 2023): technique to extend LLM context window beyond training length. RoPE is position-dependent — positions > training length are out-of-distribution. YaRN: scale RoPE base frequency + interpolate within training range + extrapolate carefully beyond. Result: extend LLaMA-3-8K context to 128K with minimal quality loss. Used by: LLaMA-3.1 (128K context). Alternative: LongRoPE. Without extension: model quality degrades rapidly beyond training context length.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase8 #YaRN #RoPE #long-context

---

**ID**: P8-025
**Front**: What is the H2O (Heavy-Hitter Oracle) technique for KV cache compression?
**Back**: H2O: observation that attention is sparse — a small subset of KV pairs ("heavy hitters") receive most attention mass. H2O: keep only the top-k highest attention-score KV pairs + recent tokens (streaming LLM style). Rest are evicted. Memory: O(k) instead of O(n). Quality: near-zero degradation if k is large enough (~20% of sequence). Use for: serving very long context with limited GPU memory. Related: Scissorhands, SnapKV. Research area: KV cache eviction is active 2024 topic.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase8 #KV-cache #compression #H2O

---

**ID**: P8-026
**Front**: What is tensor quantization calibration and why is it needed?
**Back**: Calibration: run a small dataset (128-512 samples) through the model to observe activation distributions before quantizing. Why: weights can be quantized statically (known at conversion), but activations vary by input. Calibration: (1) Find per-layer activation scales. (2) Determine which weights/activations can tolerate more aggressive quantization. (3) Outlier detection (some activations have extreme values — need special handling). Tools: AutoGPTQ, AutoAWQ use calibration datasets (WikiText, ShareGPT).
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase8 #quantization #calibration

---

**ID**: P8-027
**Front**: What is the ColPali model and what does it enable?
**Back**: ColPali (2024): late interaction retrieval for PDFs and documents using vision. Instead of OCR + text extraction → embed each PDF page as an IMAGE → compare query token embeddings against all page patch embeddings. Uses ColBERT-style late interaction scoring. Why revolutionary: PDFs have charts, tables, layouts that OCR loses — visual retrieval captures them. Pipeline: `clip-like vision encoder → per-patch embeddings → MaxSim over patches for query-document score`. Enables document RAG without any text extraction.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase8 #ColPali #multimodal #retrieval

---

**ID**: P8-028
**Front**: What is Ollama and how does it simplify local LLM deployment?
**Back**: Ollama: CLI tool for running LLMs locally. One command: `ollama run llama3` — downloads, quantizes (GGUF), runs. REST API: `curl http://localhost:11434/api/chat`. OpenAI-compatible API: `client = OpenAI(base_url="http://localhost:11434/v1")`. Supports: LLaMA, Mistral, Phi, Gemma, custom Modelfiles. Use for: local dev/testing, privacy-sensitive use cases, offline demo. Limitations: single-user, no continuous batching → not for production serving.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase8 #ollama #local #inference

---

**ID**: P8-029
**Front**: What is benchmarking an LLM inference server? What should you measure?
**Back**: Key metrics: (1) TTFT (time to first token) — measures streaming responsiveness. (2) TPOT (time per output token) — generation speed. (3) Throughput (tokens/s) at different batch sizes. (4) P50/P95/P99 latency. (5) GPU utilization %. (6) Max concurrent users before degradation. Tools: Locust (load test), `vllm benchmark_throughput.py`. Typical targets: TTFT < 500ms, TPOT < 30ms, P95 latency < 2s. Test under: single user, 10, 100, 1000 concurrent requests.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase8 #benchmarking #inference #latency

---

**ID**: P8-030
**Front**: What are the main differences between Mistral-7B and LLaMA-3-8B architecturally?
**Back**: Mistral-7B: Sliding Window Attention (4096 window), Grouped Query Attention (8 KV heads), SwiGLU FFN, rotary embeddings, 32K context. LLaMA-3-8B: full attention (no sliding window), GQA (8 KV heads), SwiGLU FFN, RoPE, 8K context (128K with LLaMA-3.1). Both use GQA, RoPE, SwiGLU — modern architecture standard. Mistral: longer context with less memory (sliding window). LLaMA-3: better quality (more training data), extended context with YaRN in 3.1.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase8 #Mistral #LLaMA #architecture
