# Phase 4 Flashcards — NLP & Transformers

[← Flashcards Index](./README.md) | [Phase 4 Files](../05_Phase4_Part1_NLP_Fundamentals.md) | [Cheat Sheet](../CheatSheets/Phase4_NLP_Transformers.md)

---

**ID**: P4-001
**Front**: Write the scaled dot-product attention formula and explain each component.
**Back**: Attention(Q,K,V) = softmax(QK^T / √d_k) V. Q (query): "what am I looking for?" K (key): "what do I contain?" V (value): "what information to pass?" QK^T: similarity scores — how much each query attends to each key. √d_k scaling: prevents softmax saturation in high dimensions. Softmax → attention weights (sum to 1). Weighted sum of V = output.
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase4 #attention #transformer #QKV

---

**ID**: P4-002
**Front**: What is multi-head attention and why is it better than single-head attention?
**Back**: Multi-head: split d_model into h heads, run attention in parallel, concatenate and project. `head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`. Output = concat(head_1,...,head_h)W^O. Why better: each head can attend to different positions and representation subspaces simultaneously. Example: one head tracks syntax, another tracks coreference, another tracks semantics.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #attention #multi-head

---

**ID**: P4-003
**Front**: What is GQA (Grouped-Query Attention)? How does LLaMA-3-8B use it?
**Back**: GQA: multiple query heads share a smaller number of key-value heads. LLaMA-3-8B: n_heads=32, n_kv_heads=8 → 4 query heads share 1 KV head. Benefit: 4× smaller KV cache at inference → 4× more throughput. Quality nearly identical to full MHA. Implementation: K,V projected to n_kv_heads dim, then repeat_interleave to n_heads before attention. Between MHA (n_kv=n_q) and MQA (n_kv=1).
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase4 #GQA #MQA #attention #LLaMA

---

**ID**: P4-004
**Front**: What is the KV cache and how does it accelerate generation?
**Back**: During autoregressive generation: for each new token, we'd recompute K,V for ALL previous tokens. KV cache: store K,V for each token after computing them. On next step: only compute K,V for the NEW token; concatenate with cached K,V. Cost: 1 forward pass per token instead of growing cost. Memory: grows linearly with sequence length. GQA dramatically reduces this memory.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #KV-cache #inference #generation

---

**ID**: P4-005
**Front**: What is RoPE (Rotary Position Embedding) and why do modern LLMs use it?
**Back**: RoPE: encodes position by rotating Q,K vectors in complex space. Position θ_m rotates the m-th token's embedding. Properties: (1) Relative position is naturally captured — dot product between positions m,n depends only on m-n. (2) Can extrapolate beyond training length (via YaRN, LongRoPE). (3) No learnable params. Used by: LLaMA, Mistral, GPT-NeoX, Qwen. Replaces absolute/sinusoidal position embeddings.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase4 #RoPE #positional-encoding

---

**ID**: P4-006
**Front**: What is the causal mask in transformer decoder? Why is it needed?
**Back**: Causal (autoregressive) mask: upper triangular matrix of -inf, set before softmax. Ensures token i can only attend to positions 0...i (past), not i+1...T (future). Without it: during training, the model sees future tokens → data leakage → model can't generalize to actual generation. At inference: masking is implicit (only have past tokens). Pre-norm + causal mask = GPT architecture.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #causal-mask #decoder #autoregressive

---

**ID**: P4-007
**Front**: What is the difference between encoder-only, decoder-only, and encoder-decoder transformers?
**Back**: Encoder-only (BERT): bidirectional (sees all tokens), good for classification/embedding/NER. Decoder-only (GPT): causal (sees only past), good for generation. Encoder-decoder (T5, BART): encoder processes input bidirectionally; decoder attends to encoder output + causal past — good for seq2seq (translation, summarization). Current trend: decoder-only LLMs dominate (GPT-4, LLaMA, Mistral).
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #transformer #encoder #decoder

---

**ID**: P4-008
**Front**: What is Byte Pair Encoding (BPE) tokenization?
**Back**: BPE: start with character vocabulary. Iteratively merge the most frequent adjacent byte pair → form new token. Repeat until vocab_size reached. Result: common words become single tokens; rare words split into subwords. Example: "transformer" → ["transform", "er"]. Handles OOV words gracefully. GPT-2/3/4, LLaMA all use BPE variants. Vocab size: 32K–128K tokens.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #tokenization #BPE

---

**ID**: P4-009
**Front**: What is Word2Vec and what are its two training objectives?
**Back**: Word2Vec: learns dense word embeddings from text. (1) CBOW (Continuous Bag of Words): predict center word from context. (2) Skip-gram: predict context words from center word. Training: move similar words together in embedding space, dissimilar apart. Famous property: king - man + woman ≈ queen (arithmetic in embedding space). Replaced by contextual embeddings (BERT) but foundational concept.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #word2vec #embeddings #NLP

---

**ID**: P4-010
**Front**: What is BERT and how does it differ from GPT?
**Back**: BERT (Bidirectional Encoder Representations from Transformers): encoder-only, sees all tokens (bidirectional attention), pretrained on MLM (masked language modeling) + NSP. GPT: decoder-only, causal (left-to-right only), pretrained on next-token prediction. BERT better for: classification, NER, semantic similarity (full context). GPT better for: generation, completion. GPT-4, LLaMA = GPT-style. Sentence-transformers = BERT-style.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #BERT #GPT #transformer

---

**ID**: P4-011
**Front**: What is masked language modeling (MLM) in BERT?
**Back**: Pretrain BERT by randomly masking 15% of tokens and predicting the masked tokens. Forces model to learn bidirectional context (must use both left and right context to predict masked token). 80% replaced with [MASK], 10% with random word, 10% unchanged (prevents model from ignoring non-masked tokens). Contrast with GPT: predicts NEXT token (unidirectional). MLM → better representations; CLM → generation ability.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #BERT #MLM #pretraining

---

**ID**: P4-012
**Front**: What is the Feed-Forward Network (FFN) in a transformer block?
**Back**: Position-wise FFN: applied to each position independently. Two linear layers with activation: FFN(x) = Linear(GELU(Linear(x))). Hidden dim: typically 4× d_model (e.g., d_model=768 → FFN_hidden=3072). LLaMA uses SwiGLU: FFN(x) = Linear(SiLU(Linear(x)) ⊙ Linear(x)) with ²⁄₃ × 4d hidden dim. FFN provides the "memory" component — most LLM knowledge is stored here. Transformer = attention (routing) + FFN (knowledge).
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #transformer #FFN #feed-forward

---

**ID**: P4-013
**Front**: What is pre-norm vs post-norm in transformers? Which do modern LLMs use?
**Back**: Post-norm (original): LayerNorm AFTER attention/FFN + residual. Pre-norm (modern): LayerNorm BEFORE attention/FFN. x → LN → Attention → + x. Benefits of pre-norm: more stable training, less sensitive to learning rate, better gradient flow. Modern LLMs (GPT-3, LLaMA, Mistral) all use pre-norm. Original "Attention is All You Need" used post-norm.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #transformer #layer-norm #pre-norm

---

**ID**: P4-014
**Front**: Why does attention scale as O(n²) and what techniques address this?
**Back**: O(n²): QK^T matrix is (n×n) — every pair of tokens interacts. Memory + compute both O(n²). For n=100K tokens: 10B attention weight pairs → infeasible. Solutions: (1) Flash Attention: reorders compute to minimize HBM reads (same O(n²) but memory-efficient). (2) Sliding window attention (Mistral): attend only to local window. (3) Linear attention (Mamba/SSMs): O(n) approx. (4) GQA: reduces KV memory by n_heads/n_kv_heads.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase4 #attention #complexity #flash-attention

---

**ID**: P4-015
**Front**: What is Flash Attention and what problem does it solve?
**Back**: Flash Attention (Dao et al. 2022): IO-aware attention algorithm. Problem: standard attention materializes full (n×n) attention matrix in GPU HBM (slow memory) → memory bandwidth bottleneck. Flash Attention: tiles computation to fit in SRAM (fast memory), never materializes full matrix. Result: same mathematical output, 3-8× faster, O(n) memory instead of O(n²). Standard in all major LLM frameworks.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase4 #flash-attention #GPU #efficiency

---

**ID**: P4-016
**Front**: What is the transformer's complexity in time and space? For both training and inference?
**Back**: Attention: time O(n²d), space O(n²+nd) per layer. FFN: time O(nd²), space O(d²). Training total: O(n²d + nd²) per layer. Memory: O(n²) for attention weights (Flash Attention reduces to O(n)). Inference: O(n) per new token with KV cache (past K,V cached). Without KV cache: O(n²) per new token. This is why KV cache is essential for serving.
**Difficulty**: Advanced
**Category**: Math
**Tags**: #phase4 #transformer #complexity

---

**ID**: P4-017
**Front**: What is perplexity and how does it evaluate language models?
**Back**: Perplexity = 2^H(P,Q) where H = cross-entropy of model on test text. Lower = better. Intuition: average branching factor — how many equally likely choices the model sees at each step. GPT-2: ~29 perplexity on WikiText-103. GPT-3: ~20. GPT-4: ~8–10. Limitation: doesn't measure factual accuracy or instruction following — why we also use MMLU, HumanEval, etc.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #perplexity #evaluation #language-model

---

**ID**: P4-018
**Front**: What is sentence-transformers and when do you use it vs OpenAI embeddings?
**Back**: Sentence-transformers: fine-tuned BERT-like models optimized for semantic similarity (trained on NLI, STS datasets). Open source, free, runs locally. `SentenceTransformer("all-MiniLM-L6-v2").encode(texts)` → 384-dim vectors. OpenAI text-embedding-3-small: 1536-dim, higher quality, costs $0.02/1M tokens. Use sentence-transformers when: cost-sensitive, offline, lower latency needed. Use OpenAI for: best quality, production RAG.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase4 #embeddings #sentence-transformers

---

**ID**: P4-019
**Front**: What are the key components of the GPT architecture in order?
**Back**: (1) Token embeddings: token_id → d_model vector. (2) Positional encoding (absolute or RoPE). (3) N × Transformer blocks: (a) Pre-LayerNorm. (b) Causal Multi-Head (or GQ) Attention + residual. (c) Pre-LayerNorm. (d) FFN (Linear-GELU-Linear or SwiGLU) + residual. (4) Final LayerNorm. (5) LM head: linear (d_model → vocab_size). Softmax → probability over next token.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #GPT #transformer #architecture

---

**ID**: P4-020
**Front**: What is the difference between padding tokens and attention masking?
**Back**: Padding: add [PAD] tokens to make all sequences same length in a batch. Attention mask: binary mask (1 = attend, 0 = ignore) passed to model to prevent attending to padding positions. In transformers: set attention scores for padding positions to -inf before softmax. Without masking: padding tokens influence attention outputs — hurts performance. HuggingFace: tokenizer returns `attention_mask` automatically.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase4 #transformer #padding #attention-mask

---

**ID**: P4-021
**Front**: What is named entity recognition (NER) and what model type handles it?
**Back**: NER: identify and classify entities in text (PERSON, ORG, LOC, DATE, etc.). Output: per-token labels. Model: encoder-only (BERT-style) with token classification head. Fine-tune: add linear layer on top of BERT token embeddings → classify each token. Evaluation: entity-level F1 (not token-level — partial matches don't count). Common in: information extraction, document understanding, knowledge graph construction.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #NER #BERT #NLP

---

**ID**: P4-022
**Front**: What is TF-IDF and when is it still useful despite being "old"?
**Back**: TF-IDF = Term Frequency × Inverse Document Frequency. TF: how often term appears in document. IDF: log(N / documents containing term) — rare terms get high weight. Still useful: (1) Keyword search / BM25 (used in Elasticsearch). (2) Hybrid search with semantic embeddings. (3) Feature extraction for traditional ML on text. (4) Fast, interpretable, no GPU needed. Replaced by embeddings for semantic tasks.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase4 #TF-IDF #NLP #search

---

**ID**: P4-023
**Front**: What is the difference between `input_ids`, `attention_mask`, and `token_type_ids` in HuggingFace?
**Back**: `input_ids`: integer token IDs (main input). `attention_mask`: 1 for real tokens, 0 for padding — prevents attention to padding. `token_type_ids`: distinguishes segment A vs segment B in BERT (sentence pair tasks). Optional for most modern models. `position_ids`: explicit position indices (usually computed internally). Always use tokenizer: `tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)`.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase4 #huggingface #tokenizer

---

**ID**: P4-024
**Front**: Why does adding special tokens like [CLS] and [SEP] matter in BERT?
**Back**: [CLS] (classification): always first token. Its final hidden state aggregates whole-sentence representation — used as input to classification head. [SEP] (separator): marks sentence boundaries in sentence-pair tasks (next sentence prediction). Pooling alternatives: mean pooling over all tokens often better than [CLS] for sentence embeddings. `sentence-transformers` uses mean pooling by default.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #BERT #special-tokens

---

**ID**: P4-025
**Front**: What is the scaling law relationship between model size, training compute, and loss?
**Back**: Chinchilla scaling law (Hoffmann et al. 2022): optimal compute split ≈ equal number of parameters and training tokens. For N parameters: train on ~20N tokens. Example: LLaMA-3-8B trained on 15T tokens (way over Chinchilla-optimal — deliberately overtrained for inference efficiency). Key insight: you can trade model size for training data to get same loss with smaller (cheaper) model.
**Difficulty**: Advanced
**Category**: Research
**Tags**: #phase4 #scaling-laws #LLM #chinchilla

---

**ID**: P4-026
**Front**: What is Cross-Attention and where is it used?
**Back**: Cross-attention: Q from decoder, K,V from encoder. Allows decoder to attend to encoder output at each generation step. Used in: encoder-decoder models (T5, BART, Whisper). Not in decoder-only (GPT, LLaMA). Multi-modal models use cross-attention: visual tokens as K,V, text tokens as Q → allows text decoder to "see" image. Same math as self-attention, different Q/K/V sources.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #cross-attention #encoder-decoder

---

**ID**: P4-027
**Front**: What is the temperature parameter in generation and how does top-p (nucleus) sampling differ from top-k?
**Back**: Temperature T: scale logits → softmax(z/T). Low T: sharper, more deterministic. High T: more random. Top-k: keep only k highest probability tokens, sample from them. Top-p (nucleus): keep smallest set of tokens whose cumulative probability ≥ p (dynamic k). Top-p preferred: adapts to output distribution naturally. Production: temperature=0.0-0.2 for factual, 0.7-1.0 for creative; top_p=0.9 common default.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase4 #generation #sampling #temperature

---

**ID**: P4-028
**Front**: What is greedy decoding vs beam search for text generation?
**Back**: Greedy: argmax at each step — fastest, often suboptimal (misses globally better sequences). Beam search: keep top-k sequences at each step, expand all, keep top-k again. k = beam width. Finds better sequences than greedy; k× more compute. Modern preference: sampling with temperature/top-p (more diverse) over beam search (tends to be generic/repetitive). Beam search still used for: translation, code generation (BLEU optimization).
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #generation #beam-search #decoding

---

**ID**: P4-029
**Front**: What is the transformer's attention complexity limitation for very long sequences?
**Back**: O(n²) in both time and memory — 128K context = 16B attention weight pairs. Approaches: (1) Flash Attention: same O(n²) but cache-efficient. (2) Sliding window (Mistral): attend to 4096 local + some global tokens. (3) Linear attention (SSM/Mamba): O(n) but approximation. (4) KV cache compression (H2O): keep only important KV pairs. Context of 1M+ (Gemini 1.5) uses custom architecture + engineering.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase4 #long-context #attention #complexity

---

**ID**: P4-030
**Front**: What is BLEU score and what are its limitations?
**Back**: BLEU (Bilingual Evaluation Understudy): n-gram overlap between generated text and reference. Range 0–1 (or 0–100). Widely used for machine translation. Limitations: (1) Doesn't capture semantics (synonym = wrong). (2) Recall bias (shorter outputs penalized). (3) Doesn't correlate with human judgment for free-form generation. Alternatives: ROUGE (summarization), BERTScore (semantic), human evaluation, GPT-4 as judge.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #BLEU #evaluation #NLP

---

**ID**: P4-031
**Front**: What is the role of the linear projection (LM head) at the top of a language model?
**Back**: LM head: `nn.Linear(d_model, vocab_size, bias=False)`. Projects from model dimension to vocabulary size. Output logits: (batch, seq_len, vocab_size). Apply softmax → probability distribution over all possible next tokens. In practice: use log_softmax + NLLLoss or cross_entropy_loss (more numerically stable). During generation: take logits at position -1 (last token) → sample → append → repeat.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #transformer #LM-head #generation

---

**ID**: P4-032
**Front**: What is positional encoding and why can't attention learn positions automatically?
**Back**: Attention is permutation-invariant: Attention(permuted input) = permuted Attention(input). Without positional info, "cat sat mat" and "mat sat cat" produce identical representations. Positional encoding injects order. Types: (1) Sinusoidal (fixed, original paper). (2) Learned absolute (BERT, GPT-2). (3) RoPE (rotation-based, relative, LLaMA). (4) ALiBi (attention bias, Bloom). RoPE dominates modern LLMs.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #positional-encoding #transformer

---

**ID**: P4-033
**Front**: How does HuggingFace's `pipeline()` abstract model loading and inference?
**Back**: `pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", device=0)`. Handles: tokenization, model loading, forward pass, decoding. Quick prototyping but less control than manual. For production: use `AutoModelForCausalLM.from_pretrained()` + `AutoTokenizer` directly. Pipeline overhead is significant — don't use in tight inference loops. Supports: text-generation, text-classification, NER, QA, summarization, translation.
**Difficulty**: Beginner
**Category**: Coding
**Tags**: #phase4 #huggingface #pipeline

---

**ID**: P4-034
**Front**: What is weight tying between the embedding layer and LM head?
**Back**: Language models: reuse the same weight matrix for both token embedding (input) and LM head (output projection). `model.lm_head.weight = model.model.embed_tokens.weight`. Benefits: (1) Reduces parameters by ~vocab_size × d_model (~125M for 32K vocab, 4096 dim). (2) Consistency: tokens with similar input embeddings get similar output scores. Used by: GPT-2, LLaMA, and most LLMs. Related to noise-contrastive estimation theory.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase4 #weight-tying #transformer #embedding

---

**ID**: P4-035
**Front**: What is semantic search and how does it differ from keyword search?
**Back**: Keyword search (BM25/TF-IDF): matches exact words — fast, interpretable, misses synonyms. Semantic search (dense retrieval): embed query + documents → find nearest neighbors by cosine similarity. Captures meaning ("automobile" ≈ "car"). Dense retrieval slower to compute but vector index (FAISS, Qdrant) makes search fast. Best: hybrid search = BM25 + semantic + reranker (cross-encoder) for final ranking.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #semantic-search #embeddings #retrieval

---

**ID**: P4-036
**Front**: What is the output of a transformer block for a single token? What shape is it?
**Back**: Input: (d_model,) vector for one token. After attention: (d_model,) — weighted sum of value vectors from all positions, projected. After FFN: (d_model,) — nonlinear transformation per position. Both have residual connections. Shape throughout: (batch, seq_len, d_model) — each position processed independently in FFN; positions interact in attention. Final output is richer representation of each token in its full context.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #transformer #shapes

---

**ID**: P4-037
**Front**: What is repetition penalty in language model generation?
**Back**: Penalty applied to logits of tokens already generated — reduces their probability to discourage repeating the same words/phrases. `repetition_penalty=1.2` is common. Implementation: divide logits of already-seen tokens by penalty factor before sampling. Higher = stronger repetition suppression. Alternative: n-gram blocking (don't generate same n-gram twice). Problem without it: LLMs can get stuck in loops repeating the same phrases.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase4 #generation #repetition-penalty

---

**ID**: P4-038
**Front**: Name the key NLP benchmarks for evaluating LLMs.
**Back**: MMLU: 57-subject multiple choice (knowledge breadth). HumanEval: Python coding (functional correctness). GSM8K: grade school math reasoning. MBPP: Python programming problems. BIG-bench: 214 tasks (diverse reasoning). HELM: standardized multi-task evaluation. MT-Bench: 2-turn multi-topic conversation (GPT-4 judge). LMSYS Chatbot Arena: human preference ranking. TruthfulQA: truthfulness. HellaSwag: commonsense.
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase4 #benchmarks #evaluation #LLM

---

**ID**: P4-039
**Front**: What is the sliding window attention in Mistral-7B and why was it introduced?
**Back**: Standard attention: O(n²) for full context. Mistral sliding window: each token attends to at most W=4096 past tokens (not all n). Rolling buffer KV cache: KV cache size fixed at W instead of growing. With n layers, effective receptive field = W × n_layers = 4096 × 32 = 131K (much larger than window). Key win: O(n·W) memory instead of O(n²). Trade-off: may miss very long-range dependencies.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase4 #sliding-window #Mistral #long-context

---

**ID**: P4-040
**Front**: What is contrastive learning in NLP and how do sentence embeddings use it?
**Back**: Contrastive learning: train embeddings so similar pairs are close, dissimilar pairs are far. Loss: InfoNCE or contrastive loss. For sentences: SimCSE (positive = same sentence with different dropout noise). sentence-transformers: fine-tuned on NLI (entailment = positive, contradiction = negative). Result: BERT embeddings optimized for cosine similarity — much better for semantic search than plain BERT.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase4 #contrastive-learning #embeddings #sentence-transformers

---

**ID**: P4-041
**Front**: What does tokenization do to numbers and what problem can this cause?
**Back**: Numbers are often split into subword tokens: "1234567" → ["1234", "567"] or ["1", "234", "567"]. Problem: models can't naturally do arithmetic because numbers aren't represented in a semantically meaningful way — "123" and "124" share no tokens. Why LLMs struggle with math: positional/carry operations on token-level representations is very hard. Solution: calculator tools, code execution (let Python handle math).
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase4 #tokenization #numbers #arithmetic

---

**ID**: P4-042
**Front**: What is a cross-encoder reranker and when do you use it in RAG?
**Back**: Cross-encoder: takes (query, document) pair jointly — richer interaction than bi-encoder. Much slower (can't precompute) but more accurate. Pipeline: (1) Bi-encoder fast retrieval → top 50–100 candidates. (2) Cross-encoder rerank → top 5. Cross-encoders: BGE-reranker, ms-marco-MiniLM. Use when: precision is critical, retrieval set is manageable. Bi-encoder only is faster; cross-encoder reranking is more accurate.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase4 #reranker #cross-encoder #RAG

---

**ID**: P4-043
**Front**: How does the HuggingFace `generate()` method work? Name 5 key parameters.
**Back**: `generate()` calls the model autoregressively until max_new_tokens or EOS. Key params: (1) `max_new_tokens`: generation budget. (2) `temperature`: randomness. (3) `top_p` / `top_k`: nucleus/top-k sampling. (4) `do_sample=True`: enable sampling (else greedy). (5) `repetition_penalty`. Also: `eos_token_id`, `pad_token_id`, `num_beams` (beam search), `num_return_sequences`. For chat models: use `apply_chat_template()` to format correctly before generate.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase4 #huggingface #generate

---

**ID**: P4-044
**Front**: What is zero-shot vs few-shot learning in LLMs?
**Back**: Zero-shot: task described in prompt only, no examples. Model must generalize from pretraining. Few-shot: include k examples (demonstrations) in prompt before the actual query. Typically 3–5 examples sufficient. Why works: in-context learning (Garg et al. 2022) — LLMs learn the task format from examples in context without weight updates. Key finding: few-shot narrows the gap with fine-tuned models on many tasks.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase4 #zero-shot #few-shot #in-context-learning

---

**ID**: P4-045
**Front**: What is Tiktoken and how does it differ from HuggingFace tokenizers?
**Back**: Tiktoken: OpenAI's BPE tokenizer. Fast (Rust core), used for GPT-2/3/4, cl100k_base (GPT-4: 100K vocab). `tiktoken.encoding_for_model("gpt-4")`. HuggingFace tokenizers: supports many models, slower for large vocab, more flexible. Key difference: Tiktoken handles byte fallback (any Unicode), GPT-4's cl100k encodes spaces within tokens. Use tiktoken to: count tokens before API calls to avoid 4096/8192/128K context limit errors.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase4 #tiktoken #tokenizer #openai
