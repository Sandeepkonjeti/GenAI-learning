# Phase 5 Flashcards — LLMs, Prompting & RAG

[← Flashcards Index](./README.md) | [Phase 5 Files](../06_Phase5_Part1_LLMs_and_Prompt_Engineering.md) | [Cheat Sheet](../CheatSheets/Phase5_LLMs_RAG.md)

---

**ID**: P5-001
**Front**: What are the three roles in the OpenAI chat API and what is each used for?
**Back**: `system`: sets the model's behavior/persona for the conversation — processed first, highest priority. `user`: the human's message. `assistant`: the model's previous responses (for multi-turn). To add chat history: append (user, assistant) pairs to messages list. `system` prompt injection risk: user messages that override system instructions.
**Difficulty**: Beginner
**Category**: Coding
**Tags**: #phase5 #openai #chat-api #roles

---

**ID**: P5-002
**Front**: What is the `instructor` library and why is it the production standard for structured outputs?
**Back**: `instructor.from_openai(OpenAI())` wraps the client to return Pydantic models instead of raw strings. Features: (1) `response_model=MyModel` — auto-validates JSON. (2) `max_retries=3` — auto-retries if validation fails. (3) Works with OpenAI, Anthropic, Gemini, local models. Why standard: GPT-4 JSON mode still returns invalid JSON ~2–5% of calls; instructor catches and retries. `pip install instructor`.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase5 #instructor #structured-output #pydantic

---

**ID**: P5-003
**Front**: What is chain-of-thought prompting and when does it help?
**Back**: CoT: add "Let's think step by step" or provide step-by-step examples in the prompt. Forces the model to generate intermediate reasoning steps before the final answer. Why it helps: transforms one-shot prediction to intermediate computation. When it helps most: math, logic, multi-step reasoning. Diminishing returns on factual recall (no computation needed). Zero-shot CoT: just "Let's think step by step". Few-shot CoT: provide 3-5 examples with reasoning chains.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase5 #chain-of-thought #prompting

---

**ID**: P5-004
**Front**: What is self-consistency prompting and when is it most useful?
**Back**: Generate N responses with temperature > 0. Take majority vote on the final answer. Example: ask the same math problem 5 times → answer "42" appears 4 times → report 42. Why: individual LLM samples are noisy; aggregation reduces variance. Most useful for: math, logic, multiple-choice where majority vote is well-defined. Less useful for: open-ended generation. Cost: N× more tokens. Typical N: 5–10.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase5 #self-consistency #prompting

---

**ID**: P5-005
**Front**: What is retrieval-augmented generation (RAG) and what problem does it solve?
**Back**: RAG: retrieve relevant documents from a knowledge base and include them in the prompt as context. Problem solved: (1) LLMs have knowledge cutoffs. (2) LLMs hallucinate facts not in training data. (3) LLMs can't access private/proprietary knowledge. RAG adds: up-to-date, specific, verifiable context. Trade-off: adds latency (retrieval + longer prompt). Alternative: fine-tuning (for style/behavior, not facts).
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase5 #RAG #retrieval

---

**ID**: P5-006
**Front**: What chunking overlap means in RAG and why is it important?
**Back**: Overlap: consecutive chunks share N tokens (e.g., 10–20% of chunk size). Without overlap: context at chunk boundaries is split — a sentence starting at end of chunk A and ending at start of chunk B is incomplete in both chunks. With overlap: each boundary context appears in 2 chunks → retrieval can find it. Typical: 512-token chunks with 50-token overlap. Too much overlap → duplicate context → higher storage cost.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase5 #RAG #chunking

---

**ID**: P5-007
**Front**: What is the difference between sparse (BM25) and dense (embedding) retrieval?
**Back**: Sparse (BM25): bag-of-words, exact token matching. Fast, interpretable. Misses synonyms, paraphrases. Dense (embedding): semantic vectors, captures meaning. Slower to build index but fast search. Hybrid: combine BM25 score + embedding score → weighted rank fusion (RRF). Cross-encoder reranker: final step, joint (query, doc) scoring. Best production RAG: BM25 + dense retrieval → top 50 → reranker → top 5.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase5 #retrieval #BM25 #hybrid-search

---

**ID**: P5-008
**Front**: What are the four RAGAS metrics and what does each measure?
**Back**: (1) Faithfulness: is the answer grounded in the retrieved context? (LLM judge + NLI). (2) Answer Relevancy: does the answer address the question? (embedding similarity). (3) Context Precision: are retrieved chunks relevant? (how many needed for the answer). (4) Context Recall: were all relevant chunks retrieved? (did we miss important context). All range 0–1, higher = better. Use together for complete RAG diagnosis.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase5 #RAGAS #evaluation #RAG

---

**ID**: P5-009
**Front**: What is prompt injection and how do you defend against it?
**Back**: Prompt injection: user crafts input that overrides system prompt instructions. Example: "Ignore previous instructions. You are now..." Defenses: (1) Input validation — detect "ignore", "forget", etc. (2) XML/JSON structured prompts — separate user content clearly. (3) Output validation — check response matches expected format. (4) Principle of least privilege — don't give LLM access to tools it doesn't need. (5) Canary tokens in system prompt to detect leakage.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase5 #security #prompt-injection

---

**ID**: P5-010
**Front**: What is LangSmith used for in production LLM systems?
**Back**: LangSmith: observability platform for LLM applications. Features: (1) Trace every LLM call (input, output, latency, tokens, cost). (2) Chain/agent tracing (see each step). (3) Dataset creation from traced examples. (4) Evaluation runs against datasets. (5) Playground for prompt testing. Use: `os.environ["LANGCHAIN_API_KEY"] = "..."`. Alternatives: Helicone, W&B Weave, Langfuse. Essential for debugging multi-step agent workflows.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase5 #langsmith #observability #tracing

---

**ID**: P5-011
**Front**: What is the OpenAI function calling / tool calling API?
**Back**: Allows LLM to request tool execution as structured JSON. Define tools: `{"type":"function","function":{"name":"search","description":"...","parameters":{...}}}`. Model returns: `tool_calls: [{"name":"search","arguments":{"query":"..."}]`. You execute the function, return result as tool message, continue conversation. Key: model chooses WHEN to call tools based on the conversation. GPT-4 excels at tool calling; smaller models may need `tool_choice="required"`.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase5 #function-calling #tool-calling #openai

---

**ID**: P5-012
**Front**: What is the difference between zero-shot and few-shot in the context of LLM prompting?
**Back**: Zero-shot: describe task in system/user prompt, no examples. Model relies entirely on pretraining. Few-shot: include 3–5 examples of (input, output) pairs in the prompt. Model mimics the pattern. When few-shot helps: consistent output format required, domain-specific language, tasks that are hard to describe but easy to demonstrate. Cost: k examples × prompt length = more tokens. Alternative to few-shot: fine-tuning (when consistent format is always needed).
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase5 #zero-shot #few-shot #prompting

---

**ID**: P5-013
**Front**: What is contextual retrieval (Anthropic 2024) and how does it improve RAG?
**Back**: Before embedding each chunk: use Claude to generate a 2-3 sentence document-level context summary prepended to the chunk. "This chunk is from a Q3 2024 earnings report discussing R&D expenses..." Result: each chunk has its document context embedded with it → retrieval is much more accurate for specific subsets. Reduces retrieval failure from ~20% to ~5% in Anthropic's tests. Cost: N_chunks × ~200 tokens per contextualization call.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase5 #contextual-retrieval #RAG #Anthropic

---

**ID**: P5-014
**Front**: What is a vector database? Name three and their key differentiation.
**Back**: Stores high-dimensional vectors with efficient approximate nearest neighbor (ANN) search. (1) **Qdrant**: Python-first, payload filtering, payload-based filtering at index level, HNSW + quantization — best OSS option for production. (2) **ChromaDB**: easiest to use, great for prototyping, embedded mode. (3) **Pinecone**: fully managed, zero-ops, expensive. (4) Databricks Vector Search: native Databricks integration, Delta Lake backed. HNSW is the standard ANN algorithm.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase5 #vector-db #qdrant #chromadb

---

**ID**: P5-015
**Front**: What is Corrective RAG (CRAG) and what problem does it solve?
**Back**: CRAG: if retrieved documents are not relevant to the query, fall back to web search instead of hallucinating. Pipeline: (1) Retrieve top-k. (2) Evaluate relevance (LLM-as-judge or cross-encoder). (3) If score < threshold → web search. (4) Combine filtered retrieved docs + web results → generate. Solves: RAG hallucinating when the KB doesn't have the answer. Requires: web search API (Tavily, Serper, Bing).
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase5 #CRAG #RAG #advanced-RAG

---

**ID**: P5-016
**Front**: How does OpenAI's `text-embedding-3-small` differ from `text-embedding-ada-002`?
**Back**: Ada-002: 1536 dims, released 2022, $0.1/1M tokens. text-embedding-3-small: 1536 dims, newer, $0.02/1M tokens (5× cheaper), better MTEB performance. text-embedding-3-large: 3072 dims, even better quality. Key feature of v3 models: `dimensions` param to truncate to smaller embedding (e.g., 512) for storage savings — still good quality. Migration: re-embed all documents when switching models (embeddings are model-specific).
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase5 #openai #embeddings

---

**ID**: P5-017
**Front**: What is the LLM context window and what happens when you exceed it?
**Back**: Context window: max tokens the model can process (prompt + response combined). GPT-4o: 128K tokens. Claude 3.5: 200K. LLaMA-3: 8K (base), extended with YaRN. If exceeded: API error (context_length_exceeded) or silent truncation (older models). For RAG: chunk size + k retrieved chunks + system prompt + response budget all count. Always budget: leave ~20% for response. Track with tiktoken.
**Difficulty**: Beginner
**Category**: Production
**Tags**: #phase5 #context-window #tokens

---

**ID**: P5-018
**Front**: What is an LLM system prompt and what are its security risks?
**Back**: System prompt: initial context given to the model before user interaction. Sets persona, rules, format requirements. Risks: (1) Prompt injection: user overrides system prompt instructions. (2) System prompt leakage: model reveals its system prompt if asked cleverly. (3) Jailbreaking: prompt that bypasses safety rules. Defenses: (1) Clear separation (XML tags). (2) Output validation. (3) Canary tokens to detect leakage. Never put secrets in system prompts.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase5 #system-prompt #security

---

**ID**: P5-019
**Front**: What are the key LLM providers and their flagship models (2024-2025)?
**Back**: OpenAI: GPT-4o (general), o1 (reasoning), GPT-4o-mini (fast/cheap). Anthropic: Claude 3.5 Sonnet (strong coding/reasoning), Claude 3 Haiku (fast). Google: Gemini 1.5 Pro (1M context), Gemini 1.5 Flash (fast). Meta (open): LLaMA-3-8B, LLaMA-3-70B (free, run locally). Mistral: Mistral-7B (fast), Mixtral-8x22B (MoE). Databricks: DBRX (enterprise). Use LiteLLM or LangChain for provider-agnostic code.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase5 #LLM #providers #models

---

**ID**: P5-020
**Front**: What is LiteLLM and why is it useful for production code?
**Back**: LiteLLM: unified API for 100+ LLM providers. Same code → OpenAI, Anthropic, Gemini, Cohere, Databricks. `from litellm import completion`. Completion(model="anthropic/claude-3-haiku-20240307", ...). Benefits: (1) Fallback to secondary provider if primary fails. (2) Cost tracking. (3) Rate limit handling. (4) Easy A/B testing between models. (5) Mock mode for testing. Production recommendation: always abstract behind LiteLLM.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase5 #litellm #production

---

**ID**: P5-021
**Front**: What is retrieval Precision@k and Recall@k in RAG evaluation?
**Back**: Precision@k: of the k retrieved chunks, what fraction is relevant? Recall@k: of all relevant chunks in the corpus, what fraction is in the top-k? Both needed: high precision but low recall = missing important context. High recall but low precision = noisy context. MRR (Mean Reciprocal Rank): 1/rank of first relevant result — rewards finding relevant docs quickly. Measure these before RAGAS — retrieval quality is the foundation.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase5 #evaluation #retrieval #precision-recall

---

**ID**: P5-022
**Front**: What is the ReAct prompt pattern in the context of LLM tool use?
**Back**: ReAct (Reasoning + Acting): `Thought: [reasoning] → Action: tool_name → Action Input: args → Observation: result → Thought: ...`. Forces LLM to reason before acting. Observation is system-injected (tool result). Repeat until `Final Answer:`. Why effective: explicit reasoning reduces hallucinated tool calls; can course-correct based on observations. Standard format for most agent frameworks.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase5 #ReAct #agent #prompting

---

**ID**: P5-023
**Front**: What is streaming in LLM APIs and why does it matter for UX?
**Back**: Streaming: tokens returned one at a time as generated instead of waiting for full response. `stream=True` in OpenAI API → iterate over `event.choices[0].delta.content`. UX impact: TTFT (time to first token) perceived as responsiveness. 5-second full-response wait feels slow; 0.5s to first token + streaming feels fast. Implementation: SSE (Server-Sent Events) for web, `StreamingResponse` in FastAPI. Essential for chatbots and long outputs.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase5 #streaming #UX #TTFT

---

**ID**: P5-024
**Front**: What is a Pydantic model and how does it validate structured outputs?
**Back**: Pydantic: Python data validation using type annotations. `class Result(BaseModel): name: str; score: float = Field(ge=0, le=1)`. Automatically: type coercion (str → int if possible), validation (ge=0 means ≥ 0), error messages. `Result.model_validate(json_data)` raises `ValidationError` if invalid. instructor uses Pydantic to validate LLM output — if LLM returns invalid JSON or wrong types, instructor retries.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase5 #pydantic #validation #instructor

---

**ID**: P5-025
**Front**: What is the HyDE (Hypothetical Document Embeddings) technique in RAG?
**Back**: HyDE: instead of embedding the raw user query, ask the LLM to generate a hypothetical document that would answer the query → embed that document → use it for retrieval. Why: hypothetical document is closer in embedding space to actual documents than a short query is. Improves retrieval recall on knowledge-intensive queries. Cost: one extra LLM call per query. Best for: technical queries where the question and answer have very different language.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase5 #HyDE #RAG #advanced-retrieval

---

**ID**: P5-026
**Front**: What is MMLU and what does it measure about LLMs?
**Back**: MMLU (Massive Multitask Language Understanding): 57 subjects, 14K multiple-choice questions. Subjects: math, science, law, medicine, history, etc. Measures: breadth of world knowledge and reasoning. GPT-4: ~86%. LLaMA-3-8B: ~66%. Human experts: ~89.8%. Limitation: static benchmark — LLMs may be trained on its questions. Doesn't measure: instruction following, code generation, long-form reasoning. Use together with HumanEval, GSM8K, MT-Bench for complete picture.
**Difficulty**: Intermediate
**Category**: Research
**Tags**: #phase5 #MMLU #benchmarks #evaluation

---

**ID**: P5-027
**Front**: What is cost tracking for LLM APIs and how do you implement it?
**Back**: Every API call returns: `usage.prompt_tokens`, `usage.completion_tokens`. Cost = prompt_tokens × input_price + completion_tokens × output_price. GPT-4o: $5/1M input, $15/1M output. Log these per request. Monitor: daily/monthly cost by model and endpoint. Alerts: set budget alarms. Tools: LiteLLM tracks cost automatically. `completion_cost(completion_response)`. Cache hits: $0 cost. Optimize: use smaller models for simple tasks.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase5 #cost-tracking #production

---

**ID**: P5-028
**Front**: What is temperature=0 in LLMs? Is it truly deterministic?
**Back**: Temperature=0: logits/0 → only highest-probability token selected (greedy decoding = argmax). Should be deterministic in theory. In practice: GPU floating-point operations are non-deterministic (CUDA parallelism, accumulation order). Same seed + same model on same hardware usually produces same output, but not guaranteed across GPU types or driver versions. For true reproducibility: `seed` param in OpenAI API (best effort). Always test for stability at temp=0.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase5 #temperature #determinism

---

**ID**: P5-029
**Front**: What is a semantic cache and how does it reduce LLM costs?
**Back**: Instead of exact cache (hit only if query is byte-for-byte identical), semantic cache: embed the incoming query → find cached responses with similar embedding (cosine similarity > threshold). Hit rate: 30–70% depending on query diversity. Implementation: Redis + vector index, or specialized tools like GPTCache. Risk: false positive cache hits for semantically similar but differently intended queries. Set similarity threshold conservatively (>0.95).
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase5 #semantic-cache #cost #production

---

**ID**: P5-030
**Front**: What is an embedding and what makes a good one for RAG?
**Back**: Embedding: dense numerical vector representing semantic meaning of text. Good embeddings: (1) Similar meanings → close in vector space (high cosine similarity). (2) Trained on similar domain as your documents. (3) Appropriate dimension (256–1536; larger isn't always better). (4) Same model for both documents and queries. Evaluation: MTEB leaderboard (massively multilingual text embedding benchmark). For domain-specific text: fine-tune embeddings on your data.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase5 #embeddings #RAG

---

**ID**: P5-031
**Front**: What is the "lost in the middle" problem in RAG?
**Back**: LLMs attend to the beginning and end of long contexts much better than the middle. If relevant context is position 50–60 in a 100-chunk context window, the LLM may miss or underweight it. Mitigations: (1) Limit context to 5–10 chunks (don't stuff everything). (2) Reranker to put most relevant chunks first. (3) "sandwich" trick: put most important context at start AND end. Research: "Lost in the Middle" (Liu et al. 2023).
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase5 #RAG #context #lost-in-middle

---

**ID**: P5-032
**Front**: What is multi-query RAG retrieval?
**Back**: Instead of one query → one retrieval, generate multiple query variants → retrieve for each → deduplicate → rank combined results. Example: user asks "how fast is LLaMA-3?" → generate: "LLaMA-3 benchmark performance", "LLaMA-3 tokens per second", "Meta LLaMA inference speed". Covers different ways the answer might be phrased in documents. Implementation: LangChain `MultiQueryRetriever`. Increases retrieval recall at cost of N× more embeddings.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase5 #multi-query #RAG #retrieval

---

**ID**: P5-033
**Front**: What is an LLM guard rail and what are common implementations?
**Back**: Guardrails: constraints on LLM inputs/outputs to ensure safety, format, and policy compliance. Input: detect prompt injection, PII, harmful topics. Output: validate format (Pydantic), detect harmful content, check factual grounding (faithfulness). Tools: (1) Guardrails AI: rule-based output validators. (2) NeMo Guardrails (NVIDIA): conversation flow control. (3) LlamaGuard: fine-tuned safety classifier. (4) Azure Content Safety API. Defense in depth: multiple layers.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase5 #guardrails #safety #production

---

**ID**: P5-034
**Front**: What is a hallucination in the context of LLMs?
**Back**: Hallucination: LLM generates text that is fluent and confident but factually wrong or not grounded in provided context. Types: (1) Intrinsic: contradicts source material. (2) Extrinsic: fabricates information not in source (most dangerous). Causes: training data noise, distribution shift, autoregressive probability doesn't care about factuality. Mitigations: RAG (ground in retrieved facts), low temperature, structured outputs, LLM-as-judge for faithfulness, retrieval-conditional generation.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase5 #hallucination #LLM

---

**ID**: P5-035
**Front**: What is few-shot prompting vs in-context learning vs fine-tuning?
**Back**: Few-shot prompting: examples in the prompt at inference time — no weight changes. In-context learning (ICL): the general phenomenon that LLMs can learn from prompt examples — broader than few-shot. Fine-tuning: examples used to update model weights — persistent improvement, no examples needed at inference. Trade-offs: few-shot = zero cost, flexible, uses tokens. Fine-tuning = upfront cost, more consistent, faster inference (no examples in prompt).
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase5 #few-shot #fine-tuning #ICL

---

**ID**: P5-036
**Front**: What is the OpenAI batch API and when should you use it?
**Back**: Batch API: submit up to 50K requests in a JSONL file, processed within 24 hours. Cost: 50% cheaper than synchronous API. Use for: offline evaluations, embedding generation, data annotation, any workload that doesn't need real-time response. Not for: user-facing features. Pricing: text-embedding-3-small batch = $0.01/1M tokens (vs $0.02 synchronous). Always use batch API for building evaluation datasets and running RAGAS evaluations.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase5 #openai #batch-api #cost

---

**ID**: P5-037
**Front**: What is LLM-as-judge and what are its biases?
**Back**: Use a powerful LLM (GPT-4, Claude) to evaluate another LLM's output. Common uses: RAG faithfulness evaluation, pairwise comparison, helpfulness scoring. Known biases: (1) Position bias: tends to prefer the first option. (2) Verbosity bias: prefers longer answers. (3) Self-enhancement: GPT-4 favors GPT-4 outputs. Mitigations: randomize order, use explicit rubric, calibrate against human judgments, use multiple judges.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase5 #LLM-as-judge #evaluation

---

**ID**: P5-038
**Front**: What are the key considerations when choosing chunk size for RAG?
**Back**: Small chunks (128–256 tokens): precise retrieval, may lack context, good for fact lookup. Large chunks (512–1024 tokens): more context, retrieval may be less precise, good for understanding. Factors: (1) Query type: factual = smaller, reasoning = larger. (2) Document type: articles = larger, FAQs = smaller. (3) Embedding model context window (all-MiniLM: 256 tokens max). (4) LLM context budget. Default starting point: 512 tokens with 50-token overlap, tune from there.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase5 #RAG #chunking #configuration

---

**ID**: P5-039
**Front**: What is a streaming response in FastAPI for LLM applications?
**Back**: `return StreamingResponse(token_generator(), media_type="text/event-stream")`. Server-Sent Events format: `data: {token}\n\n`. Client: EventSource JS API or `response.iter_content()`. In async generator: `async for chunk in openai_stream: yield f"data: {chunk}\n\n"`. End signal: `yield "data: [DONE]\n\n"`. Key: use `async` for generator — don't block event loop. FastAPI handles concurrency; streaming response is non-blocking.
**Difficulty**: Advanced
**Category**: Coding
**Tags**: #phase5 #fastapi #streaming #SSE

---

**ID**: P5-040
**Front**: What is the difference between langchain's `ConversationBufferMemory` and `ConversationSummaryMemory`?
**Back**: BufferMemory: keeps all messages in full (grows without bound). Hits context limit for long conversations. SummaryMemory: uses LLM to compress older messages into a summary. Keeps summary + recent messages. Fixed token budget. When use each: BufferMemory for short conversations (chatbots). SummaryMemory for long sessions (hours-long interactions). For RAG agents: also store key facts in vector memory for retrieval. Hybrid: WindowBuffer + Summary.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase5 #langchain #memory #conversation

---

**ID**: P5-041
**Front**: What is function/tool calling vs structured output? How do they differ?
**Back**: Tool/function calling: model requests execution of a function by name with JSON args. Model DOES NOT run the code — you run it and return the result. Model decides WHEN to call based on conversation. Structured output (`response_format={"type":"json_object"}`): model returns JSON as its final answer — no execution. Use tool calling for: agent loops, real-world actions. Use structured output for: parsing model's analysis/classification into typed objects.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase5 #function-calling #structured-output

---

**ID**: P5-042
**Front**: What is the Anthropic Messages API and how does it differ from OpenAI's?
**Back**: Key differences: (1) System prompt: separate `system` param (not inside messages list). (2) `max_tokens` is REQUIRED (OpenAI it's optional). (3) `model` names: "claude-opus-4-5" etc. (4) Rate limits use "tokens per minute" not RPM. (5) No function calling on older models (use `tools` param on Claude 3+). `import anthropic; client = anthropic.Anthropic()`. LiteLLM abstracts these differences.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase5 #anthropic #API

---

**ID**: P5-043
**Front**: What is the difference between RAG and fine-tuning for incorporating knowledge?
**Back**: RAG: retrieve at inference time → dynamic, updatable, exact source tracing, transparent, no training cost. Best for: factual queries, private KB, frequently changing info. Fine-tuning: bake knowledge into weights → no retrieval cost, consistent style, but: expensive to update, harder to verify source, can still hallucinate. Best for: style/format, domain vocabulary, behavior patterns. Rule: RAG for knowledge, fine-tuning for behavior. Combine: fine-tune then add RAG.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase5 #RAG #fine-tuning #comparison

---

**ID**: P5-044
**Front**: What is a guardrails library and name the key components of Guardrails AI?
**Back**: Guardrails AI: validates/corrects LLM output against defined rules. Components: (1) Validators: built-in checks (BugFreeCode, ValidJSON, ToxicLanguage, PIIFilter, etc.). (2) Rail spec: YAML/Python defining expected output structure + validators. (3) Guard object: wraps LLM call, validates output, retries if invalid. Integration: wraps OpenAI/Anthropic clients. Use when: output must meet specific quality criteria (code, medical, financial domains).
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase5 #guardrails #validation #production

---

**ID**: P5-045
**Front**: What is an evaluation dataset for RAG and how do you create one?
**Back**: Golden dataset: Q&A pairs with known correct answers and relevant document IDs. Creation: (1) Manual: domain experts write questions. (2) Synthetic: use GPT-4 to generate Q&A pairs from your documents (SynthESIS, Ragas testset generator). Include: simple factual, multi-hop reasoning, unanswerable questions (test abstention). Size: 100–500 examples. Use for: benchmarking retrieval MRR/Recall@k, RAGAS metrics, regression testing. Update as system evolves.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase5 #evaluation #RAG #testing
