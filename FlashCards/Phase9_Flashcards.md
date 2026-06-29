# Phase 9 Flashcards — Production AI Engineering

[← Flashcards Index](./README.md) | [Phase 9 File](../10_Phase9_Production_AI_Engineering.md) | [Cheat Sheet](../CheatSheets/Phase9_Production.md)

---

**ID**: P9-001
**Front**: What are TTFT and TPOT? What are good production targets?
**Back**: TTFT (Time to First Token): latency from request to first token in stream. User perceives this as responsiveness. Target: < 500ms. TPOT (Time Per Output Token): latency for each subsequent token. Target: < 30ms (=~33 tokens/s, faster than reading). Total latency = TTFT + TPOT × n_tokens. GPU bound: TPOT determined by matmul speed. Network bound: TTFT includes request travel + queue time. Profile separately: TTFT optimization ≠ TPOT optimization.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase9 #latency #TTFT #TPOT #metrics

---

**ID**: P9-002
**Front**: What is a `StreamingResponse` in FastAPI for LLM apps?
**Back**: `return StreamingResponse(generator(), media_type="text/event-stream")`. Generator yields SSE-formatted strings: `yield f"data: {token}\n\n"`. End signal: `yield "data: [DONE]\n\n"`. Async generator: `async def generator(): async for chunk in llm_stream: yield...`. Critical: use async generator — don't block event loop. Client: `EventSource` JS API or `requests` with `stream=True`. FastAPI handles concurrent streams via asyncio event loop.
**Difficulty**: Advanced
**Category**: Coding
**Tags**: #phase9 #fastapi #streaming #SSE

---

**ID**: P9-003
**Front**: What is the OWASP LLM Top 10? Name the top 5 risks.
**Back**: LLM01: Prompt Injection — user input overrides instructions. LLM02: Insecure Output Handling — trusting LLM output without validation. LLM03: Training Data Poisoning — backdoors via training data. LLM04: Model DoS — expensive queries exhaust budget. LLM05: Supply Chain — compromised model or dependency. LLM06: Sensitive Info Disclosure — PII in responses. LLM07: Insecure Plugin Design — tools without input validation. LLM08: Excessive Agency. LLM09: Overreliance. LLM10: Model Theft.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #OWASP #security #LLM

---

**ID**: P9-004
**Front**: How do you implement rate limiting for an LLM API?
**Back**: Token bucket: allow N requests per minute per user. Implementation: Redis-based sliding window counter. `redis.incr(f"rate:{user_id}:{minute}")`. If count > limit → 429 Too Many Requests. FastAPI middleware: `from slowapi import Limiter`. Per-token budget: track `usage.total_tokens` per user per day, reject when exceeded. Best practice: rate limit by both requests AND tokens (expensive prompts can exhaust budget with few requests). Include `Retry-After` header in 429 response.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #rate-limiting #API #security

---

**ID**: P9-005
**Front**: What should you log for every LLM request in production?
**Back**: Minimum: (1) Request ID (UUID). (2) Timestamp + duration. (3) Model name + version. (4) Prompt token count. (5) Completion token count. (6) Cost estimate. (7) TTFT + total latency. (8) Status (success/error/timeout). (9) User/session ID (anonymized). (10) Response (sampled or full). Never log: (1) API keys. (2) PII in plaintext. (3) Sensitive user content (check ToS). Use: structured JSON logging → Elasticsearch or Databricks Delta table. Sample 100% early; reduce to 10% at scale.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase9 #logging #observability

---

**ID**: P9-006
**Front**: What is MLflow and how is it used for LLM experiment tracking?
**Back**: MLflow: open-source MLOps platform. `with mlflow.start_run(): mlflow.log_params({...}); mlflow.log_metrics({...})`. For LLMs: `mlflow.log_text(prompt, "prompt.txt")`. LLM-specific: `mlflow.openai.autolog()` — auto-logs all OpenAI calls. Evaluation: `mlflow.evaluate(model_uri, data, evaluators=["default"])`. Model registry: version, stage (Staging/Production). Databricks: MLflow built-in, integrated with Unity Catalog model registry. Also: model serving from MLflow models.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase9 #mlflow #MLops #tracking

---

**ID**: P9-007
**Front**: What is semantic caching and how do you implement it?
**Back**: Semantic cache: cache LLM responses keyed by query embedding similarity (not exact string match). Query arrives → embed → search cache (cosine similarity > 0.95 → hit, return cached). Miss → call LLM → store in cache. Implementation: Redis + vector index (Redis Vector Similarity Search or `redis-om-python`). Hit rate: 30–70% for repeated queries. Threshold matters: too low → wrong cache hits (similar but different intent). Too high → low hit rate. Cache TTL: set based on content volatility.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #semantic-cache #redis

---

**ID**: P9-008
**Front**: What is a CI/CD evaluation gate for LLM systems?
**Back**: Gate: block deployment if LLM quality metrics regress. Implementation: (1) Maintain golden Q&A test set. (2) On every PR: run test set through new model/prompt/pipeline. (3) Compute: RAGAS faithfulness, relevance, latency P95. (4) Compare vs baseline metrics. (5) Fail CI if any metric drops > threshold. GitHub Actions: `python evaluate.py --min-faithfulness 0.85; if exit 1: fail`. Why critical: prompt changes silently degrade quality without CI gates. Deploy only if metrics pass.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #CI-CD #evaluation #testing

---

**ID**: P9-009
**Front**: What is the Databricks Mosaic AI Model Serving?
**Back**: Serverless LLM deployment on Databricks. Features: (1) Serve custom models (HuggingFace, mlflow). (2) Serve foundation models (LLaMA, Mistral, DBRX via Model Serving endpoints). (3) Auto-scaling (scale to zero when idle, scale up on traffic). (4) OpenAI-compatible API. (5) Pay per token. Create: `mlflow.models.serve(model_uri, port=8080)` or UI. Advantages: native Delta Lake + Unity Catalog integration, MLflow tracking built-in. Use for: production serving when using Databricks as primary platform.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #databricks #model-serving

---

**ID**: P9-010
**Front**: What is the Mosaic AI Vector Search in Databricks?
**Back**: Managed vector database on Databricks backed by Delta Lake. Features: (1) Sync with Delta table — auto-update embeddings when source table changes. (2) Supports: dot product, cosine similarity, L2 distance. (3) Metadata filtering. (4) OpenAI + Databricks embedding models. Create: `VectorSearchClient().create_delta_sync_index(endpoint_name, index_name, source_table, ...)`. Use for: RAG on Databricks without managing external vector DB. Advantage: stays in sync with Unity Catalog data.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #databricks #vector-search

---

**ID**: P9-011
**Front**: What is the Databricks AI Gateway and what problem does it solve?
**Back**: AI Gateway: centralized proxy for all LLM API calls within an organization. Features: (1) Rate limiting per user/team. (2) Cost tracking by user/department. (3) Request/response logging. (4) PII detection. (5) Model fallback. (6) A/B testing between models. Problem: 100 teams using GPT-4 → no visibility, no cost control, no audit trail. AI Gateway provides: a single entry point with governance. Databricks: `ai_gateway` route in model serving config. Enables: org-wide LLM governance without changing app code.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #databricks #AI-gateway #governance

---

**ID**: P9-012
**Front**: What is a Prometheus metric and how do you instrument an LLM API?
**Back**: Prometheus: time-series metrics DB with pull model (scrapes `/metrics` endpoint). Instrument with `prometheus_client`. Key metrics for LLM API: `llm_request_total` (Counter), `llm_request_duration_seconds` (Histogram), `llm_token_total` (Counter by type: prompt/completion), `llm_cost_usd` (Counter), `cache_hit_total` (Counter). `Histogram.observe(latency)` → P50/P95/P99 percentiles in Grafana. Alert: P95 latency > 3s, error_rate > 1%, cost_per_hour > $X.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #prometheus #monitoring #metrics

---

**ID**: P9-013
**Front**: What is LangSmith dataset evaluation and how does it work?
**Back**: (1) Create LangSmith dataset from examples: `client.create_dataset("RAG test")`. (2) Add examples: `client.create_examples(inputs=[...], outputs=[...], dataset_id=...)`. (3) Define evaluator: function that takes (run, example) → score. (4) Run evaluation: `evaluate(your_chain, data=dataset, evaluators=[evaluator])`. (5) View results in LangSmith UI — compare runs. Integrate with CI: `pytest tests/` → calls LangSmith evaluations. Automated quality tracking across code changes.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #langsmith #evaluation #CI

---

**ID**: P9-014
**Front**: What is Docker Compose and how is it used in AI application deployment?
**Back**: Docker Compose: define multi-container apps in `docker-compose.yml`. `services: api: image: my-api; redis: image: redis:7; qdrant: image: qdrant/qdrant`. `depends_on: redis, qdrant`. `docker compose up -d` → starts all services. Use for: local dev stack, staging environment. Production: Kubernetes (orchestration). For AI apps: api (FastAPI), redis (cache), qdrant (vector db), prometheus (metrics). Benefit: reproducible environment; no "works on my machine."
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase9 #docker #deployment

---

**ID**: P9-015
**Front**: What is a PII detection middleware for LLM APIs and why is it needed?
**Back**: PII (Personally Identifiable Information): names, emails, SSNs, credit cards, etc. in LLM prompts/responses. Risks: (1) Logging PII → privacy violation, regulatory non-compliance (GDPR, HIPAA). (2) LLM provider sees sensitive data. (3) PII leaks in responses. Detection: presidio (Microsoft), spacy NER, regex patterns. Middleware: scan input before sending to LLM, scan output before returning. Actions: anonymize (replace with PERSON_1), reject request, alert. Databricks AI Gateway has built-in PII detection option.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #PII #privacy #security #middleware

---

**ID**: P9-016
**Front**: What is model A/B testing in production and how do you implement it?
**Back**: A/B test: route X% of traffic to model A, Y% to model B. Compare: quality metrics (RAGAS, user feedback), latency, cost. Implementation: (1) Feature flag: `if random() < 0.1: use model_b else: use model_a`. (2) Log which model was used. (3) Collect user feedback (thumbs up/down). (4) Statistical test after sufficient samples. Tools: LaunchDarkly, Databricks AI Gateway, custom. Run for: at least 1000 samples or 1 week before concluding. Beware: novelty effect (users rate new things higher initially).
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #A-B-testing #model-serving

---

**ID**: P9-017
**Front**: What is a health check endpoint and why is it needed for LLM services?
**Back**: `@app.get("/health")`: returns `{"status": "ok", "model": "gpt-4o", "version": "1.2.0"}`. Used by: load balancer (route traffic only to healthy instances), Kubernetes liveness/readiness probes. Liveness: is the service running? Readiness: is the service ready for traffic (model loaded, dependencies connected)? LLM-specific check: verify LLM API is reachable (ping OpenAI), verify vector DB connection. If unhealthy: return 503, load balancer routes around. Essential for zero-downtime deployments.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase9 #health-check #reliability

---

**ID**: P9-018
**Front**: What is MLflow Model Registry and how does it enable production AI workflows?
**Back**: Model Registry: version-controlled model store with stage transitions. Stages: None → Staging → Production → Archived. Workflow: (1) Log model: `mlflow.pyfunc.log_model(...)`. (2) Register: `mlflow.register_model(run_uri, "my-model")`. (3) Promote to Staging (automated testing), then Production (manual approval). (4) Load in serving: `mlflow.pyfunc.load_model("models:/my-model/Production")`. Databricks: integrated with Unity Catalog for access control and lineage. Enables: reproducible deployments, rollback on failure.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase9 #mlflow #model-registry #MLops

---

**ID**: P9-019
**Front**: What is LLM observability and how does it differ from traditional API monitoring?
**Back**: Traditional API: monitor HTTP status codes, latency, throughput. LLM observability: all of that PLUS: (1) Prompt quality (prompt drift over time). (2) Output quality (faithfulness, relevance). (3) Token usage trends. (4) Cost per query (not just latency). (5) Topic distribution of queries. (6) Model behavior changes (e.g., GPT-4 API silently updated). (7) Hallucination rates. Tools: LangSmith, Helicone, W&B Weave, Langfuse. Key challenge: output quality is subjective — need automated eval.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #observability #LLM #monitoring

---

**ID**: P9-020
**Front**: What is the difference between `sync` and `async` FastAPI endpoints for LLM serving?
**Back**: Sync `def chat(...)`: FastAPI runs in thread pool. If 100 concurrent requests, 100 threads blocked waiting for LLM response. Thread pool size limits concurrency. Async `async def chat(...)`: FastAPI uses asyncio. Awaits LLM responses without blocking. All 100 requests handled concurrently by one event loop thread. For I/O-bound work (LLM calls, DB queries): ALWAYS use async. Use `asyncio.gather` for parallel LLM calls. Sync code in async: `loop.run_in_executor(None, sync_function)`.
**Difficulty**: Advanced
**Category**: Coding
**Tags**: #phase9 #fastapi #async #performance

---

**ID**: P9-021
**Front**: What is Grafana and how is it used to visualize LLM system metrics?
**Back**: Grafana: open-source dashboards connected to Prometheus (and other sources). Create panels: time series (latency over time), stat (total requests), bar gauge (requests by model). Key LLM dashboard panels: TTFT P50/P95, tokens/second, cost per hour, cache hit rate, error rate, active requests. Alert: set thresholds → send to Slack/PagerDuty. `docker run -p 3000:3000 grafana/grafana`. Data source: add Prometheus. Import community dashboards for FastAPI + Prometheus.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase9 #grafana #monitoring #dashboard

---

**ID**: P9-022
**Front**: What is the TorchDistributor in Databricks and how does it work?
**Back**: `from pyspark.ml.torch.distributor import TorchDistributor`. Launches distributed PyTorch training across Databricks cluster. `TorchDistributor(num_processes=4, local_mode=False, use_gpu=True).run(train_fn)`. Handles: NCCL initialization, process group setup, checkpoint coordination. Train function: standard PyTorch DDP code. Integration: Unity Catalog for data access, MLflow for tracking, Delta for checkpoints. Supports: DDP, FSDP, DeepSpeed (via Accelerate). Best for: fine-tuning LLMs on Databricks with enterprise data.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #databricks #TorchDistributor #training

---

**ID**: P9-023
**Front**: What is an inference endpoint timeout strategy?
**Back**: LLM inference can take 10-60+ seconds for long outputs. Timeout strategy: (1) HTTP timeout: client-side max wait (e.g., 30s). (2) Server-side: `asyncio.wait_for(llm_call(), timeout=25)`. (3) Streaming: avoids timeout (first token arrives quickly, rest stream). (4) Task queue: for long tasks, return job ID → client polls. (5) Budget tokens: `max_tokens=200` ensures bounded generation time. Best practice: stream everything → never timeout on first token. Long-form: use background job + webhook callback.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #timeout #inference #reliability

---

**ID**: P9-024
**Front**: What is a canary deployment for LLM model updates?
**Back**: Canary: route 1-5% of traffic to new model, 95-99% to current. Monitor: latency, error rate, user feedback, quality metrics for canary group. If metrics OK after 1-24h → gradually increase to 100%. If regression → rollback (redirect all traffic back). Benefits: limit blast radius of bugs. LLM-specific: also monitor output quality (RAGAS, thumbs down rate) — model update can be functionally correct but lower quality. Feature flags: control canary percentage without redeployment.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #canary #deployment #reliability

---

**ID**: P9-025
**Front**: What are the key components of a production RAG system architecture?
**Back**: Full stack: (1) FastAPI async server. (2) Redis semantic cache. (3) Qdrant/Databricks Vector Search for retrieval. (4) BM25 index for hybrid search. (5) Reranker (cross-encoder). (6) LLM with streaming (via LiteLLM). (7) LangSmith for tracing. (8) Prometheus + Grafana for metrics. (9) RAGAS evaluation CI gate. (10) MLflow model registry. (11) Docker Compose for local dev. (12) Kubernetes/Databricks for production. This is Project 22 — the portfolio centerpiece.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #RAG #architecture #production

---

**ID**: P9-026
**Front**: What is prompt versioning and why should you treat prompts like code?
**Back**: Prompts are code: small wording changes can significantly affect output. Prompt versioning: (1) Store prompts in version control (Git). (2) Each prompt version has a unique identifier. (3) Link prompt version to experiment/model run. (4) When prompt changes → run evaluation suite. (5) Track which prompt version is in production. Tools: LangSmith Prompt Hub, Langfuse, custom DB. Anti-pattern: hardcoded strings in app code → no versioning, no rollback. Like database migrations: treat prompt changes as deployments.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #prompt-versioning #MLops

---

**ID**: P9-027
**Front**: What is Pydantic BaseSettings and how is it used for LLM app configuration?
**Back**: `from pydantic_settings import BaseSettings`. `class Settings(BaseSettings): openai_api_key: str; model: str = "gpt-4o"; max_tokens: int = 1000; model_config = SettingsConfigDict(env_file=".env")`. Usage: `settings = Settings()`. Auto-reads from environment variables AND `.env` file. Type validation on startup (fail fast if OPENAI_API_KEY missing). Benefits: (1) No `os.environ.get()` scattered throughout code. (2) Type-safe config. (3) Single source of truth. (4) Easy testing (override via env).
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase9 #pydantic #config #settings

---

**ID**: P9-028
**Front**: What is an LLM circuit breaker pattern?
**Back**: Circuit breaker: if LLM API fails repeatedly, "open" the circuit (stop sending requests) for a cooldown period, then try again. States: Closed (normal) → Open (failing) → Half-open (testing recovery). Implementation: `pybreaker` library. `@breaker` decorator on LLM call function. Benefits: (1) Prevent cascading failures. (2) Fail fast (no waiting for timeouts). (3) Allow recovery. (4) Fallback to cheaper model or cached response during outage. Essential for: production systems using external LLM APIs.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #circuit-breaker #reliability #fallback

---

**ID**: P9-029
**Front**: What is the difference between hot and cold LLM model serving?
**Back**: Hot serving: model weights loaded in GPU memory, ready to serve requests immediately. Cold start: model must be loaded from disk/network first (10s–5min). Implications: (1) Serverless (Databricks Model Serving, AWS SageMaker): may cold start for infrequent traffic → high initial latency. (2) Always-on: no cold start, higher cost when idle. (3) Scale-to-zero: reduce cost when no traffic, accept cold start. Solutions: keep-warm requests (ping endpoint every N minutes), minimum instance=1. Choose based on traffic pattern.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #model-serving #cold-start

---

**ID**: P9-030
**Front**: What LLM benchmark should you run to evaluate a fine-tuned model before production?
**Back**: Domain evaluation first: create 100–500 test cases from your specific use case — most important. Generic benchmarks: (1) MMLU: overall knowledge preservation (check didn't regress). (2) MT-Bench: instruction following quality. (3) HumanEval: if code is relevant. (4) Custom: task-specific eval against your golden dataset. Run before/after fine-tuning to compare. Key: did fine-tuning improve target task without catastrophically hurting general capabilities? Check MMLU drop < 2% is a common threshold.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #evaluation #fine-tuning #benchmarks

---

**ID**: P9-031
**Front**: What is MLflow `pyfunc` and why is it used for LLM deployment?
**Back**: `mlflow.pyfunc`: generic model format for any Python object. For LLMs: wrap prompt template + LLM client in pyfunc. `class LLMModel(mlflow.pyfunc.PythonModel): def predict(self, ctx, model_input): ...`. Log: `mlflow.pyfunc.log_model("llm_model", python_model=LLMModel(), ...)`. Load: `model = mlflow.pyfunc.load_model("runs:/run_id/llm_model")`. Benefits: (1) Version controlled. (2) Reproducible (captures deps). (3) Deploy to Databricks Model Serving directly. (4) A/B test different prompt versions.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #mlflow #pyfunc #deployment

---

**ID**: P9-032
**Front**: What is a model fallback chain and how do you implement it?
**Back**: Fallback: if primary model fails/is slow/unavailable → try secondary → tertiary. Implementation with LiteLLM: `completion(model=["gpt-4o", "claude-opus-4-5", "gpt-3.5-turbo"], ...)` — tries in order. Or: `try: await gpt4_call(); except (RateLimitError, APIError): await claude_call()`. Fallback strategy: same quality model (OpenAI → Anthropic) or graceful degradation (GPT-4o → GPT-4o-mini). Timeout-based fallback: if primary takes > 3s, fallback to faster model.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #fallback #reliability #litellm

---

**ID**: P9-033
**Front**: What is the difference between synchronous evaluation and asynchronous evaluation in LLM CI/CD?
**Back**: Synchronous: run eval in the PR pipeline — blocks merge until eval passes. Best for: fast evals (<5 min). Catches regressions before merge. Asynchronous: trigger eval after deploy, notify if fails. Best for: slow/expensive evals. More realistic (tests production system). Combined strategy: (1) Fast eval (100 samples) in CI — blocks merge. (2) Full eval (1000 samples) after deploy — alert if fails. (3) Continuous eval: run golden set every hour in production.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #CI-CD #evaluation #async

---

**ID**: P9-034
**Front**: What is the MLflow AI Gateway (previously Deployments)?
**Back**: `mlflow.deployments`: proxy API for LLM providers. `client = mlflow.deployments.get_deploy_client("databricks")`. `client.predict(endpoint="databricks-dbrx-instruct", inputs={"messages": [...]})`. Supported: OpenAI, Anthropic, Cohere, Databricks, Azure OpenAI. Benefits: (1) Unified API across providers. (2) Rate limiting. (3) Cost tracking. (4) Audit logging. Different from Databricks AI Gateway (UI-configured); MLflow deployments is code-configured. Use for: standardizing LLM access in Databricks notebooks and jobs.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #mlflow #AI-gateway #databricks

---

**ID**: P9-035
**Front**: What are the key considerations for LLM data privacy in production?
**Back**: (1) Data residency: does the API provider store requests? OpenAI: no training on API data (with paid plan). Anthropic: similar. (2) PII scrubbing: detect and anonymize before sending. (3) Encryption: TLS in transit, encryption at rest for logs. (4) Access control: who can see prompt/response logs? Audit trail. (5) Retention: how long are logs kept? Set deletion policy. (6) EU/GDPR: Azure OpenAI for EU data residency. (7) On-prem option: Ollama/vLLM for highly sensitive data. Never send HIPAA/PCI data to shared APIs without BAA.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase9 #privacy #GDPR #security #compliance
