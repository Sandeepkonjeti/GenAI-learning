# Phase 6 Flashcards — AI Agents & MCP

[← Flashcards Index](./README.md) | [Phase 6 File](../07_Phase6_AI_Agents_and_MCP.md) | [Cheat Sheet](../CheatSheets/Phase6_Agents.md)

---

**ID**: P6-001
**Front**: What is the ReAct pattern for AI agents? What does each letter stand for?
**Back**: ReAct = Reasoning + Acting. Pattern: Thought: [reason about what to do] → Action: [tool_name] → Action Input: [args] → Observation: [tool result] → repeat. Key insight: explicit reasoning before action reduces hallucinated tool calls. The model can course-correct: "Observation: Error: file not found. Thought: I should search for the file first." Terminates with "Final Answer:" after gathering enough info.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase6 #ReAct #agent #pattern

---

**ID**: P6-002
**Front**: What is LangGraph and how does it differ from LangChain's chains?
**Back**: LangChain chains: linear sequential (A→B→C). LangGraph: directed graphs with cycles — supports loops, branching, conditional routing. State: TypedDict that accumulates across nodes. Nodes: functions that transform state. Edges: routing logic (conditional or fixed). Why better: agents need loops (retry if tool fails), branching (route to different specialists), persistence (interrupt and resume). LangGraph = the standard for production agent systems.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase6 #langgraph #agent #graph

---

**ID**: P6-003
**Front**: What is the Model Context Protocol (MCP)?
**Back**: MCP: open standard (Anthropic 2024) for LLMs to communicate with external tools/services. Like USB-C for AI tools. Components: MCP server (exposes tools), MCP client (LLM host — Claude, LangGraph). Transport: stdio or HTTP/SSE. Tools exposed as JSON schema descriptions. Benefits: (1) Reusable servers (build once, use with any MCP client). (2) Standard protocol. (3) Growing ecosystem of pre-built servers. Differentiating project: Databricks MCP server.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase6 #MCP #protocol

---

**ID**: P6-004
**Front**: How do you prevent an agent from running forever in an infinite loop?
**Back**: (1) Max iterations hard cap: `max_iterations=10` or `recursion_limit=25` in LangGraph. (2) Loop detection: hash (node, input) pairs; stop if repeated. (3) Decreasing patience: after N retries without progress, abort. (4) Time limit: total wall-clock time budget. (5) Cost limit: track tokens spent, abort if over budget. (6) Human-in-the-loop interrupt: `interrupt_before="tool_node"` in LangGraph for high-risk operations.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase6 #agent #safety #loops

---

**ID**: P6-005
**Front**: What is the difference between a stateless and stateful agent?
**Back**: Stateless: no memory between sessions. Each invocation is independent — model only sees current conversation. Stateful: persists memory across sessions. Types: (1) In-context: keep conversation history. (2) External DB: store facts in database, retrieve when needed. (3) Vector memory: embed and store past interactions. For customer service bots: must be stateful (remember user profile). LangGraph: use `MemorySaver` or custom `SqliteSaver` for persistence.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase6 #agent #memory #stateful

---

**ID**: P6-006
**Front**: Why should you NEVER use `eval()` in an agent tool? What should you use instead?
**Back**: `eval()` executes arbitrary Python. Via prompt injection: attacker causes agent to call `eval("__import__('os').system('rm -rf /')")`. Even with "restricted" builtins: bypasses are well-documented. Use instead: `simpleeval` library — evaluates math expressions safely without code execution access. Or: implement a specific calculator rather than general eval. OWASP LLM Top 10: LLM08 Excessive Agency — tools must be minimal and safe.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase6 #security #eval #agent #simpleeval

---

**ID**: P6-007
**Front**: What is a supervisor pattern in multi-agent systems?
**Back**: Supervisor: orchestrator agent that delegates tasks to specialized worker agents. Structure: user → supervisor → {research_agent | code_agent | write_agent} → supervisor → user. Supervisor decides: which agent to call, when to combine results, when the task is complete. Workers: each has specialized tools and system prompt. Key challenge: supervisor must detect when workers fail or loop and handle gracefully. Implemented in LangGraph: supervisor is a node that routes via conditional edges.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase6 #multi-agent #supervisor #pattern

---

**ID**: P6-008
**Front**: What makes a good tool docstring for an LLM agent?
**Back**: The LLM reads the docstring to decide WHEN and HOW to use the tool. Good docstring: (1) First line: what the tool does and when to use it. (2) Args section: clear name, type, constraints. (3) Returns: describe format of output. (4) Example (optional but helpful). Bad: "Searches things." Good: "Search the internal knowledge base for technical documentation. Use when the user asks about product features, API specs, or engineering decisions. Returns up to 5 ranked results."
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase6 #agent #tools #docstring

---

**ID**: P6-009
**Front**: What is the `StateGraph` in LangGraph and how do you compile it?
**Back**: `StateGraph(StateType)` creates a graph over a typed state dict. Add nodes: `graph.add_node("agent", agent_fn)`. Set entry: `graph.set_entry_point("agent")`. Add edges: `graph.add_edge("tools", "agent")`. Add conditional: `graph.add_conditional_edges("agent", route_fn, {"tools": "tools", END: END})`. Compile: `app = graph.compile(checkpointer=MemorySaver())`. Run: `app.invoke({"messages": [...]}, config={"configurable":{"thread_id":"1"}})`.
**Difficulty**: Advanced
**Category**: Coding
**Tags**: #phase6 #langgraph #graph #code

---

**ID**: P6-010
**Front**: What is human-in-the-loop in agentic AI and how does LangGraph support it?
**Back**: Human-in-the-loop: pause agent execution for human review/approval before proceeding. LangGraph: `graph.compile(interrupt_before=["tool_node"])`. Agent pauses at tool_node, waits for human input. Resume: `app.invoke(Command(resume=human_input), config=...)`. Use for: high-risk actions (deleting files, sending emails, making purchases), low-confidence decisions, regulatory compliance. Principle: minimize autonomous actions in real-world systems.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase6 #human-in-the-loop #langgraph #safety

---

**ID**: P6-011
**Front**: What are the four types of memory an AI agent can use?
**Back**: (1) In-context (working memory): current conversation messages — fast, temporary, limited by context window. (2) Semantic (episodic): vector DB of past interactions — long-term, retrieved by similarity. (3) Procedural: system prompt rules and tool descriptions — how to behave. (4) Entity: structured facts about entities (user name, preferences) — stored in DB, retrieved by key. Production agents combine all four.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase6 #agent #memory #types

---

**ID**: P6-012
**Front**: What is tool output truncation and why is it important?
**Back**: Tools can return arbitrarily long output (web search, database query). If you return 50KB to the LLM: (1) Exceeds context window. (2) "Lost in the middle" — LLM may miss important parts. (3) Costs $$ in tokens. Best practice: truncate to max 2000 characters or a configurable `max_output_tokens`. Return structured summary if possible. For code execution output: return only last N lines + any exceptions.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase6 #agent #tools #production

---

**ID**: P6-013
**Front**: What is the Plan-and-Execute agent pattern?
**Back**: Unlike ReAct (interleaved planning+execution), Plan-and-Execute: (1) Planning phase: LLM generates complete multi-step plan for the task. (2) Execution phase: execute each step sequentially, possibly replanning if a step fails. Benefits: (1) Better for complex long-horizon tasks. (2) Can parallelize independent steps. (3) Easier to audit the plan. (4) More predictable behavior. LangGraph: separate "planner" and "executor" nodes. Also called "planner-actor" or "orchestrator-worker."
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase6 #plan-execute #agent #pattern

---

**ID**: P6-014
**Front**: How do you implement a Text-to-SQL agent safely?
**Back**: (1) Validate query: only allow SELECT (not INSERT/UPDATE/DELETE). (2) Schema-aware prompting: include table definitions in system prompt. (3) Error correction: if SQL fails, feed error back to LLM and retry (up to 3 times). (4) Row limit: always append `LIMIT 100` if not present. (5) Parameterized queries: avoid string injection (user input should be in WHERE clause via parameters, not concatenation). (6) Read-only DB user: least privilege.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase6 #text-to-sql #agent #security

---

**ID**: P6-015
**Front**: What is parallel tool execution in agents and when is it beneficial?
**Back**: Many agents call tools sequentially even when tools are independent. Parallel: `asyncio.gather(tool1(args1), tool2(args2), tool3(args3))`. When beneficial: independent sub-questions that each need a search/lookup. Example: "Compare pricing for products A, B, and C" → three independent searches in parallel. Reduce latency from 3×T to ~T. LangGraph: use `send()` API to fan out to parallel nodes. Challenge: collecting and merging results.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase6 #parallel #tools #agent

---

**ID**: P6-016
**Front**: What is Reflexion agent pattern?
**Back**: Reflexion (Shinn et al. 2023): add a self-reflection step after each attempt. Loop: (1) Attempt task. (2) Evaluate output (self-grade or external). (3) Reflect: "what did I do wrong? how can I improve?" (4) Retry with reflection in context. (5) Keep best attempt. Why effective: iterative self-improvement without weight updates. Use for: coding (run tests, fix bugs), writing (self-critique then revise), math (check arithmetic).
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase6 #reflexion #agent #pattern

---

**ID**: P6-017
**Front**: What is the handoff protocol in multi-agent systems?
**Back**: Structured message format when one agent passes work to another. Should include: (1) Task description (what needs to be done). (2) Completed work so far (context). (3) Specific request (what the receiving agent should do). (4) Success criteria (how to know when done). (5) Constraints (tools available, time budget). Without structured handoffs: agents misunderstand the task and redo work. LangGraph: encode in state TypedDict.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase6 #multi-agent #handoff

---

**ID**: P6-018
**Front**: What is LangSmith tracing for agents and why is it essential?
**Back**: Agents run 5–20 LLM calls per user request with branching logic — debugging is nearly impossible without tracing. LangSmith: captures every LLM call (inputs, outputs, latency, tokens), every tool call, agent decision points, full graph traversal. Set `LANGCHAIN_TRACING_V2=true; LANGCHAIN_API_KEY=...`. View at smith.langchain.com. Essential for: debugging unexpected behavior, finding where the agent fails, evaluating agent performance offline.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase6 #langsmith #observability #debugging

---

**ID**: P6-019
**Front**: What is the Annotated TypedDict pattern in LangGraph state?
**Back**: `from typing import Annotated; import operator`. `class AgentState(TypedDict): messages: Annotated[list, operator.add]; tool_calls: int`. `Annotated[list, operator.add]` means: when merging state from parallel branches, ADD the lists (not replace). Without Annotated: parallel nodes overwrite each other. `operator.add` for lists (append). Custom reducers for deduplication or last-write-wins. Every LangGraph state key should have explicit reducer.
**Difficulty**: Advanced
**Category**: Coding
**Tags**: #phase6 #langgraph #state #annotated

---

**ID**: P6-020
**Front**: What is an MCP tool schema and what fields are required?
**Back**: MCP tool: `types.Tool(name="tool_name", description="What this tool does...", inputSchema={...})`. inputSchema: JSON Schema object. Required fields: `type`, `properties`, `required`. Example: `{"type":"object","properties":{"query":{"type":"string","description":"Search query"},"limit":{"type":"integer","minimum":1,"maximum":20}},"required":["query"]}`. The description is critical — LLM reads it. MCP client presents tools to LLM; LLM decides which to call.
**Difficulty**: Advanced
**Category**: Coding
**Tags**: #phase6 #MCP #tools #schema

---

**ID**: P6-021
**Front**: What is the difference between an agent and a chain in LangChain?
**Back**: Chain: fixed, deterministic sequence of steps (A→B→C always). No decision-making. Agent: uses LLM to DECIDE which tools to call and in what order — dynamic. Chains: cheaper, predictable, testable. Agents: flexible, handle open-ended tasks, but non-deterministic, harder to test, can fail in unexpected ways. Use chain when: sequence is fixed. Use agent when: need dynamic tool selection based on input. LangGraph generalizes both.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase6 #agent #chain #langchain

---

**ID**: P6-022
**Front**: What tool should an agent use to safely evaluate arithmetic expressions?
**Back**: `simpleeval` library: `from simpleeval import simple_eval`. `simple_eval("2 + 3 * 4")` → 14. Supports: basic math, variables, comparison, boolean logic. Does NOT execute Python code — only evaluates mathematical expressions. Install: `pip install simpleeval`. Never use Python's `eval()` — exploitable via prompt injection regardless of `__builtins__` restrictions. For more complex calculations: have the agent write Python code and execute in a sandbox (Docker, subprocess with timeout).
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase6 #simpleeval #security #tools

---

**ID**: P6-023
**Front**: What is the "excessive agency" risk in AI agents (OWASP LLM08)?
**Back**: LLM performs actions with more autonomy, scope, or impact than required. Example: agent with DELETE permission when only READ is needed. Mitigations: (1) Principle of least privilege: only expose tools the agent actually needs. (2) Tool output validation: check that tool ran as expected. (3) Human approval for destructive operations. (4) Reversibility: prefer reversible actions (trash → permanent delete). (5) Audit log of all tool calls. This is the agent security concern.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase6 #security #OWASP #excessive-agency

---

**ID**: P6-024
**Front**: What is `interrupt_before` in LangGraph and how does it enable human oversight?
**Back**: `app = graph.compile(interrupt_before=["dangerous_tool_node"])`. When execution reaches dangerous_tool_node, it PAUSES and returns current state to caller. Human can: approve (resume), modify (update state then resume), or reject (stop). Resume: `app.invoke(Command(resume="approved"), config=config)`. Use for: email sending, database writes, API calls with side effects, irreversible actions. Key for safe agentic systems.
**Difficulty**: Advanced
**Category**: Coding
**Tags**: #phase6 #langgraph #interrupt #human-in-loop

---

**ID**: P6-025
**Front**: What is a "swarm" multi-agent architecture?
**Back**: Swarm: all agents have the same tools but work on different parts of a problem in parallel. No dedicated supervisor. Agents coordinate via shared state. OpenAI's `swarm` library (experimental): lightweight, agents hand off to each other via function calls. Use for: parallel data processing, ensemble approaches, redundancy. Compare to supervisor: supervisor = hierarchical (better for specialized tasks); swarm = flat (better for parallelism).
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase6 #swarm #multi-agent

---

**ID**: P6-026
**Front**: How do you handle tool failures gracefully in an agent loop?
**Back**: Tools can fail: API timeout, bad input, rate limit, etc. Graceful handling: (1) Return error description as string (not raise exception) — let LLM decide how to respond. (2) Retry with backoff for transient failures (3 retries max). (3) Try alternative tool: "web_search failed, trying knowledge_base." (4) Report to user: "I couldn't access the service. Here's what I know from my training..." (5) Log error for debugging. Never surface raw stack traces to users.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase6 #agent #error-handling

---

**ID**: P6-027
**Front**: What is the difference between MCP stdio transport and HTTP transport?
**Back**: Stdio: MCP server communicates via standard input/output streams. Local only. Simple to implement. No network. Used for: CLI tools, local IDE integrations (Cursor). HTTP/SSE: server exposes HTTP endpoints. Remote access. Multiple clients. Use for: shared infrastructure, web-deployed MCP servers. Claude Desktop: uses stdio. LangGraph / remote access: use HTTP. Security note: HTTP transport needs authentication — MCP has no built-in auth; add it in your server framework.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase6 #MCP #transport #stdio #HTTP

---

**ID**: P6-028
**Front**: What is the "oracle" anti-pattern in agent design?
**Back**: Oracle anti-pattern: agent designed to answer ANY question with any tool, but in practice lacks the knowledge/tools to handle most queries well. Symptoms: frequent "I can't help with that" or hallucinated answers when tools fail. Fix: (1) Clearly scope the agent's domain. (2) Add a "not_my_domain" response path. (3) Design fallback behavior for out-of-scope queries. (4) Test agent on negative examples (queries it shouldn't handle). Scope-limited agents perform much better.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase6 #agent #anti-pattern #design

---

**ID**: P6-029
**Front**: How do you test an AI agent?
**Back**: Challenges: non-deterministic, multi-step, tool interactions. Approaches: (1) Unit test individual tools (deterministic). (2) Mock LLM responses for deterministic agent path testing. (3) Golden trace tests: record successful agent traces, replay and verify same tool calls. (4) Evaluation dataset: queries with expected final answers — run agent and score. (5) Canary queries: known-answer queries run regularly in production. (6) LangSmith evaluators: automated evaluation against criteria.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase6 #testing #agent

---

**ID**: P6-030
**Front**: What is a Databricks MCP server and why is it valuable as a project?
**Back**: MCP server that exposes Databricks capabilities as AI tools: run SQL queries, list tables/schemas, read Delta tables, trigger jobs, query MLflow experiments. Value: (1) Any MCP-compatible LLM can now query Databricks (Claude, LangGraph). (2) Natural language to Databricks SQL. (3) Unique project — very few exist. (4) Directly leverages your data engineering background. Tools to expose: `run_sql`, `list_tables`, `get_schema`, `search_catalog`, `run_job`. Security: read-only tokens, SELECT-only SQL validation.
**Difficulty**: Advanced
**Category**: Production
**Tags**: #phase6 #MCP #databricks #project

---

**ID**: P6-031
**Front**: What is the difference between a tool call and a subagent call?
**Back**: Tool call: structured API call to a specific function (search_db, run_code). Returns data. No independent reasoning. Subagent call: delegating to another LLM agent with its own tools, memory, and reasoning. Returns completed work. Tool call = function. Subagent = specialist. When to use subagent: task requires multi-step reasoning by a specialist (code agent for complex code, research agent for deep research). Subagents are more expensive but more capable.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase6 #agent #subagent #tools

---

**ID**: P6-032
**Front**: What is Tavily and how is it used in agents?
**Back**: Tavily: search API optimized for AI agents. Returns structured results (URL, title, content snippets) formatted for LLM consumption. Handles: Google search, news, academic papers. Free tier: 1000 queries/month. `from tavily import TavilyClient; client.search(query, max_results=5)`. Alternative to: Serper API, Bing Search API. Use for: web-grounded agents that need current information (CRAG fallback, research agents). Essential for Project 16 (Personal Research Agent).
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase6 #tavily #search #agent

---

**ID**: P6-033
**Front**: What is the LangGraph `Send` API and when is it used?
**Back**: `Send("node_name", state)` sends work to a node in parallel. Used for fan-out patterns: supervisor generates N subtasks → `[Send("worker", {"task": t}) for t in tasks]` → all workers run in parallel → results merged via Annotated reducer. Without Send: sequential loops. With Send: true parallelism. Use for: multi-query RAG (query each search engine in parallel), parallel tool calls, ensemble agent voting.
**Difficulty**: Advanced
**Category**: Coding
**Tags**: #phase6 #langgraph #parallel #send

---

**ID**: P6-034
**Front**: What is the difference between `langchain` and `langchain_community`?
**Back**: `langchain`: core abstractions (chains, agents, tools, memory, callbacks). Stable, minimal deps. `langchain_community`: integrations with third-party services (vector stores, LLMs, tools). Many contributors, variable quality. `langchain_openai`: OpenAI-specific (ChatOpenAI, OpenAIEmbeddings). `langchain_anthropic`: Anthropic-specific. Import from specific packages: `from langchain_openai import ChatOpenAI` (not `from langchain.chat_models import ChatOpenAI` — deprecated). LangGraph is separate package.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase6 #langchain #imports

---

**ID**: P6-035
**Front**: What are three signs that an agent needs a human-in-the-loop checkpoint?
**Back**: (1) **Irreversibility**: action cannot be undone (sending email, deleting data, making payment). (2) **High cost**: action has significant financial or reputational impact. (3) **Low confidence**: agent's confidence in the interpretation is below threshold (ambiguous user intent). Also: (4) Sensitive data access (PII, financial records). (5) External party impact (posting to social media, contacting customers). Principle: autonomous actions OK for reversible, low-stakes, high-confidence situations.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase6 #human-in-loop #agent #safety
