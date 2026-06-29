# Phase 6: AI Agents & MCP (Model Context Protocol)
**Months 13–14 | Difficulty: 8/10 | Label: 🔴 Must Learn**

> **Previous Phase**: [Phase 5b — Embeddings, VectorDB & RAG](./06_Phase5_Part2_Embeddings_VectorDB_RAG.md)  
> **Next Phase**: [Phase 7a — Fine-Tuning & LoRA](./08_Phase7_Part1_FineTuning_and_LoRA.md)

---

## Table of Contents

- [Phase Overview](#phase-overview)
- [Part A: The Agent Mental Model](#part-a-the-agent-mental-model)
  - [What Is an Agent?](#what-is-an-agent)
  - [The Agent Loop: Reason → Act → Observe](#the-agent-loop-reason--act--observe)
  - [ReAct: Synergizing Reasoning and Acting](#react-synergizing-reasoning-and-acting)
  - [Agent Memory Types](#agent-memory-types)
- [Part B: Tool Use](#part-b-tool-use)
  - [Function Calling (OpenAI)](#function-calling-openai)
  - [Anthropic Tool Use](#anthropic-tool-use)
  - [Building Custom Tools](#building-custom-tools)
- [Part C: Planning Patterns](#part-c-planning-patterns)
  - [Single Agent with Tools](#single-agent-with-tools)
  - [Multi-Agent Systems](#multi-agent-systems)
  - [Plan-and-Execute Pattern](#plan-and-execute-pattern)
- [Part D: LangGraph for Stateful Agents](#part-d-langgraph-for-stateful-agents)
  - [Why Stateful Agents Need Graphs](#why-stateful-agents-need-graphs)
  - [Core LangGraph Concepts](#core-langgraph-concepts)
  - [Building a Research Agent with LangGraph](#building-a-research-agent-with-langgraph)
- [Part E: Model Context Protocol (MCP)](#part-e-model-context-protocol-mcp)
  - [What Is MCP?](#what-is-mcp)
  - [MCP Architecture: Server/Client Model](#mcp-architecture-serverclient-model)
  - [MCP Primitives: Resources, Tools, Prompts](#mcp-primitives-resources-tools-prompts)
  - [Building an MCP Server in Python](#building-an-mcp-server-in-python)
  - [Project: MCP Server for Databricks](#project-mcp-server-for-databricks)
- [Resources](#resources)
- [Projects](#projects)
- [Common Mistakes](#common-mistakes)
- [Mastery Checklist](#mastery-checklist)

---

## Phase Overview

| Attribute | Details |
|-----------|---------|
| **Duration** | Months 13–14 (4 weeks) |
| **Daily Time** | 1.5–2 hours |
| **Difficulty** | 8/10 |
| **Label** | 🔴 Must Learn |
| **Prerequisites** | Phase 5a (LLMs, prompting), Phase 5b (RAG basics) |
| **Outcome** | Can build multi-step agents with tools, planning, memory; can implement MCP servers |

---

## Part A: The Agent Mental Model

### What Is an Agent?

```
Traditional software:  deterministic, hand-coded logic
  input → [fixed decision tree] → output

LLM-powered agent:     LLM decides what to do based on context
  input → [LLM reasons] → [LLM decides to use tool] → [tool result] → [LLM reasons again] → output
```

An agent is a system where an LLM:
1. **Perceives** its environment (text, tool results, memory)
2. **Reasons** about what to do next
3. **Acts** by calling tools or producing output
4. **Observes** the results
5. **Repeats** until the goal is achieved

The critical difference from RAG: agents are **active** and **multi-step**. A RAG system does one retrieval. An agent can do research, write code, execute it, fix errors, and iterate — all autonomously.

---

### The Agent Loop: Reason → Act → Observe

```
┌─────────────────────────────────────────────┐
│                 AGENT LOOP                  │
│                                             │
│  ┌─────────┐                                │
│  │  INPUT  │ ← User request / task          │
│  └────┬────┘                                │
│       ↓                                     │
│  ┌─────────┐                                │
│  │ REASON  │ ← LLM thinks about what to do  │
│  └────┬────┘                                │
│       ↓                                     │
│  ┌─────────┐   ┌──────────────────────┐     │
│  │   ACT   ├──→│  TOOLS               │     │
│  └────┬────┘   │  - search()          │     │
│       │        │  - code_executor()   │     │
│       │        │  - database_query()  │     │
│       │        │  - web_browser()     │     │
│       │        │  - file_reader()     │     │
│       │        └──────────┬───────────┘     │
│       ↓                   ↓                 │
│  ┌─────────┐   ┌──────────────────────┐     │
│  │ OBSERVE │ ← │  Tool Result         │     │
│  └────┬────┘   └──────────────────────┘     │
│       │                                     │
│       ↓ (continue loop or terminate)        │
│  ┌─────────┐                                │
│  │  DONE?  │                                │
│  └─────────┘                                │
└─────────────────────────────────────────────┘
```

---

### ReAct: Synergizing Reasoning and Acting

ReAct (Yao et al., 2022) is the foundational agent pattern. It interleaves reasoning traces with actions.

```python
from openai import OpenAI
import json
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ReAct prompt format:
# Thought → Action → Observation → Thought → Action → ... → Final Answer

REACT_SYSTEM = """You are a helpful research assistant. 
Solve tasks using this pattern:

Thought: Reason about what to do
Action: tool_name({"param": "value"})
Observation: [tool result will appear here]
... (repeat as needed)
Final Answer: [your answer to the user]

Available tools:
- search(query): Search the web for information
- calculate(expression): Evaluate a mathematical expression
- get_time(): Get the current time

IMPORTANT: Always start each step with "Thought:", "Action:", or "Final Answer:"
"""

def parse_action(text: str) -> tuple[str, dict] | None:
    """Parse 'Action: tool_name({"key": "value"})' format."""
    if "Action:" not in text:
        return None
    
    action_text = text.split("Action:")[-1].strip().split("\n")[0]
    
    # Extract tool name and arguments
    tool_name = action_text.split("(")[0].strip()
    args_str = action_text[action_text.find("(")+1:action_text.rfind(")")]
    
    try:
        args = json.loads(args_str) if args_str else {}
    except json.JSONDecodeError:
        args = {"query": args_str.strip('"')}
    
    return tool_name, args

# Mock tool implementations
def search(query: str) -> str:
    return f"[Search results for '{query}': Found relevant information about {query}...]"

def calculate(expression: str) -> str:
    # SECURITY NOTE: Never use eval() for untrusted LLM output — prompt injection can
    # craft expressions that escape the restricted namespace. Use a safe math library instead.
    # pip install simpleeval
    try:
        from simpleeval import simple_eval
        return str(simple_eval(expression))
    except Exception as e:
        return f"Invalid expression: {e}"

def get_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

TOOLS = {"search": search, "calculate": calculate, "get_time": get_time}

def run_react_agent(user_query: str, max_steps: int = 10) -> str:
    """Run a ReAct agent until it reaches a Final Answer or max steps."""
    messages = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user", "content": user_query}
    ]
    
    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stop=["Observation:"]  # Stop before observation (we'll fill this in)
        ).choices[0].message.content
        
        print(f"\n--- Step {step + 1} ---")
        print(response)
        
        messages.append({"role": "assistant", "content": response})
        
        # Check if done
        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip()
        
        # Parse and execute action
        parsed = parse_action(response)
        if parsed:
            tool_name, args = parsed
            if tool_name in TOOLS:
                result = TOOLS[tool_name](**args)
            else:
                result = f"Error: Unknown tool '{tool_name}'"
            
            observation = f"Observation: {result}"
            print(observation)
            messages.append({"role": "user", "content": observation})
    
    return "Max steps reached without finding an answer."
```

---

### Agent Memory Types

```python
# Memory is what makes agents effective over long tasks
memory_types = {
    "In-Context": {
        "description": "Conversation history in the context window",
        "capacity": "Limited by context length (4K-200K tokens)",
        "persistence": "Only within the current session",
        "retrieval": "Always present (no retrieval needed)",
        "use_case": "Short conversations, single-session tasks",
    },
    "External (RAG)": {
        "description": "Vector database with semantic retrieval",
        "capacity": "Millions of documents",
        "persistence": "Permanent across sessions",
        "retrieval": "Semantic search at each step",
        "use_case": "Knowledge base, long-term facts",
    },
    "Working Memory": {
        "description": "Structured state (dict/DB) for current task",
        "capacity": "Limited by state size",
        "persistence": "Duration of task execution",
        "retrieval": "Direct key lookup",
        "use_case": "Task state, intermediate results",
    },
    "Episodic (Summarized)": {
        "description": "Compressed summaries of past conversations",
        "capacity": "Unlimited (compressed)",
        "persistence": "Permanent",
        "retrieval": "Semantic search by episode",
        "use_case": "Long-running agents, user profiles",
    },
}

class AgentMemory:
    """Combined memory system for production agents."""
    
    def __init__(self, max_in_context: int = 20):
        self.conversation_history: list[dict] = []  # In-context memory
        self.working_memory: dict = {}              # Current task state
        self.max_in_context = max_in_context
    
    def add_message(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})
        
        # Compress if too long
        if len(self.conversation_history) > self.max_in_context:
            self._compress_history()
    
    def _compress_history(self) -> None:
        """Summarize old messages to save context space."""
        # Keep last N messages; summarize the rest
        keep = self.max_in_context // 2
        to_compress = self.conversation_history[:-keep]
        
        # In production: use LLM to summarize to_compress
        summary = f"[Summary of {len(to_compress)} earlier messages: user asked about various topics]"
        
        self.conversation_history = [
            {"role": "system", "content": summary}
        ] + self.conversation_history[-keep:]
    
    def set_working_memory(self, key: str, value) -> None:
        self.working_memory[key] = value
    
    def get_working_memory(self, key: str):
        return self.working_memory.get(key)
    
    def get_context(self) -> list[dict]:
        return self.conversation_history
```

---

## Part B: Tool Use

### Function Calling (OpenAI)

```python
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define tools in OpenAI's schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Execute a SQL query on the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Mock tool implementations
def get_weather(location: str, unit: str = "celsius") -> dict:
    return {"location": location, "temperature": 22, "unit": unit, "condition": "sunny"}

def execute_sql(query: str) -> dict:
    # In production: actually execute SQL
    return {"result": [{"name": "John", "age": 30}], "rows": 1}

TOOL_FUNCTIONS = {"get_weather": get_weather, "execute_sql": execute_sql}

def run_agent_with_tools(user_message: str) -> str:
    """Agent loop with OpenAI function calling."""
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"  # "auto" lets model decide; "required" forces tool use
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        # Check if model wants to call tools
        if not message.tool_calls:
            # No more tool calls — final response
            return message.content
        
        # Execute each tool call
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            print(f"Calling {function_name}({arguments})")
            
            if function_name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[function_name](**arguments)
            else:
                result = {"error": f"Unknown function: {function_name}"}
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

# Test
answer = run_agent_with_tools("What's the weather in Paris? Is it warmer than 25 Celsius?")
print(answer)
```

---

### Building Custom Tools

```python
from typing import Callable, Any
import inspect
import ast

def tool(description: str = ""):
    """
    Decorator to turn a Python function into an agent tool.
    Automatically generates the OpenAI tool schema from the function signature.
    """
    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        type_map = {
            str: "string", int: "integer", float: "number",
            bool: "boolean", list: "array", dict: "object"
        }
        
        properties = {}
        required = []
        
        for name, param in sig.parameters.items():
            annotation = param.annotation
            param_type = type_map.get(annotation, "string")
            
            # Get docstring parameter description
            doc = inspect.getdoc(func) or ""
            
            properties[name] = {"type": param_type, "description": f"Parameter: {name}"}
            
            if param.default == inspect.Parameter.empty:
                required.append(name)
        
        func._tool_schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description or inspect.getdoc(func) or func.__name__,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        return func
    return decorator


# Usage:
@tool("Search the web for current information")
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return relevant results."""
    # In production: call a real search API (Tavily, SerpAPI, Bing)
    return f"Search results for '{query}': [relevant information]"

@tool("Execute Python code and return the output")
def python_repl(code: str) -> str:
    """Execute Python code in a sandboxed environment."""
    # SECURITY: Never exec user code in production without sandboxing
    # Use: e2b.dev (sandboxed code execution), or Docker containers
    import io
    import contextlib
    
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": {"print": print, "range": range, "len": len}})
        return output.getvalue() or "Code executed successfully (no output)"
    except Exception as e:
        return f"Error: {e}"

@tool("Query a database with SQL")
def database_query(sql: str, database: str = "default") -> str:
    """Execute a read-only SQL query on the specified database."""
    # Validate: only SELECT statements
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed"
    
    # In production: connect to actual database
    return f"Query result: [simulated results for: {sql[:50]}...]"
```

---

## Part C: Planning Patterns

### Plan-and-Execute Pattern

```python
# For complex multi-step tasks, planning upfront is better than pure ReAct
# Plan-and-Execute:
#   1. PLAN: LLM generates a complete plan (list of steps)
#   2. EXECUTE: Execute each step with tools
#   3. REPLAN: If a step fails, replan from the current state

def plan_and_execute(task: str) -> str:
    """Plan first, then execute step by step."""
    
    # Step 1: Generate plan
    plan_response = client.chat.completions.create(
        model="gpt-4o",  # Use a strong model for planning
        messages=[{
            "role": "user",
            "content": f"""Create a step-by-step plan to complete this task:

Task: {task}

Output a numbered list of concrete steps. Each step should be actionable.
Consider what information you need and in what order to get it."""
        }]
    ).choices[0].message.content
    
    steps = [line.strip() for line in plan_response.split("\n") if line.strip() and line[0].isdigit()]
    print(f"Plan:\n{plan_response}\n")
    
    # Step 2: Execute each step
    results = []
    for i, step in enumerate(steps):
        print(f"\n--- Executing Step {i+1}: {step} ---")
        
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Execute this step precisely."},
                {"role": "user", "content": f"Task: {task}\n\nExecute this step: {step}\n\nPrevious results: {results}"}
            ],
            tools=tools
        ).choices[0].message.content
        
        results.append({"step": step, "result": result})
        print(f"Result: {result}")
    
    # Step 3: Synthesize final answer
    final = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"Task: {task}\n\nSteps executed:\n{json.dumps(results, indent=2)}\n\nProvide a complete, coherent final answer."
        }]
    ).choices[0].message.content
    
    return final
```

---

## Part D: LangGraph for Stateful Agents

### Why Stateful Agents Need Graphs

Simple agent loops work for straightforward tasks. Complex agents need:
- **Branching**: different paths based on conditions
- **Cycles**: retry loops, human-in-the-loop
- **Parallel execution**: run multiple tools simultaneously
- **State persistence**: checkpoint and resume
- **Streaming**: observe intermediate steps

LangGraph provides all of this with a graph-based abstraction.

---

### Core LangGraph Concepts

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Annotated
import operator

# 1. Define State — what data flows through the graph
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]    # Conversation history (append-only)
    task: str                                   # Original task
    plan: list[str]                             # Generated plan steps
    current_step: int                           # Which step we're on
    results: dict                               # Intermediate results
    final_answer: str                           # Final output

# 2. Define Nodes — each node is a function that takes State and returns State update
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def should_continue(state: AgentState) -> str:
    """Conditional edge: decide whether to continue or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"  # Go to tool execution node
    return END          # Finish

def call_model(state: AgentState) -> dict:
    """Node: call the LLM."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 3. Build the graph
def build_research_agent():
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools=[web_search, python_repl]))
    
    # Add edges
    graph.set_entry_point("agent")
    
    # Conditional edge: after agent runs, check if we need tools
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    
    # After tools: always go back to agent
    graph.add_edge("tools", "agent")
    
    return graph.compile()

# 4. Run the graph
agent_graph = build_research_agent()

config = {"recursion_limit": 25}  # Max iterations before stopping
result = agent_graph.invoke(
    {
        "messages": [HumanMessage(content="Research the latest advances in RAG and write a summary.")],
        "task": "Research RAG advances",
        "plan": [],
        "current_step": 0,
        "results": {},
        "final_answer": ""
    },
    config=config
)

print(result["messages"][-1].content)
```

### Building a Research Agent with LangGraph

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver  # Persistence
from typing import TypedDict, Annotated
import operator

class ResearchState(TypedDict):
    query: str
    search_results: list[str]
    analysis: str
    final_report: str
    messages: Annotated[list, operator.add]

def search_node(state: ResearchState) -> dict:
    """Search for information."""
    results = web_search(state["query"])
    return {
        "search_results": [results],
        "messages": [AIMessage(content=f"Found: {results[:200]}...")]
    }

def analyze_node(state: ResearchState) -> dict:
    """Analyze the search results."""
    context = "\n".join(state["search_results"])
    analysis = llm.invoke(
        f"Analyze these search results and extract key insights:\n{context}"
    ).content
    return {"analysis": analysis}

def write_report_node(state: ResearchState) -> dict:
    """Write a final report."""
    report = llm.invoke(
        f"Write a comprehensive report based on this analysis:\n{state['analysis']}"
    ).content
    return {"final_report": report}

def needs_more_search(state: ResearchState) -> str:
    """Decide if more searching is needed."""
    if len(state["search_results"]) < 3:
        return "search"   # Search more
    return "write"        # Enough data, write report

# Build the research agent graph
research_graph = StateGraph(ResearchState)
research_graph.add_node("search", search_node)
research_graph.add_node("analyze", analyze_node)
research_graph.add_node("write", write_report_node)

research_graph.set_entry_point("search")
research_graph.add_edge("search", "analyze")
research_graph.add_conditional_edges(
    "analyze",
    needs_more_search,
    {"search": "search", "write": "write"}
)
research_graph.add_edge("write", END)

# Add persistence (checkpointing)
memory = SqliteSaver.from_conn_string(":memory:")
research_app = research_graph.compile(checkpointer=memory)
```

---

## Part E: Model Context Protocol (MCP)

### What Is MCP?

MCP (Model Context Protocol) is an open standard by Anthropic that defines how AI applications connect to data sources and tools. It solves the "N×M integration problem":

```
Without MCP: N AI apps × M data sources = N×M custom integrations

Claude ←→ GitHub (custom)
Claude ←→ Slack (custom)
Claude ←→ Databases (custom)
GPT   ←→ GitHub (different custom)
...

With MCP: N + M (standardized protocol)

Claude ──→ MCP Protocol ←── GitHub MCP Server
GPT    ──→ MCP Protocol ←── Slack MCP Server
Any AI ──→ MCP Protocol ←── Any MCP Server
```

---

### MCP Architecture: Server/Client Model

```
┌─────────────────────────────────────────────────────┐
│                  HOST APPLICATION                    │
│  (Claude Desktop, Cursor, your custom agent)         │
│                                                      │
│  ┌──────────────┐        ┌──────────────────────┐   │
│  │  MCP CLIENT  │ ←────→ │  CONVERSATION/AGENT  │   │
│  └──────┬───────┘        └──────────────────────┘   │
└─────────┼───────────────────────────────────────────┘
          │ MCP Protocol (JSON-RPC 2.0 over stdio/SSE)
          │
┌─────────┴───────────────────────────────────────────┐
│                  MCP SERVER                          │
│  (your Python/Node.js server)                        │
│                                                      │
│  Resources   │  Tools         │  Prompts            │
│  - Files     │  - Functions   │  - Templates        │
│  - DB tables │  - API calls   │  - Instructions     │
│  - API data  │  - Computations│                     │
└─────────────────────────────────────────────────────┘
```

---

### MCP Primitives: Resources, Tools, Prompts

| Primitive | What It Is | Example |
|-----------|-----------|---------|
| **Resources** | Data the LLM can read | File contents, database rows, API responses |
| **Tools** | Functions the LLM can call | `run_query()`, `create_issue()`, `send_email()` |
| **Prompts** | Reusable prompt templates | `code_review_prompt`, `summarize_document` |

---

### Building an MCP Server in Python

```python
# pip install mcp

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import asyncio
import json
from datetime import datetime

# Create MCP server
server = Server("my-data-server")

# ── Resources ──
@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """List all available resources (like listing files or tables)."""
    return [
        types.Resource(
            uri="db://table/users",
            name="Users Table",
            description="User records from the database",
            mimeType="application/json"
        ),
        types.Resource(
            uri="file://reports/latest.md",
            name="Latest Report",
            description="The most recent analysis report",
            mimeType="text/markdown"
        )
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    """Return the content of a resource."""
    if uri == "db://table/users":
        # In production: actually query your database
        users = [
            {"id": 1, "name": "Alice", "role": "engineer"},
            {"id": 2, "name": "Bob", "role": "analyst"}
        ]
        return json.dumps(users, indent=2)
    
    elif uri == "file://reports/latest.md":
        return "# Latest Report\n\nKey findings: [...]"
    
    raise ValueError(f"Unknown resource: {uri}")

# ── Tools ──
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all available tools."""
    return [
        types.Tool(
            name="execute_query",
            description="Execute a read-only SQL query on the database",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT query to execute"
                    },
                    "database": {
                        "type": "string",
                        "description": "Database name",
                        "default": "default"
                    }
                },
                "required": ["sql"]
            }
        ),
        types.Tool(
            name="get_metrics",
            description="Get system metrics for a given time range",
            inputSchema={
                "type": "object",
                "properties": {
                    "metric_name": {"type": "string"},
                    "start_time": {"type": "string", "description": "ISO datetime"},
                    "end_time": {"type": "string", "description": "ISO datetime"}
                },
                "required": ["metric_name"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute a tool call."""
    
    if name == "execute_query":
        sql = arguments["sql"]
        
        # Security: only allow SELECT
        if not sql.strip().upper().startswith("SELECT"):
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "Only SELECT queries allowed"})
            )]
        
        # In production: execute actual SQL
        result = {"rows": [{"col1": "value1"}], "count": 1}
        return [types.TextContent(type="text", text=json.dumps(result))]
    
    elif name == "get_metrics":
        metric = arguments["metric_name"]
        result = {"metric": metric, "value": 42.5, "timestamp": datetime.now().isoformat()}
        return [types.TextContent(type="text", text=json.dumps(result))]
    
    raise ValueError(f"Unknown tool: {name}")

# ── Prompts ──
@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    """List reusable prompt templates."""
    return [
        types.Prompt(
            name="data_analysis",
            description="Analyze a dataset and provide insights",
            arguments=[
                types.PromptArgument(name="table_name", description="Table to analyze", required=True),
                types.PromptArgument(name="focus_area", description="What aspect to focus on", required=False)
            ]
        )
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
    """Return a filled-in prompt template."""
    if name == "data_analysis":
        table = arguments.get("table_name", "unknown")
        focus = arguments.get("focus_area", "general trends")
        
        return types.GetPromptResult(
            description=f"Analysis prompt for {table}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"""Please analyze the '{table}' table with a focus on {focus}.

First, examine the data structure by reading the resource.
Then provide:
1. Summary statistics
2. Key patterns and trends
3. Anomalies or interesting findings
4. Actionable recommendations"""
                    )
                )
            ]
        )
    
    raise ValueError(f"Unknown prompt: {name}")

# Run the server
async def main():
    async with stdio_server() as streams:
        await server.run(*streams, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Project: MCP Server for Databricks

This project is uniquely valuable for your background as a Data Engineer.

```python
# databricks_mcp_server.py
# An MCP server that exposes Databricks capabilities to Claude

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from databricks import sql as databricks_sql
from databricks.sdk import WorkspaceClient
import asyncio
import json
import os

server = Server("databricks-mcp")
w = WorkspaceClient(
    host=os.getenv("DATABRICKS_HOST"),
    token=os.getenv("DATABRICKS_TOKEN")
)

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_tables",
            description="List all tables in a Databricks catalog/schema",
            inputSchema={
                "type": "object",
                "properties": {
                    "catalog": {"type": "string", "description": "Catalog name"},
                    "schema": {"type": "string", "description": "Schema name"}
                },
                "required": []
            }
        ),
        types.Tool(
            name="run_sql",
            description="Execute a SQL query on Databricks (read-only)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query"},
                    "warehouse_id": {"type": "string", "description": "SQL warehouse ID"}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="get_job_status",
            description="Get the status of a Databricks job",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer", "description": "Job ID"}
                },
                "required": ["job_id"]
            }
        ),
        types.Tool(
            name="list_clusters",
            description="List active Databricks clusters",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    
    if name == "list_tables":
        catalog = arguments.get("catalog", "main")
        schema = arguments.get("schema", "default")
        
        # Use Databricks SDK to list tables
        tables = w.tables.list(catalog_name=catalog, schema_name=schema)
        table_list = [{"name": t.name, "type": t.table_type.value if t.table_type else "UNKNOWN"} 
                     for t in tables]
        return [types.TextContent(type="text", text=json.dumps(table_list, indent=2))]
    
    elif name == "run_sql":
        query = arguments["query"]
        
        # Security: only allow read operations
        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
        if any(kw in query.upper() for kw in forbidden):
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "Only SELECT/DESCRIBE/SHOW queries allowed"})
            )]
        
        warehouse_id = arguments.get("warehouse_id", os.getenv("DATABRICKS_WAREHOUSE_ID"))
        
        with databricks_sql.connect(
            server_hostname=os.getenv("DATABRICKS_HOST"),
            http_path=f"/sql/1.0/warehouses/{warehouse_id}",
            access_token=os.getenv("DATABRICKS_TOKEN")
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchmany(100)  # Limit to 100 rows
                result = [dict(zip(columns, row)) for row in rows]
        
        return [types.TextContent(type="text", text=json.dumps(result[:20], indent=2, default=str))]
    
    elif name == "get_job_status":
        job_id = arguments["job_id"]
        runs = w.jobs.list_runs(job_id=job_id, active_only=False, limit=1)
        latest = next(iter(runs), None)
        
        if not latest:
            return [types.TextContent(type="text", text=json.dumps({"error": "No runs found"}))]
        
        result = {
            "job_id": job_id,
            "run_id": latest.run_id,
            "state": latest.state.life_cycle_state.value if latest.state else "UNKNOWN",
            "result": latest.state.result_state.value if latest.state and latest.state.result_state else None,
            "start_time": latest.start_time,
        }
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "list_clusters":
        clusters = w.clusters.list()
        cluster_list = [
            {"id": c.cluster_id, "name": c.cluster_name, "state": c.state.value if c.state else "UNKNOWN"}
            for c in clusters
        ]
        return [types.TextContent(type="text", text=json.dumps(cluster_list, indent=2))]

# Run
async def main():
    async with stdio_server() as streams:
        await server.run(*streams, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())

# claude_desktop_config.json entry:
config_example = """
{
  "mcpServers": {
    "databricks": {
      "command": "python",
      "args": ["C:/path/to/databricks_mcp_server.py"],
      "env": {
        "DATABRICKS_HOST": "https://your-workspace.azuredatabricks.net",
        "DATABRICKS_TOKEN": "your-token",
        "DATABRICKS_WAREHOUSE_ID": "your-warehouse-id"
      }
    }
  }
}
"""
```

---

## Part F: Multi-Agent Systems

### When a Single Agent Is Not Enough

A single ReAct agent degrades when:
1. **Context bloat**: long tool histories exceed the context window
2. **Expertise dilution**: one prompt cannot simultaneously be an expert researcher, a code writer, and a data analyst
3. **Parallelism**: independent sub-tasks are forced to run serially
4. **Fault isolation**: one failed sub-task should not abort the entire workflow

Multi-agent architectures solve these by decomposing tasks into specialized, coordinated workers.

> **Rule of thumb**: Build with one agent first. Add a second agent only when you can precisely name what the first agent cannot do that a second agent would solve. Most systems that appear to need multi-agent actually need better prompts.

### The Supervisor-Worker Pattern

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Literal
import operator

# ── Define shared state ──
class AgentState(TypedDict):
    task: str
    plan: str
    completed_steps: List[str]     # accumulated by each worker
    final_answer: str
    next: str                       # which worker to call next

# ── Specialized workers ──
research_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
code_llm = ChatOpenAI(model="gpt-4o", temperature=0)

research_agent = create_react_agent(
    research_llm,
    tools=[search_tool, web_scrape_tool],
    state_modifier="You are a research specialist. Find accurate, sourced information."
)

code_agent = create_react_agent(
    code_llm,
    tools=[code_executor_tool, test_runner_tool],
    state_modifier="You are a code specialist. Write correct, tested code."
)

# ── Supervisor decides routing ──
SUPERVISOR_PROMPT = """You are a task supervisor managing specialist agents.

TASK: {task}
COMPLETED STEPS:
{completed}

Based on what has been done, who should act next?
- 'research' — if more information is needed
- 'code' — if implementation or computation is needed  
- 'FINISH' — if the task is complete

Respond with ONLY one of: research, code, FINISH"""

supervisor_llm = ChatOpenAI(model="gpt-4o-mini")

def supervisor(state: AgentState) -> dict:
    completed_text = "\n".join(state.get("completed_steps", [])) or "None yet"
    response = supervisor_llm.invoke(
        SUPERVISOR_PROMPT.format(task=state["task"], completed=completed_text)
    )
    return {"next": response.content.strip()}

def research_node(state: AgentState) -> dict:
    result = research_agent.invoke({"messages": [("user", state["task"])]})
    last_message = result["messages"][-1].content
    return {"completed_steps": [f"[research] {last_message}"]}

def code_node(state: AgentState) -> dict:
    context = "\n".join(state.get("completed_steps", []))
    result = code_agent.invoke({"messages": [("user", f"Context:\n{context}\n\nTask: {state['task']}")]})
    last_message = result["messages"][-1].content
    return {"completed_steps": [f"[code] {last_message}"]}

# ── Build the graph ──
graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor)
graph.add_node("research", research_node)
graph.add_node("code", code_node)

# Supervisor routes to worker or terminates
graph.add_conditional_edges(
    "supervisor",
    lambda s: s["next"],
    {"research": "research", "code": "code", "FINISH": END}
)
# Workers always return to supervisor
graph.add_edge("research", "supervisor")
graph.add_edge("code", "supervisor")
graph.set_entry_point("supervisor")

app = graph.compile()

# Run it
result = app.invoke({
    "task": "Research the latest LLaMA-3 benchmark results and write a Python script that plots them",
    "completed_steps": []
})
```

### Loop Detection and Termination

Agents can loop. Always add an iteration guard:

```python
class AgentState(TypedDict):
    task: str
    completed_steps: List[str]
    iteration_count: int            # increment each step

def supervisor(state: AgentState) -> dict:
    # Hard cap: never exceed 15 steps
    if state.get("iteration_count", 0) >= 15:
        return {"next": "FINISH"}
    
    # Detect repeated identical actions (last 3 steps all the same)
    recent = state.get("completed_steps", [])[-3:]
    if len(recent) == 3 and len(set(recent)) == 1:
        return {"next": "FINISH"}   # Stuck in a loop
    
    # Normal routing...
    return {"next": route_to_worker(state), "iteration_count": state.get("iteration_count", 0) + 1}
```

### Agent Handoff: What Information to Pass

When workers hand off to each other, include three things:

```python
HANDOFF_TEMPLATE = """
ORIGINAL TASK:
{original_task}

ALREADY COMPLETED (do not repeat this work):
{completed_steps}

YOUR SPECIFIC SUBTASK:
{specific_subtask}

EXPECTED OUTPUT FORMAT:
{output_format}
"""
# This prevents workers from re-doing work and losing context about the goal.
```

### Parallel Agent Execution

For independent sub-tasks, run agents in parallel:

```python
import asyncio

async def run_parallel_research(queries: List[str]) -> List[str]:
    """Run multiple research agents simultaneously."""
    tasks = [
        research_agent.ainvoke({"messages": [("user", q)]})
        for q in queries
    ]
    results = await asyncio.gather(*tasks)
    return [r["messages"][-1].content for r in results]

# Gather then synthesize:
# 1. Run N agents in parallel (2-5× faster than sequential)
# 2. Combine their outputs in a single synthesis step
synthesis_prompt = """Synthesize these research findings into a coherent summary:
{findings}"""
```

### Common Mistakes in Multi-Agent Systems

| Mistake | Consequence | Fix |
|---------|------------|-----|
| No iteration cap | Infinite loop, unbounded cost | Always add `iteration_count >= N` guard |
| Workers don't see previous steps | Repeated work, conflicting outputs | Pass `completed_steps` to every worker |
| Too many agents for simple tasks | Debugging nightmare | Start single-agent; add agents only for clear reasons |
| No structured handoff format | Workers misunderstand context | Use the HANDOFF_TEMPLATE pattern |
| Agents share mutable state | Race conditions | Use immutable state; accumulate with `operator.add` |

---

## Resources

| Rank | Resource | Type | Cost | Why |
|------|----------|------|------|-----|
| 1 | [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/) | Docs | Free | Best intro to stateful agents. Build a working agent in 30 min. |
| 2 | [MCP Documentation (Anthropic)](https://modelcontextprotocol.io/docs/) | Docs | Free | Official MCP spec and Python SDK tutorial |
| 3 | [ReAct Paper (Yao et al., 2022)](https://arxiv.org/abs/2210.03629) | Paper | Free | Foundational agent paper. 8 pages. Read it. |
| 4 | [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) | Docs | Free | How to implement tool use with OpenAI API |
| 5 | [Building LLM Agents (Lilian Weng)](https://lilianweng.github.io/posts/2023-06-23-agent/) | Blog | Free | Best overview of agent architectures |

---

## Projects

### Project 16: Personal Research Agent
**Difficulty**: 7/10 | **Time**: 2 weeks

Build a research agent that:
1. Accepts a research question
2. Searches the web with Tavily or SerpAPI
3. Extracts key information from 5-10 sources
4. Synthesizes a structured report
5. Cites sources with links
6. Includes a confidence score for each claim

Implemented with LangGraph (with proper state and checkpointing).

---

### Project 17: Databricks MCP Server
**Difficulty**: 7/10 | **Time**: 1 week | **Unique to your background**

Implement the Databricks MCP server above, extended with:
- List notebooks in workspace
- Get cluster metrics
- Read Delta table schema and sample data
- Run a Databricks job
- Query job run history

Test by connecting to Claude Desktop and asking natural language questions about your Databricks workspace.

---

### Project 17b: Text-to-SQL Agent
**Difficulty**: 6/10 | **Time**: 1 week | **Perfect fit for your SQL background**

This is one of the highest-value AI use cases for data engineers.

**Build a Text-to-SQL agent that**:
- Accepts natural language questions about a database
- Generates SQL using the table schema as context
- Executes the query on a read-only SQLite/DuckDB connection
- If query fails: shows the error to the LLM and asks it to fix the query (error correction loop)
- Formats and explains the result in plain English

```python
import sqlite3
from openai import OpenAI

client = OpenAI()
conn = sqlite3.connect("company.db", check_same_thread=False)
conn.execute("PRAGMA query_only = ON")   # Read-only safety

def get_schema() -> str:
    """Extract full schema for the LLM context."""
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    schema_parts = []
    for (table,) in tables:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        schema_parts.append(f"CREATE TABLE {table} ({col_defs});")
    return "\n".join(schema_parts)

def text_to_sql_agent(question: str, max_retries: int = 3) -> str:
    """LLM agent that generates and self-corrects SQL."""
    schema = get_schema()
    messages = [
        {"role": "system", "content": f"You are a SQL expert. Database schema:\n{schema}\n\nGenerate valid SQLite SQL to answer the user's question. Output ONLY the SQL query, no explanation."},
        {"role": "user", "content": question}
    ]
    
    for attempt in range(max_retries):
        sql = client.chat.completions.create(model="gpt-4o-mini", messages=messages).choices[0].message.content.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        try:
            rows = conn.execute(sql).fetchmany(20)
            col_names = [d[0] for d in conn.execute(sql).description]
            result_table = tabulate(rows, headers=col_names)
            
            # Ask LLM to explain the result
            explanation = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Question: {question}\nSQL: {sql}\nResults:\n{result_table}\n\nExplain the results in plain English."}]
            ).choices[0].message.content
            return f"SQL: {sql}\n\nResult:\n{result_table}\n\nExplanation: {explanation}"
        
        except sqlite3.Error as e:
            # Error correction: give the LLM the error and ask it to fix
            messages.append({"role": "assistant", "content": sql})
            messages.append({"role": "user", "content": f"That query failed with: {e}\nPlease fix the SQL query."})
    
    return f"Could not generate valid SQL after {max_retries} attempts."
```

**Extend with**:
- Schema-aware embeddings for large databases (too many tables for the context window)
- Query result caching for common questions
- Support for Databricks SQL endpoints (your production leverage)
- Spider/BIRD benchmark evaluation

---

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| No exit condition in agent loop | Infinite loop, wasted tokens | Always set max_steps and validate termination |
| Trusting agent output without validation | Security risks, incorrect results | Always validate tool call arguments and outputs |
| Giving agent write access from the start | Accidental data modification | Start read-only; add write tools carefully with confirmation |
| Not handling tool failures | Agent gets stuck | Add try/except in all tools; return descriptive error messages |
| Using GPT-3.5 for complex planning | Poor quality plans | Use GPT-4o or Claude 3.5 for planning; smaller model for execution |
| No state persistence | Can't resume interrupted tasks | Always use LangGraph checkpointing for long tasks |

---

## Mastery Checklist

### Agents
- [ ] Can explain the agent loop (reason → act → observe)
- [ ] Implemented ReAct agent from scratch without frameworks
- [ ] Understands the four memory types and when to use each
- [ ] Can build a multi-step tool-use agent with OpenAI function calling

### LangGraph
- [ ] Understands nodes, edges, and state in LangGraph
- [ ] Built at least one agent graph with conditional edges
- [ ] Implemented persistence/checkpointing

### MCP
- [ ] Can explain what problem MCP solves
- [ ] Understands resources, tools, and prompts primitives
- [ ] Built a working MCP server with at least 3 tools
- [ ] Connected MCP server to Claude Desktop or another host

### Projects
- [ ] Project 16 (Research Agent) complete and on GitHub
- [ ] Project 17 (Databricks MCP Server) working with your Databricks workspace

---

## Moving to Phase 7

**Before proceeding to [Phase 7a: Fine-Tuning & LoRA](./08_Phase7_Part1_FineTuning_and_LoRA.md), confirm:**

- [ ] Working research agent with LangGraph
- [ ] Databricks MCP server connected and tested
- [ ] Can explain when to use agents vs. simple RAG

**Why Phase 7 comes next**: You now know how to use LLMs effectively (RAG + agents). Phase 7 teaches you how to adapt LLMs themselves — training them on your own data for specific tasks.

---

## Phase Completion & Readiness Assessment

> Complete this assessment before Phase 7. Agents and MCP are among the most valuable skills in 2026 AI engineering — and your Databricks background gives you a unique edge here.

---

### 1. Knowledge Checklist

**Agent Fundamentals**
- [ ] What distinguishes an agent from a pipeline: perception, planning, action, memory
- [ ] ReAct pattern: Thought → Action → Observation loop
- [ ] Tool use: function calling, JSON schema, argument validation
- [ ] Plan-and-Execute: planning phase vs. execution phase
- [ ] Agent failure modes: loops, hallucinated actions, incomplete tasks

**Memory Types**
- [ ] In-context memory: what's in the current prompt
- [ ] External memory: vector DB lookups during agent execution
- [ ] Working memory: structured scratchpad for intermediate state
- [ ] Procedural memory: fine-tuned into weights (Phase 7)
- [ ] Episodic memory: stored conversation histories

**LangGraph**
- [ ] StateGraph: nodes, edges, conditional edges, state schema
- [ ] Checkpointing: SQLite and Redis persistence
- [ ] Human-in-the-loop: `interrupt_before`, `update_state`
- [ ] Streaming: `graph.stream()` for real-time output
- [ ] Parallelism: running nodes in parallel (fan-out + fan-in)

**MCP (Model Context Protocol)**
- [ ] What problem MCP solves: N+M vs. N×M integrations
- [ ] MCP components: resources, tools, prompts
- [ ] MCP transport: stdio and HTTP+SSE
- [ ] Client-server architecture: LLM app (client) ↔ MCP server (tool provider)
- [ ] MCP server implementation: `list_tools`, `call_tool`, `list_resources`, `read_resource`

---

### 2. Practical Skills Checklist

- [ ] Implement a ReAct agent from scratch (without LangChain) with a 3-tool set
- [ ] Use OpenAI function calling to build a tool-use agent
- [ ] Build a tool decorator that auto-generates OpenAI schema from function signature + docstring
- [ ] Build a LangGraph agent with: memory, tools, conditional edges, and checkpointing
- [ ] Implement an agent memory class using a vector database
- [ ] Build an MCP server with at least 2 tools and 1 resource using the MCP Python SDK
- [ ] Connect your MCP server to Claude Desktop and verify it works
- [ ] Handle agent failures gracefully (retry logic, fallback paths)

---

### 3. Coding Challenges

**Challenge A — ReAct from Scratch**
```python
# Implement a ReAct agent with NO LangChain or LangGraph:
# Tools: search (mock Tavily), calculator (eval safe expressions), wikipedia_summary
# Loop:
#   1. LLM generates: Thought: ... Action: tool_name[arg]
#   2. Parse action with regex
#   3. Execute tool
#   4. Append Observation: result to context
#   5. Repeat until: Final Answer: ...
# Max iterations: 8
# Test on: 5 multi-step questions requiring at least 2 tool calls each
# Handle: tool errors, malformed actions, max iteration exceeded
```

**Challenge B — LangGraph Stateful Agent**
```python
# Build a research assistant agent with LangGraph:
# State: { messages, research_notes, sources, final_report }
# Nodes: plan_research, search_web, extract_info, synthesize, write_report
# Conditional edges: should_continue_research → search or synthesize
# Checkpointing: SQLite (ability to resume interrupted runs)
# Human-in-the-loop: pause before write_report for human approval
# Test: "Research the current state of LoRA fine-tuning for 7B models"
```

**Challenge C — Databricks MCP Server**
```python
# Build a complete MCP server for your Databricks workspace:
# Tools:
#   - list_tables(catalog, schema) → List[TableInfo]
#   - run_sql(query, timeout_seconds) → DataFrame as markdown
#   - get_job_status(job_id) → JobStatus
#   - list_clusters() → List[ClusterInfo]
#   - get_notebook(path) → notebook content
# Resources:
#   - databricks://schemas → all schemas as JSON
#   - databricks://tables/{schema} → tables in schema
# Prompts:
#   - data_exploration_prompt(table_name) → ready-to-use prompt
# Connect to Claude Desktop and demo: "What tables are in the NSR schema?"
```

---

### 4. Mini Project

**Research Agent** (Project 16 from the roadmap):
- LangGraph agent with: Tavily web search, URL reader, calculator, final report writer
- State maintains: research notes, sources visited, draft sections
- Persistence with SQLite (can resume if interrupted)
- Output: structured Markdown report with citations and confidence scores
- Test on 5 research questions in your domain (data engineering / AI)

---

### 5. Capstone Project

**Databricks MCP Server + Agent Integration** (Projects 16 + 17):
- MCP server: fully functional with tools (SQL, jobs, clusters, notebooks)
- LangGraph research agent: uses MCP tools to answer Databricks-related questions
- Example agent task: "Find the top 5 slowest Databricks jobs from the last 7 days, explain why they might be slow, and suggest optimisations"
- The agent: queries job history via MCP, reads notebook code via MCP, uses its ML knowledge to suggest fixes
- Output: detailed analysis report

---

### 6. Interview Questions

**Beginner**

1. **Q: What is an AI agent and how does it differ from a simple LLM call?**
   A: An agent is an LLM in a loop that can: perceive inputs, decide actions, use tools, and iterate until a goal is reached. A single LLM call is one-shot: input → output. An agent is iterative: it takes actions, observes results, and plans next steps until the task is complete.

2. **Q: What is the ReAct prompting pattern?**
   A: ReAct interleaves Thought (reasoning) and Action (tool call): "Thought: I need to find X. Action: search[X]. Observation: [result]. Thought: Now I can... Final Answer: [answer]." This makes agent reasoning transparent and debuggable.

3. **Q: What is function calling in LLM APIs?**
   A: Function calling allows you to pass a schema of available tools (name, description, parameters as JSON Schema). The LLM returns a structured JSON object with the tool name and arguments to call instead of free text. You execute the tool and pass the result back.

4. **Q: What is LangGraph?**
   A: LangGraph is a library for building stateful, multi-step AI workflows as directed graphs. Nodes are functions (LLM calls, tool calls), edges define control flow, and state is a typed dictionary passed between nodes. Supports persistence, streaming, and human-in-the-loop.

5. **Q: What is MCP (Model Context Protocol)?**
   A: MCP is an open protocol (Anthropic) for connecting AI assistants to external tools and data sources. MCP servers expose: resources (data), tools (actions), and prompts (templates). Clients (Claude, Claude Desktop, AI apps) connect to any MCP server — enabling N+M integrations instead of N×M.

6. **Q: What is the difference between agent memory types?**
   A: In-context: what's currently in the prompt. Working: temporary scratchpad. External: long-term vector store retrieved per query. Episodic: past conversation logs. Procedural: skills baked into model weights via fine-tuning. Agents typically use in-context + external + working memory.

7. **Q: What is a tool schema in OpenAI function calling?**
   A: A JSON Schema object describing: function name, description (used by LLM to decide when to call it), and parameters (with types, descriptions, required fields). Good descriptions are critical — the LLM reads them to decide which tool to use.

**Intermediate**

8. **Q: How does LangGraph checkpointing work?**
   A: LangGraph stores the entire agent state (messages, variables) to a persistent storage backend (SQLite, Redis, PostgreSQL) after each node execution. To resume: load the last checkpoint and the graph continues from where it stopped. Essential for: long-running tasks, human approval workflows, fault tolerance.

9. **Q: What are the failure modes of agents and how do you handle them?**
   A: (1) Infinite loops: max iterations + conditional exit. (2) Tool errors: try/except returning error descriptions to the LLM. (3) Hallucinated tool calls: validate against schema before execution. (4) Incomplete tasks: detect "stuck" state and escalate to human. (5) Context overflow: summarise old messages.

10. **Q: What is the Plan-and-Execute pattern and when is it better than ReAct?**
    A: Plan-and-Execute: first generate a full plan (list of steps), then execute each step. Better for: tasks where steps are known upfront, when parallel execution is possible. ReAct is better for: exploratory tasks where the next step depends on previous results. Plan-and-Execute is more efficient; ReAct is more adaptive.

11. **Q: Explain the N+M vs. N×M problem that MCP solves.**
    A: Without MCP: each AI application must implement custom integrations for each tool (N apps × M tools = N×M integrations). With MCP: each tool implements one MCP server, each app implements one MCP client (N apps + M tools = N+M implementations). Like USB for AI integrations.

12. **Q: What is a conditional edge in LangGraph and give an example?**
    A: A conditional edge uses a function to route to different nodes based on state. Example: after a tool call, check if the answer is complete. If yes → `END`. If no → `call_tools`. This enables dynamic branching: `graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})`.

13. **Q: How would you implement rate limiting for an agent that makes many LLM calls?**
    A: (1) Token bucket algorithm: track tokens used per minute. (2) asyncio.Semaphore to limit concurrent calls. (3) exponential backoff with jitter on RateLimitError. (4) Batch planning: generate multiple tool calls in one LLM call. (5) Cache repeated identical tool calls within a session.

**Advanced**

14. **Q: How do you evaluate agent performance?**
    A: (1) Task completion rate: does the agent complete the goal? (2) Tool call accuracy: does it call the right tools with right arguments? (3) Faithfulness: are its conclusions supported by tool outputs? (4) Efficiency: how many steps/tokens to complete? (5) Trajectory evaluation: LangSmith traces compared against expert-labeled trajectories.

15. **Q: What is the "lost control" safety problem in agents and how do you mitigate it?**
    A: An agent executing real-world actions (sending emails, running SQL, deleting files) can make irreversible mistakes. Mitigations: (1) human-in-the-loop approval before high-impact actions; (2) dry-run mode that shows planned actions before execution; (3) action severity classification (read vs. write vs. delete requires different approval); (4) allowlist of permitted actions.

16. **Q: How does MCP differ from OpenAI function calling?**
    A: Function calling is model-specific and embedded in the LLM call itself — tools are defined per call. MCP is model-agnostic: any MCP-compatible client can use any MCP server. MCP also supports resources (not just tools) and persistent connections. MCP enables tool reuse across different AI applications without reimplementation.

17. **Q: What is tree-of-thought reasoning and how does it differ from ReAct?**
    A: Tree-of-thought explores multiple reasoning paths simultaneously and evaluates them. Like beam search for reasoning. ReAct follows a single chain. ToT is better for: complex problems with multiple valid approaches, when backtracking is needed. Much more expensive (multiple branches).

18. **Q: How would you build an agent that can learn from its own mistakes?**
    A: (1) Store failed task trajectories in a database. (2) After each failure, run a "reflection" LLM call that identifies what went wrong. (3) Store the reflection in the agent's episodic memory. (4) At the start of similar tasks, retrieve relevant failures and include them in the context. This is the Reflexion pattern.

19. **Q: What are MCP resources and how do they differ from tools?**
    A: Resources are data that the AI application can read (like files or database schemas). They're static or dynamic content accessible via URI. Tools are functions that take arguments and return results — they have side effects (run SQL, call APIs). Resources = "read access", tools = "action".

20. **Q: How would you architect a multi-agent system where specialised agents collaborate?**
    A: Supervisor agent: routes tasks to specialised sub-agents based on task type. Sub-agents: each specialised for one domain (code agent, research agent, data agent). Communication: via shared LangGraph state or message passing. Memory: shared vector DB for cross-agent context. Orchestration: the supervisor verifies sub-agent outputs before accepting them.

---

### 7. Self-Assessment Quiz

- [ ] What is the ReAct loop? Describe each step.
- [ ] What is the difference between an agent and a chain?
- [ ] What does a tool schema need to include?
- [ ] What is a LangGraph StateGraph?
- [ ] What is a conditional edge vs. a regular edge?
- [ ] What is MCP and what problem does it solve?
- [ ] What are MCP resources vs. tools vs. prompts?
- [ ] What is checkpointing in LangGraph?
- [ ] Name 3 agent failure modes and their mitigations.
- [ ] What is working memory in an agent?
- [ ] What is human-in-the-loop in LangGraph?
- [ ] What is the Plan-and-Execute pattern?
- [ ] What does `interrupt_before` do in LangGraph?
- [ ] What is an MCP stdio transport vs. HTTP+SSE?
- [ ] How does function calling differ from regular prompting?
- [ ] What is the N+M vs N×M problem?
- [ ] How do you handle tool call errors in a ReAct loop?
- [ ] What is the Reflexion pattern for agents?
- [ ] What is a supervisor agent in a multi-agent system?
- [ ] How do you prevent an agent from running infinite loops?
- [ ] What is a subgraph in LangGraph?
- [ ] What is tool use vs. code interpreter in an agent context?
- [ ] How does streaming work in LangGraph?
- [ ] What is an agent's "final answer" condition?
- [ ] What is episodic memory and how is it stored?

**Scoring**: 22–25 ✅ = Ready. 17–21 = Review weak areas. Below 17 = Spend more time on Phase 6.

---

### 8. Common Mistakes

| Mistake | Why It Happens | How to Avoid |
|---------|---------------|--------------|
| Not limiting max iterations | "It will terminate eventually" | Always set max_iterations=10; agents can loop forever |
| Tool schemas with poor descriptions | Just the parameter type is enough | The LLM reads descriptions to decide which tool to call; write them like documentation |
| No error handling in tools | Happy path works | Wrap tool functions in try/except; return error strings instead of raising exceptions |
| Trusting all agent actions blindly | LLMs are mostly reliable | Validate parsed tool arguments against schema before executing; especially for SQL/file operations |
| State schema that's too complex | Modelling everything upfront | Start simple; add state fields only when needed |
| Not testing the agent with adversarial inputs | It works for demo cases | Test: incomplete information, wrong tool calls, contradictory observations |
| Building MCP servers without testing the client | Server seems correct | Always test end-to-end by connecting Claude Desktop; errors in the protocol are easy to miss |

---

### 9. Readiness Criteria

You are ready for Phase 7 when **all** of the following are true:

- [ ] I implemented a ReAct agent from scratch without any framework
- [ ] I built a LangGraph agent with checkpointing and conditional edges
- [ ] I built and connected a Databricks MCP server to Claude Desktop
- [ ] I demonstrated a multi-tool agent completing a real task (Research Agent)
- [ ] I scored 22/25 or higher on the Self-Assessment Quiz
- [ ] I can answer at least 16/20 Interview Questions correctly
- [ ] I understand how MCP servers are different from REST APIs

---

### 10. Revision Summary

```
REACT LOOP
─────────────────────────────────────────────────────
Thought: [reasoning about what to do]
Action:  tool_name[argument]
Observation: [tool output]
...repeat...
Final Answer: [final response]

LANGGRAPH ANATOMY
─────────────────────────────────────────────────────
StateGraph:       define nodes + edges + state schema
Nodes:            Python functions that modify state
Edges:            deterministic flow
Conditional edges: route based on state (if/else logic)
Checkpointing:    persist state to SQLite/Redis after each node

MCP PROTOCOL
─────────────────────────────────────────────────────
Resources: read-only data (schemas, files, logs)
Tools:     actions with side effects (run SQL, call API)
Prompts:   parameterised template prompts
Transport: stdio (local) or HTTP+SSE (remote)
N+M:       N apps + M tools; not N×M custom integrations
```

---

### 11. Next Phase Prerequisites

**What Phase 7 (Fine-Tuning) requires from Phase 6:**

| Phase 6 Concept | How Phase 7 Uses It |
|----------------|---------------------|
| LLM behaviour understanding | Fine-tuning changes behaviour; you need a baseline to compare against |
| Instruction following patterns | SFT dataset format mirrors agent instruction patterns |
| Tool schema design | Function calling patterns inform instruction dataset design |
| Agent evaluation | Fine-tuned models are evaluated on the same tasks as agents |
| MCP tools | Databricks MCP tools can generate training data for fine-tuning |
| RLHF preference concept | Phase 6 teaches you to recognise good vs. bad outputs — Phase 7 formalises this |

**The critical dependency**: Phase 7 is about teaching LLMs new behaviours through training data. The best training data mimics the instruction-following patterns you've mastered in Phases 5 and 6. You'll write datasets in ChatML format — which is exactly the message format you used for agents and prompts.

---

*Phase 6 | Part of the [GenAI Engineer Roadmap](./00_README.md)*
