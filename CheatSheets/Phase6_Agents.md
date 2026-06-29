# Phase 6 — AI Agents & MCP Cheat Sheet

[← CheatSheets Index](./README.md) | [Full Phase File](../07_Phase6_AI_Agents_and_MCP.md)

---

## Agent Patterns

| Pattern | Description | When to Use |
|---------|------------|-------------|
| ReAct | Thought → Action → Observation loop | General tool use |
| Plan-and-Execute | Plan all steps, then execute | Long multi-step tasks |
| Reflexion | Self-critique and revise | Quality-sensitive tasks |
| Supervisor-Worker | Orchestrator delegates to specialized agents | Complex multi-domain tasks |
| LLM-as-Judge | LLM evaluates its own or other agent output | Quality gates |

---

## ReAct Pattern (Core)

```python
REACT_PROMPT = """You have access to these tools:
{tool_descriptions}

Use this exact format:
Thought: [reasoning about what to do]
Action: tool_name
Action Input: input_value
Observation: [tool result - filled by system]
... (repeat Thought/Action/Observation as needed)
Final Answer: [your answer to the user]

Question: {question}"""
```

---

## LangGraph Quick Reference

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # append-only
    iteration: int

def agent_node(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response], "iteration": state["iteration"] + 1}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
app = graph.compile()
```

---

## Tool Design Checklist

```python
from langchain_core.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the knowledge base. Returns top matching documents.
    
    Args:
        query: The search query in natural language
        limit: Maximum number of results (1-20)
    """
    # NEVER use eval() on user input — use parameterized queries
    results = db.search(query, limit=min(limit, 20))
    return "\n".join(str(r) for r in results)
```

**Tool design rules**:
1. Clear, descriptive docstring (the LLM reads it)
2. Input validation before execution
3. Return string (structured JSON preferred)
4. Handle errors gracefully — return error description, don't raise
5. Never `eval()` user input — use `simpleeval` or parameterized queries

---

## MCP (Model Context Protocol) Pattern

```python
# Server side — expose tools over MCP
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

server = Server("my-mcp-server")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [types.Tool(
        name="run_query",
        description="Execute a SQL query on the data warehouse",
        inputSchema={"type": "object", "properties": {
            "sql": {"type": "string", "description": "SELECT query only"}
        }, "required": ["sql"]}
    )]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "run_query":
        # Validate: only allow SELECT
        sql = arguments["sql"].strip()
        if not sql.upper().startswith("SELECT"):
            return [types.TextContent(type="text", text="Error: only SELECT allowed")]
        result = await execute_query(sql)
        return [types.TextContent(type="text", text=str(result))]
```

---

## Multi-Agent Patterns

```
Supervisor pattern:
User → Supervisor Agent → [Research Agent | Code Agent | Write Agent]
                       ↑__________________________|

Key considerations:
- Loop detection: track (agent, input) pairs; stop if repeated
- Max iterations: hard cap (default 10)
- Handoff protocol: use structured messages with explicit "handoff_to" field
- Parallel execution: use asyncio.gather for independent sub-tasks
```

---

## Memory Types

| Type | Scope | Implementation | Use Case |
|------|-------|---------------|---------|
| In-context | Current conversation | Messages list | Short-term dialogue |
| Vector memory | Long-term semantic | Qdrant/ChromaDB | Retrieve past interactions |
| Entity memory | Structured facts | Dict/JSON | Track entities (user prefs, names) |
| Episodic | Past interactions | DB + retrieval | "Last time you asked about X..." |

---

## Common Mistakes

- **Unbounded agent loops** — always set `max_iterations`
- **`eval()` in tool implementations** — use `simpleeval` or safe parsers
- **No tool output length limit** — truncate long outputs (>2000 chars) before returning to LLM
- **No loop detection** — agents can get stuck in Thought/Action cycles
- **Forgetting to validate tool inputs** — LLM can hallucinate invalid arguments

---

## Interview Quick-Hits

**Q: What is the ReAct pattern and why is it effective?**  
A: ReAct (Reasoning + Acting) interleaves thinking steps with tool use. The "Thought" step lets the model reason before acting; "Observation" feeds results back. More reliable than action-only because the model can course-correct.

**Q: How do you prevent agent loops?**  
A: (1) Max iterations hard cap. (2) Detect repeated (action, input) pairs. (3) Decreasing confidence scores after repeated failures. (4) Explicit termination conditions in state.

**Q: What is MCP and why does it matter?**  
A: Model Context Protocol — Anthropic's open standard for LLMs to communicate with tools/services. Like USB-C for AI: any MCP client (Claude, LangGraph) works with any MCP server. Enables a marketplace of reusable AI tools.

**Q: How do you design tools that LLMs use reliably?**  
A: (1) Write docstrings as if explaining to a junior engineer. (2) Use specific parameter names and types. (3) Return structured JSON. (4) Handle edge cases gracefully with descriptive error messages.
