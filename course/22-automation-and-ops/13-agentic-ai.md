# Agentic AI, MCP, and extended thinking

Chapter 10 covered LLMs as a *function* — text in, structured output out. Since 2024–2025 the frontier moved to **agents**: LLMs that run a tool-use loop, decide what to do next, and act over many steps. Plus two infrastructure shifts that matter for production: the **Model Context Protocol (MCP)** for connecting models to tools/data, and **extended thinking** for harder reasoning. This chapter is the working guide as of 2026.

## The agentic loop

A non-agentic call is one request, one response. An **agent** is a loop:

```
   ┌─────────────────────────────────────────┐
   │  prompt + tools + history                │
   │            │                             │
   │            ▼                             │
   │   model decides: respond OR call a tool  │
   │            │                             │
   │     ┌──────┴──────┐                      │
   │   respond      tool_use                  │
   │     │              │                     │
   │   done       run tool, append result ────┘
```

The model is given a set of tools. On each turn it either produces a final answer or requests a tool call; you execute the tool, append the result, and loop. The loop ends when the model stops asking for tools.

```python
import anthropic

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "get_bars",
        "description": "Fetch OHLCV bars for a symbol between two dates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "start": {"type": "string"},
                "end": {"type": "string"},
            },
            "required": ["symbol", "start", "end"],
        },
    },
]


def run_tool(name: str, args: dict) -> str:
    if name == "get_bars":
        from engine.data import YFinanceFeed
        bars = YFinanceFeed().bars(args["symbol"], args["start"], args["end"])
        return bars.tail(20).to_json()
    raise ValueError(f"unknown tool {name}")


def agent(user_message: str, max_turns: int = 10) -> str:
    messages = [{"role": "user", "content": user_message}]
    for _ in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Final answer
            return "".join(b.text for b in response.content if b.type == "text")

        # Execute every requested tool, append results
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})
    raise RuntimeError("agent exceeded max turns")
```

That's a complete agent in ~40 lines. The `max_turns` cap is non-negotiable — without it, a confused model can loop forever burning tokens.

## Where agents earn their keep (in ops)

- **Triage with investigation.** "This alert fired — pull the logs, check the metric, summarise the likely cause." The agent calls log/metric tools, reasons, reports.
- **Data exploration.** "Which of our 500 symbols had unusual volume yesterday and why?" — the agent queries, filters, follows up.
- **Multi-step report assembly.** Pull data → compute → format → render, with the model orchestrating.
- **Self-correcting parsers.** Parse a messy document; if validation fails, the model sees the error and retries with a different approach.

Where they don't: anything with a deterministic answer (write the function), anything latency-sensitive (an agent is many round-trips), and anything irreversible without human approval.

## The Claude Agent SDK

For production agents, the **Claude Agent SDK** wraps the loop, tool execution, context management, and error handling so you don't hand-roll the orchestration above. It handles:

- The tool-use loop with retries.
- Context compaction when history grows long.
- Streaming partial results.
- Permission gating on tool calls.

```python
# Conceptual — the SDK manages the loop you wrote by hand above
from claude_agent_sdk import Agent

agent = Agent(
    model="claude-sonnet-4-6",
    tools=[get_bars, run_scanner, send_report],
    system="You are an ops assistant. Investigate, then summarise.",
    max_turns=15,
)
result = agent.run("Investigate the overnight P&L drop on the pairs book.")
```

Use the SDK for real agents; hand-roll only to understand the mechanics or for a trivial one-tool loop.

## Model Context Protocol (MCP)

MCP (Anthropic, late 2024; broadly adopted across the industry by 2026) is an **open standard for connecting LLMs to tools and data sources**. Instead of hand-writing tool schemas in every app, you run an **MCP server** that exposes tools/resources, and any MCP-capable client (Claude Desktop, your agent, other vendors' models) can use them.

The value: write your "expose our trading database to an LLM" integration **once** as an MCP server; every agent and assistant can use it, with no per-app glue.

A minimal MCP server in Python (`mcp` package):

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("trading-tools")


@mcp.tool()
def get_position(symbol: str) -> dict:
    """Return the current position for a symbol."""
    return {"symbol": symbol, "qty": lookup_position(symbol)}


@mcp.tool()
def run_scanner(name: str, universe: list[str]) -> list[dict]:
    """Run a named scanner over a universe and return candidates."""
    from engine.scanners import registry
    import pandas as pd
    df = registry.get(name)().scan(universe, pd.Timestamp.utcnow())
    return df.to_dict(orient="records")


@mcp.resource("config://universe")
def get_universe() -> str:
    """Expose the tradeable universe as a resource."""
    return "\n".join(DEFAULT_UNIVERSE)


if __name__ == "__main__":
    mcp.run()                         # serves over stdio by default; HTTP/SSE also supported
```

The `@mcp.tool()` decorator turns a typed Python function into an MCP-exposed tool — the schema is derived from the type hints (same idea as the scanner registry in Module 18). Point Claude Desktop, your agent, or any MCP client at this server and the tools are available.

**Tools** are actions (the model calls them). **Resources** are data the model can read (like files or query results). **Prompts** are reusable templates the server can offer.

When to build an MCP server: when the same tools/data should be reachable by multiple AI clients, or when you want a clean boundary between "our systems" and "the model." For a single one-off agent, inline tool definitions (the first example) are simpler.

!!! warning "MCP security"
    An MCP server exposes capabilities to a model that may act on untrusted input. Treat every tool as a potential attack surface: validate inputs, scope permissions narrowly (read-only where possible), require human approval for anything that writes or trades, and never expose raw shell/SQL execution to an autonomous agent. Prompt injection through tool results is a real risk — content the model reads can try to redirect it.

## Extended thinking

For hard reasoning tasks, Claude supports **extended thinking** — the model reasons internally before answering, with a configurable token budget:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4000,
    thinking={"type": "enabled", "budget_tokens": 8000},
    messages=[{"role": "user", "content": "Design a leakage-free CV scheme for this overlapping-label dataset..."}],
)

for block in response.content:
    if block.type == "thinking":
        pass                          # the model's reasoning (usually not shown to end users)
    elif block.type == "text":
        print(block.text)             # the final answer
```

Extended thinking helps on genuinely hard problems — multi-step math, complex planning, debugging subtle logic. It costs more tokens and latency. Don't enable it for classification or extraction; reserve it for tasks where you'd want a human to "think before answering."

## Combining the pieces — an ops agent

A realistic production assistant:

- **Agent loop** (or Agent SDK) for the orchestration.
- **MCP server** exposing your read-only data tools (positions, bars, scanner results, logs).
- **Extended thinking** for the analysis step.
- **Human approval** gating any tool that writes or trades.
- **Structured output** (chapter 10) for the final report.
- **Cost controls** (chapter 10) — caps on turns, tokens, and tool calls per session.

The result is a "junior analyst" that investigates and drafts, with a human approving any consequential action.

## Pitfalls

!!! danger "Autonomous agents with write access"
    An agent that can place trades, modify databases, or run shell commands without human approval is one prompt-injection or hallucination away from catastrophe. Gate every irreversible action behind a human. For trading specifically, an agent should *propose*, never *execute*.

!!! warning "Unbounded loops"
    Always cap `max_turns`, total tokens, and tool calls. A confused agent loops; the cap is your circuit breaker.

!!! warning "Tool-result prompt injection"
    A document the agent fetches can contain "ignore previous instructions and...". Treat tool results as untrusted input; the model may act on them. Sandbox, validate, and keep a human in the loop for consequential steps.

!!! warning "Cost blowups"
    Agents make many model calls per task. A loop that should take 3 turns sometimes takes 15. Budget per-session token caps and monitor spend.

## Bottom line

For agentic AI in production (2026):

- **Hand-rolled loop** to understand it; **Claude Agent SDK** for real agents.
- **MCP server** to expose your tools/data once for any client; validate inputs, scope narrowly.
- **Extended thinking** only for genuinely hard reasoning.
- **Human-in-the-loop** for anything irreversible — agents propose, humans execute.
- **Hard caps** on turns, tokens, and tool calls.

## End of Module 22

You now have the full operational-engineering toolkit, from file handling and pipelines through backends, enterprise patterns, and the modern agentic-AI layer. The next module covers testing, packaging, and distribution.

Continue to **[Module 23 — Testing & Packaging](../23-testing-packaging/index.md)**.
