"""Sequential-thinking reasoning module backed by a real MCP server.

The agent's reasoning chain is recorded through a Model Context Protocol (MCP)
sequential-thinking server (see `mcp_server/sequential_thinking_server.py`), launched
and driven via `src/mcp_client.py`. The LLM proposes ordered thoughts; each is logged
to the MCP server, which tracks the thought history and reports whether more thinking
is needed — exactly the role of a sequential-thinking MCP server.

If the MCP server or the LLM is unavailable, the module degrades gracefully to an
in-process reasoning pass so the workflow never breaks.
"""

import json
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.config import Config
from src.state import ReasoningStep

SEQUENTIAL_THINKING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a sequential-thinking reasoning engine for a market-intelligence agent.
Given a problem and context, reason step by step. Produce an ordered chain of
thoughts. Each step should build on the previous one. End with interim conclusions.

Think about: what is known, what is uncertain, what is a real market signal vs noise,
and what still needs investigation. Keep each step concise (1-2 sentences).""",
        ),
        (
            "human",
            """Problem: {problem}

Context:
{context}

Produce {num_steps} sequential reasoning steps that work toward a decision.""",
        ),
    ]
)


class _ReasoningTrace(BaseModel):
    steps: List[ReasoningStep]


def _track_via_mcp(steps: List[ReasoningStep]) -> None:
    """Record the reasoning chain through the sequential-thinking MCP server.

    Best-effort: failures are swallowed so reasoning still proceeds.
    """
    try:
        from src.mcp_client import MCPStdioClient, default_server_command

        total = len(steps)
        with MCPStdioClient(default_server_command(), timeout=8.0) as client:
            client.initialize()
            # Confirm the tool is advertised (real MCP discovery step).
            tools = {t.get("name") for t in client.list_tools()}
            if "sequentialthinking" not in tools:
                return
            for i, step in enumerate(steps, start=1):
                client.call_tool(
                    "sequentialthinking",
                    {
                        "thought": step.thought,
                        "thoughtNumber": i,
                        "totalThoughts": total,
                        "nextThoughtNeeded": i < total,
                        "conclusion": step.conclusion or "",
                    },
                )
    except Exception:
        return


def sequential_reason(
    problem: str,
    context: str,
    config: Config,
    num_steps: int = 4,
) -> List[ReasoningStep]:
    """
    Run a sequential-thinking reasoning pass and return the ordered trace.

    When an LLM is configured, it proposes the thoughts; the chain is then tracked
    through the MCP sequential-thinking server. Falls back to a heuristic step if no
    LLM is configured or the call fails.
    """
    if not config.uses_llm:
        steps = [
            ReasoningStep(
                step=1,
                thought=f"Heuristic pass (no LLM): {problem}",
                conclusion="Proceeding with rule-based coverage check.",
            )
        ]
        if config.use_mcp_sequential_thinking:
            _track_via_mcp(steps)
        return steps

    from src.llm import get_llm

    try:
        llm = get_llm(config)
        structured = llm.with_structured_output(_ReasoningTrace)
        chain = SEQUENTIAL_THINKING_PROMPT | structured
        result = chain.invoke(
            {"problem": problem, "context": context[:6000], "num_steps": num_steps}
        )
        steps = result.steps or []
        for i, s in enumerate(steps, start=1):
            s.step = i
        if steps and config.use_mcp_sequential_thinking:
            _track_via_mcp(steps)
        return steps
    except Exception as e:  # pragma: no cover - defensive
        return [
            ReasoningStep(
                step=1,
                thought=f"Sequential reasoning unavailable: {e}",
                conclusion="Falling back to heuristic coverage check.",
            )
        ]
