"""Sequential-thinking reasoning module.

Mirrors the pattern of a sequential-thinking MCP server: the agent reasons in an
explicit, ordered chain of steps before producing a structured decision. This keeps
long-horizon reasoning auditable and is MCP-compatible (a real sequential-thinking
MCP server could be substituted for `sequential_reason` without changing callers).
"""

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


def sequential_reason(
    problem: str,
    context: str,
    config: Config,
    num_steps: int = 4,
) -> List[ReasoningStep]:
    """
    Run a sequential-thinking reasoning pass and return the ordered trace.

    Falls back to a single heuristic step if no LLM is configured or the call fails.
    """
    if not config.uses_llm:
        return [
            ReasoningStep(
                step=1,
                thought=f"Heuristic pass (no LLM): {problem}",
                conclusion="Proceeding with rule-based coverage check.",
            )
        ]

    # Imported lazily so the module stays importable without LLM extras installed.
    from src.llm import get_llm

    try:
        llm = get_llm(config)
        structured = llm.with_structured_output(_ReasoningTrace)
        chain = SEQUENTIAL_THINKING_PROMPT | structured
        result = chain.invoke(
            {"problem": problem, "context": context[:6000], "num_steps": num_steps}
        )
        steps = result.steps or []
        # Normalize step numbering
        for i, s in enumerate(steps, start=1):
            s.step = i
        return steps
    except Exception as e:  # pragma: no cover - defensive
        return [
            ReasoningStep(
                step=1,
                thought=f"Sequential reasoning unavailable: {e}",
                conclusion="Falling back to heuristic coverage check.",
            )
        ]
