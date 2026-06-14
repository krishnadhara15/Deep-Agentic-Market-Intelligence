"""Research-state tracking, context off-load, and persistence.

Supports Specific Aim 1: long-horizon agentic workflows that maintain persistent
research state (pending / active / done tasks) and prevent reasoning degradation
as context grows by off-loading evidence into a compact running summary.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from src.state import EvidenceItem, ReasoningStep, Signal, SubQuestion, TaskRecord


def make_task_records(
    sub_questions: List[SubQuestion], loop: int, status: str = "pending"
) -> List[TaskRecord]:
    """Create tracked task records for a set of sub-questions."""
    records: List[TaskRecord] = []
    for sq in sub_questions:
        records.append(
            {
                "id": uuid.uuid4().hex[:8],
                "question": sq.question,
                "category": sq.category,
                "language": getattr(sq, "language", "English"),
                "status": status,
                "loop": loop,
            }
        )
    return records


def summarize_task_state(task_records: List[TaskRecord]) -> dict:
    """Count tasks by status for quick progress reporting."""
    counts = {"pending": 0, "active": 0, "done": 0}
    for rec in task_records:
        counts[rec.get("status", "pending")] = counts.get(rec.get("status", "pending"), 0) + 1
    counts["total"] = len(task_records)
    return counts


def offload_context(
    running_summary: str,
    evidence: List[EvidenceItem],
    max_chars: int = 4000,
) -> str:
    """
    Context off-load: compress accumulated evidence into a compact running summary.

    This keeps the working context bounded during long-running investigations so
    reasoning quality does not degrade as evidence grows.
    """
    lines = [running_summary] if running_summary else []
    by_category: dict = {}
    for item in evidence:
        by_category.setdefault(item["category"], set())
        # Keep one concise line of evidence per category
        if len(by_category[item["category"]]) < 2:
            by_category[item["category"]].add(item["summary"][:240])

    for category, summaries in by_category.items():
        for s in summaries:
            lines.append(f"[{category}] {s}")

    summary = "\n".join(lines)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + " ...[truncated]"
    return summary


def persist_state(
    output_dir: str,
    timestamp: str,
    research_question: str,
    target: str,
    task_records: List[TaskRecord],
    evidence: List[EvidenceItem],
    signals: List[Signal],
    knowledge_graph: dict,
    reasoning_trace: List[ReasoningStep],
    confidence: float,
    gaps: List[str],
) -> Path:
    """Persist the full research state to a JSON file for inspection / resumption."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"research_state_{timestamp}.json"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "research_question": research_question,
        "target": target,
        "task_state": summarize_task_state(task_records),
        "task_records": task_records,
        "evidence_count": len(evidence),
        "signals": [s.model_dump() for s in signals],
        "knowledge_graph": knowledge_graph,
        "reasoning_trace": [r.model_dump() for r in reasoning_trace],
        "confidence": confidence,
        "gaps": gaps,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
