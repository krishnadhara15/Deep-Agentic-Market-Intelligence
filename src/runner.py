"""Shared research runner for CLI and web UI."""

from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from src.config import Config
from src.graph import build_graph
from src.knowledge_graph import render_pyvis
from src.memory import persist_state
from src.state import KnowledgeGraph
from src.temporal import (
    compute_trends,
    load_history,
    render_trends_markdown,
    update_history,
)

UpdateCallback = Callable[[str, dict], None]


def run_research(
    question: str,
    config: Config,
    output_dir: str = "outputs",
    max_loops: Optional[int] = None,
    target: Optional[str] = None,
    on_update: Optional[UpdateCallback] = None,
) -> dict:
    """
    Run the deep-agent market-intelligence workflow.

    Returns a dict with keys: report, report_path, state_path, kg_path,
    knowledge_graph, signals, task_records, reasoning_trace, confidence, gaps.
    """
    config.validate()
    loops = max_loops if max_loops is not None else config.max_loops
    target = target or config.target

    graph = build_graph(config)
    initial_state = {
        "research_question": question,
        "target": target,
        "sub_questions": [],
        "pending_sub_questions": [],
        "task_records": [],
        "running_summary": "",
        "evidence": [],
        "signals": [],
        "ranked_signals": [],
        "signal_trends": [],
        "knowledge_graph": {},
        "reasoning_trace": [],
        "confidence": 0.0,
        "gaps": [],
        "synthesis": None,
        "report": "",
        "loop_count": 0,
        "max_loops": loops,
        "research_sufficient": False,
    }

    final: dict = dict(initial_state)
    for mode, chunk in graph.stream(
        initial_state, stream_mode=["updates", "values"]
    ):
        if mode == "updates":
            for node_name, node_output in chunk.items():
                if on_update and isinstance(node_output, dict):
                    on_update(node_name, node_output)
        elif mode == "values":
            final = chunk

    report = final.get("report", "")
    if not report:
        raise RuntimeError("No report generated.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Statistically-ranked signals (fall back to raw signals)
    signals = final.get("ranked_signals") or final.get("signals", [])
    reasoning_trace = final.get("reasoning_trace", [])
    kg_dict = final.get("knowledge_graph", {})

    # Temporal perception tracking: compare to prior runs, then record this run
    history = load_history(config.perception_history_path)
    trends = compute_trends(target, signals, history)
    update_history(config.perception_history_path, target, signals, timestamp)
    trends_md = render_trends_markdown(trends)
    if trends_md:
        report = report.rstrip() + "\n\n" + trends_md

    report_path = out / f"market_intel_report_{timestamp}.md"
    report_path.write_text(report, encoding="utf-8")
    state_path = persist_state(
        output_dir=output_dir,
        timestamp=timestamp,
        research_question=question,
        target=target,
        task_records=final.get("task_records", []),
        evidence=final.get("evidence", []),
        signals=signals,
        knowledge_graph=kg_dict,
        reasoning_trace=reasoning_trace,
        confidence=final.get("confidence", 0.0),
        gaps=final.get("gaps", []),
        signal_trends=[t.model_dump() for t in trends],
    )

    # Render knowledge graph to interactive HTML
    kg_path = None
    if kg_dict:
        try:
            kg = KnowledgeGraph(**kg_dict)
            kg_path = render_pyvis(kg, str(out / f"knowledge_graph_{timestamp}.html"))
        except Exception:
            kg_path = None

    return {
        "report": report,
        "report_path": report_path,
        "state_path": state_path,
        "kg_path": kg_path,
        "knowledge_graph": kg_dict,
        "signals": signals,
        "task_records": final.get("task_records", []),
        "reasoning_trace": reasoning_trace,
        "confidence": final.get("confidence", 0.0),
        "gaps": final.get("gaps", []),
        "evidence_count": len(final.get("evidence", [])),
        "signal_trends": trends,
        "evidence": final.get("evidence", []),
    }
