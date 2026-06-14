"""LangGraph node implementations for Deep Agentic Market Intelligence."""

from typing import List

from src.config import Config
from src.internal_data import internal_evidence_for
from src.knowledge_graph import extract_graph
from src.llm import get_llm
from src.memory import make_task_records, offload_context
from src.signal_detection import score_signals_statistically
from src.prompts import (
    PLANNER_PROMPT,
    REPORT_PROMPT,
    RESEARCHER_PROMPT,
    SIGNAL_PROMPT,
    SYNTHESIS_PROMPT,
    VERIFIER_PROMPT,
    format_bullets,
    format_themes,
)
from src.sequential_thinking import sequential_reason
from src.state import (
    EvidenceItem,
    KnowledgeGraph,
    ResearchState,
    ResearchTask,
    Signal,
    SignalList,
    SubQuestionList,
    SynthesisResult,
    VerifierResult,
)
from src.tools import localize_query, multi_source_search


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #
def plan_node(state: ResearchState, config: Config) -> dict:
    """Decompose the research question into branch-aware sub-questions."""
    llm = get_llm(config)
    structured_llm = llm.with_structured_output(SubQuestionList)

    chain = PLANNER_PROMPT | structured_llm
    result = chain.invoke(
        {
            "research_question": state["research_question"],
            "target": state.get("target", config.target),
            "categories": format_bullets(config.categories),
            "themes": format_themes(),
            "languages": ", ".join(config.languages),
            "max_sub_questions": config.max_sub_questions,
        }
    )
    sub_questions = result.sub_questions[: config.max_branches]
    return {
        "sub_questions": sub_questions,
        "task_records": make_task_records(sub_questions, loop=0, status="done"),
    }


# --------------------------------------------------------------------------- #
# Research (per branch, parallel)
# --------------------------------------------------------------------------- #
def research_node(state: ResearchTask, config: Config) -> dict:
    """Multi-source retrieval + signal scoring for a single research branch."""
    sub_q = state["sub_question"]
    target = state.get("target", config.target)
    language = getattr(sub_q, "language", "English")
    llm = get_llm(config)

    search_query = localize_query(sub_q.question, language, config)
    results, _ = multi_source_search(
        search_query,
        config,
        web_results=config.searches_per_question,
        community_results=2,
    )

    # Fuse heterogeneous internal data relevant to this branch
    internal_items = internal_evidence_for(
        sub_q.category, sub_q.question, config.internal_data_dir, language
    )

    formatted = "\n\n".join(
        f"[{r['source_type']}|reliability {r['reliability']:.2f}] "
        f"{r['title']} ({r['url']})\n{r['content']}"
        for r in results
    ) or "No search results found for this query."
    if internal_items:
        formatted += "\n\n" + "\n\n".join(
            f"[internal|reliability {i['reliability']:.2f}] {i['source_title']}\n{i['snippet']}"
            for i in internal_items
        )

    chain = RESEARCHER_PROMPT | llm
    summary = chain.invoke(
        {
            "target": target,
            "sub_question": sub_q.question,
            "category": sub_q.category,
            "language": language,
            "search_results": formatted,
        }
    ).content

    evidence_items: List[EvidenceItem] = []
    for r in results:
        evidence_items.append(
            {
                "sub_question": sub_q.question,
                "category": sub_q.category,
                "language": language,
                "source_title": r["title"],
                "source_url": r["url"],
                "source_type": r["source_type"],
                "reliability": r["reliability"],
                "snippet": r["content"][:500],
                "summary": summary,
            }
        )
    for i in internal_items:
        i["summary"] = summary
        evidence_items.append(i)
    if not evidence_items:
        evidence_items.append(
            {
                "sub_question": sub_q.question,
                "category": sub_q.category,
                "language": getattr(sub_q, "language", "English"),
                "source_title": "No results",
                "source_url": "",
                "source_type": "web",
                "reliability": 0.0,
                "snippet": "No search results found.",
                "summary": summary,
            }
        )

    # Signal vs noise detection on this branch's summary
    signals = _score_signals(summary, sub_q.category, config)

    return {"evidence": evidence_items, "signals": signals}


def _score_signals(summary: str, category: str, config: Config) -> List[Signal]:
    try:
        llm = get_llm(config)
        structured = llm.with_structured_output(SignalList)
        chain = SIGNAL_PROMPT | structured
        result = chain.invoke({"category": category, "summary": summary})
        out = []
        for s in result.signals:
            s.category = category
            s.is_signal = s.score >= config.signal_threshold
            out.append(s)
        return out
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Knowledge graph
# --------------------------------------------------------------------------- #
def knowledge_graph_node(state: ResearchState, config: Config) -> dict:
    """Extract and merge a dynamic knowledge graph from accumulated evidence."""
    target = state.get("target", config.target)
    new_graph = extract_graph(state["evidence"], target, config)

    existing = KnowledgeGraph(**state.get("knowledge_graph", {})) if state.get(
        "knowledge_graph"
    ) else KnowledgeGraph()
    existing.merge(new_graph)

    # Context off-load: compress evidence into a running summary
    running_summary = offload_context(
        state.get("running_summary", ""), state["evidence"]
    )

    return {
        "knowledge_graph": existing.model_dump(),
        "running_summary": running_summary,
    }


# --------------------------------------------------------------------------- #
# Verifier / thinker
# --------------------------------------------------------------------------- #
def verifier_node(state: ResearchState, config: Config) -> dict:
    """Thinker/verifier: sequential reasoning + gap analysis + follow-up tasks."""
    target = state.get("target", config.target)
    signals = state.get("signals", [])
    strong = [s for s in signals if getattr(s, "is_signal", False)]

    # Sequential-thinking pass over current coverage
    reasoning_steps = sequential_reason(
        problem=(
            f"Is the evidence sufficient to report on emerging competitors to {target}? "
            "What gaps remain?"
        ),
        context=state.get("running_summary", "")
        + f"\nSignals: {len(signals)} (strong: {len(strong)})",
        config=config,
        num_steps=4,
    )

    new_loop_count = state["loop_count"] + 1
    updates: dict = {
        "loop_count": new_loop_count,
        "reasoning_trace": reasoning_steps,
    }

    try:
        llm = get_llm(config)
        structured = llm.with_structured_output(VerifierResult)
        chain = VERIFIER_PROMPT | structured
        result: VerifierResult = chain.invoke(
            {
                "target": target,
                "research_question": state["research_question"],
                "categories": format_bullets(config.categories),
                "themes": format_themes(),
                "reasoning_trace": "\n".join(
                    f"{r.step}. {r.thought}" for r in reasoning_steps
                ),
                "running_summary": state.get("running_summary", "")[:4000],
                "signal_count": len(signals),
                "strong_signal_count": len(strong),
                "loop_count": new_loop_count,
                "max_loops": state["max_loops"],
            }
        )
    except Exception as e:
        updates.update(
            {
                "research_sufficient": True,
                "confidence": 0.5,
                "gaps": [f"Verifier error: {e}"],
                "pending_sub_questions": [],
            }
        )
        return updates

    updates["confidence"] = result.confidence
    updates["gaps"] = result.gaps

    if result.is_sufficient or new_loop_count >= state["max_loops"]:
        updates["research_sufficient"] = True
        updates["pending_sub_questions"] = []
    elif result.follow_up_questions:
        follow_ups = result.follow_up_questions[:3]
        updates["research_sufficient"] = False
        updates["pending_sub_questions"] = follow_ups
        updates["task_records"] = make_task_records(
            follow_ups, loop=new_loop_count, status="done"
        )
    else:
        updates["research_sufficient"] = True
        updates["pending_sub_questions"] = []

    return updates


# --------------------------------------------------------------------------- #
# Synthesize
# --------------------------------------------------------------------------- #
def synthesize_node(state: ResearchState, config: Config) -> dict:
    """Signal-ranked, knowledge-graph-informed synthesis."""
    llm = get_llm(config)
    structured_llm = llm.with_structured_output(SynthesisResult)

    # Statistical signal detection: blend reasoning score with evidence corroboration
    ranked = score_signals_statistically(
        state.get("signals", []),
        state["evidence"],
        config.statistical_weight,
        config.signal_threshold,
    )
    signals_text = "\n".join(
        f"- (combined {s.combined_score:.2f} = reason {s.score:.2f}/stat "
        f"{s.statistical_score:.2f}; {s.source_count} sources {s.source_types}) {s.statement}"
        for s in ranked[:15]
    ) or "No scored signals."

    all_evidence = "\n\n---\n\n".join(
        f"Category: {e['category']} | Source: {e['source_title']} "
        f"({e['source_type']}, reliability {e['reliability']:.2f})\n"
        f"Summary: {e['summary']}"
        for e in state["evidence"]
    )

    chain = SYNTHESIS_PROMPT | structured_llm
    result = chain.invoke(
        {
            "target": state.get("target", config.target),
            "research_question": state["research_question"],
            "themes": format_themes(),
            "signals": signals_text,
            "all_evidence": all_evidence,
        }
    )
    return {"synthesis": result, "ranked_signals": ranked}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report_node(state: ResearchState, config: Config) -> dict:
    """Generate the final markdown market-intelligence report."""
    llm = get_llm(config, for_report=True)
    synthesis: SynthesisResult = state.get("synthesis")
    if synthesis is None:
        return {"report": "# Report\n\nNo synthesis data available."}

    synthesis_text = (
        "Brand Insights:\n"
        + "\n".join(
            f"- {b.brand_name} ({b.category}) vs {', '.join(b.pg_products_competed)}: "
            f"{'; '.join(b.why_popular)}"
            for b in synthesis.brand_insights
        )
        + "\n\nTheme Clusters:\n"
        + "\n".join(f"- {t.theme}: {t.description}" for t in synthesis.theme_clusters)
        + "\n\nKey Trends:\n"
        + "\n".join(f"- {t}" for t in synthesis.key_trends)
    )

    signals = state.get("ranked_signals") or sorted(
        state.get("signals", []), key=lambda s: getattr(s, "score", 0), reverse=True
    )
    signals_text = "\n".join(
        f"- (combined {getattr(s, 'combined_score', 0) or s.score:.2f}, "
        f"{'signal' if s.is_signal else 'noise'}; {s.source_count} sources) {s.statement}"
        for s in signals[:15]
    ) or "No scored signals."

    kg = state.get("knowledge_graph", {})
    kg_text = "\n".join(
        f"- {r['source']} --{r['relation']}--> {r['target']}"
        for r in kg.get("relationships", [])[:25]
    ) or "No relationships extracted."

    evidence_citations = "\n".join(
        f"- {e['source_title']}: {e['source_url']}"
        for e in state["evidence"]
        if e["source_url"]
    )

    chain = REPORT_PROMPT | llm
    response = chain.invoke(
        {
            "target": state.get("target", config.target),
            "research_question": state["research_question"],
            "languages": ", ".join(config.languages),
            "synthesis": synthesis_text,
            "signals": signals_text,
            "knowledge_graph": kg_text,
            "confidence": f"{state.get('confidence', 0.0):.2f}",
            "evidence_citations": evidence_citations,
        }
    )
    return {"report": response.content}
