"""Tavily-only node implementations (no LLM required).

A thin fallback path so the platform can run and produce a report without an LLM
key. Advanced reasoning (signal scoring, LLM knowledge-graph extraction, verifier
gap analysis) is reduced to heuristics in this mode.
"""

from typing import List

from src.config import CONSUMER_THEME_TAGS, DEFAULT_SUB_QUESTIONS, Config
from src.knowledge_graph import _heuristic_graph
from src.memory import make_task_records, offload_context
from src.state import (
    BrandInsight,
    EvidenceItem,
    ResearchState,
    ResearchTask,
    Signal,
    SubQuestion,
    SynthesisResult,
    ThemeCluster,
)
from src.tools import multi_source_search


def plan_node_tavily(state: ResearchState, config: Config) -> dict:
    sub_questions = [
        SubQuestion(**sq) for sq in DEFAULT_SUB_QUESTIONS[: config.max_branches]
    ]
    return {
        "sub_questions": sub_questions,
        "task_records": make_task_records(sub_questions, loop=0, status="done"),
    }


def research_node_tavily(state: ResearchTask, config: Config) -> dict:
    sub_q = state["sub_question"]
    results, answer = multi_source_search(
        sub_q.question, config, web_results=config.searches_per_question,
        community_results=1, include_answer=True,
    )
    summary = answer or "\n".join(f"- {r['title']}: {r['content'][:200]}" for r in results)
    summary = summary or "No Tavily summary available."

    evidence: List[EvidenceItem] = []
    for r in results:
        evidence.append(
            {
                "sub_question": sub_q.question,
                "category": sub_q.category,
                "language": getattr(sub_q, "language", "English"),
                "source_title": r["title"],
                "source_url": r["url"],
                "source_type": r["source_type"],
                "reliability": r["reliability"],
                "snippet": r["content"][:500],
                "summary": summary,
            }
        )
    if not evidence:
        evidence.append(
            {
                "sub_question": sub_q.question,
                "category": sub_q.category,
                "language": "English",
                "source_title": "No results",
                "source_url": "",
                "source_type": "web",
                "reliability": 0.0,
                "snippet": "No results.",
                "summary": summary,
            }
        )

    # Heuristic signal: treat the Tavily answer as a medium-strength signal.
    signal = Signal(
        statement=summary[:200],
        score=0.6,
        is_signal=True,
        reasoning="Tavily-aggregated answer across sources.",
        category=sub_q.category,
    )
    return {"evidence": evidence, "signals": [signal]}


def knowledge_graph_node_tavily(state: ResearchState, config: Config) -> dict:
    target = state.get("target", config.target)
    graph = _heuristic_graph(state["evidence"], target)
    running_summary = offload_context(state.get("running_summary", ""), state["evidence"])
    return {"knowledge_graph": graph.model_dump(), "running_summary": running_summary}


def verifier_node_tavily(state: ResearchState, config: Config) -> dict:
    return {
        "loop_count": state["loop_count"] + 1,
        "research_sufficient": True,
        "confidence": 0.5,
        "gaps": [],
        "pending_sub_questions": [],
        "reasoning_trace": [],
    }


def synthesize_node_tavily(state: ResearchState, config: Config) -> dict:
    seen: set = set()
    brand_insights: List[BrandInsight] = []
    for item in state["evidence"]:
        cat = item["category"]
        if cat in seen:
            continue
        seen.add(cat)
        brand_insights.append(
            BrandInsight(
                brand_name=f"Emerging brands in {cat}",
                category=cat,
                pg_products_competed=[f"{state.get('target', config.target)} {cat} portfolio"],
                why_popular=[item["summary"][:300]],
                theme_tags=CONSUMER_THEME_TAGS[:3],
                evidence_summary=item["summary"][:400],
            )
        )
    theme_clusters = [
        ThemeCluster(
            theme=t,
            description="Recurring consumer preference theme across the research.",
            example_brands=[bi.category for bi in brand_insights[:3]],
        )
        for t in CONSUMER_THEME_TAGS[:4]
    ]
    key_trends = [
        "DTC and digital-native brands are gaining share across categories.",
        "Sustainability and clean ingredients drive consumer switching.",
        "Niche insurgents attack individual categories rather than the incumbent holistically.",
        "Community and social branding outperform legacy mass-market positioning.",
    ]
    return {
        "synthesis": SynthesisResult(
            brand_insights=brand_insights,
            theme_clusters=theme_clusters,
            key_trends=key_trends,
        )
    }


def write_report_node_tavily(state: ResearchState, config: Config) -> dict:
    target = state.get("target", config.target)
    question = state["research_question"]
    synthesis = state.get("synthesis")

    by_category: dict = {}
    for item in state["evidence"]:
        by_category.setdefault(item["category"], []).append(item)

    sections = []
    for category, items in by_category.items():
        summary = items[0]["summary"] if items else "No data."
        sources = "\n".join(
            f"  - [{i['source_title']}]({i['source_url']}) ({i['source_type']})"
            for i in items
            if i["source_url"]
        )
        sections.append(f"### {category.title()}\n\n{summary}\n\n**Sources:**\n{sources}\n")

    themes = "\n".join(
        f"- **{t.theme}:** {t.description}" for t in (synthesis.theme_clusters if synthesis else [])
    )
    trends = "\n".join(f"- {t}" for t in (synthesis.key_trends if synthesis else []))
    all_sources = "\n".join(
        f"- [{e['source_title']}]({e['source_url']})"
        for e in state["evidence"]
        if e["source_url"]
    )
    kg = state.get("knowledge_graph", {})
    kg_text = "\n".join(
        f"- {r['source']} --{r['relation']}--> {r['target']}"
        for r in kg.get("relationships", [])[:25]
    ) or "No relationships extracted."

    report = f"""# Emerging Competitors to {target}
## Market Intelligence Report (Tavily mode)

**Research Question:** {question}
**Mode:** Tavily web research (no LLM)

---

## Research Methodology
Deep-agent workflow run in Tavily-only mode: branch-aware multi-source retrieval,
heuristic signal tagging, heuristic knowledge graph, single-pass verification.

## Emerging Competitors by Category

{"".join(sections)}

## Knowledge Graph Highlights

{kg_text}

## Consumer Preference Themes

{themes}

## Key Market Trends & Business Insights

{trends}

## Sources & Citations

{all_sources}

---
*Tavily-only mode. For full agentic reasoning, set LLM_PROVIDER=gemini.*
"""
    return {"report": report}
