"""Dynamic knowledge-graph construction and visualization.

Supports Specific Aim 2: dynamic entity/attribute graph construction and
relationship discovery between brands, products, communities, and trends.
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate

from src.config import ENTITY_TYPES, Config
from src.state import EvidenceItem, KnowledgeGraph

KG_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You extract a market-intelligence knowledge graph from research evidence.

Identify entities and typed relationships. Entity types must be one of:
{entity_types}.

Relationships should connect entities, e.g.:
- (brand) competes_with (brand/company)
- (brand) driven_by (attribute/trend)
- (brand) popular_in (community/region)
- (product) belongs_to (brand)
- (trend) influences (brand/category)

Only extract entities and relationships clearly supported by the evidence.
Use canonical, concise names (e.g. "Harry's", "Gillette", "sustainability").""",
        ),
        (
            "human",
            """Target company under analysis: {target}

Evidence:
{evidence}

Extract the knowledge graph (entities + relationships).""",
        ),
    ]
)


def extract_graph(
    evidence: List[EvidenceItem], target: str, config: Config
) -> KnowledgeGraph:
    """Extract a knowledge graph from evidence using the LLM."""
    if not config.uses_llm or not evidence:
        return _heuristic_graph(evidence, target)

    from src.llm import get_llm

    evidence_text = "\n\n".join(
        f"[{e['category']}] {e['source_title']}: {e['summary'][:500]}"
        for e in evidence[:30]
    )
    try:
        llm = get_llm(config)
        structured = llm.with_structured_output(KnowledgeGraph)
        chain = KG_EXTRACTION_PROMPT | structured
        result = chain.invoke(
            {
                "entity_types": ", ".join(ENTITY_TYPES),
                "target": target,
                "evidence": evidence_text,
            }
        )
        return result
    except Exception:
        return _heuristic_graph(evidence, target)


def _heuristic_graph(evidence: List[EvidenceItem], target: str) -> KnowledgeGraph:
    """Minimal fallback graph linking the target to each researched category."""
    from src.state import Entity, Relationship

    entities = [Entity(name=target, type="company")]
    relationships: List[Relationship] = []
    seen = set()
    for e in evidence:
        cat = e["category"]
        if cat and cat not in seen:
            seen.add(cat)
            entities.append(Entity(name=cat, type="attribute"))
            relationships.append(
                Relationship(source=target, target=cat, relation="competes_in")
            )
    return KnowledgeGraph(entities=entities, relationships=relationships)


def to_networkx(graph: KnowledgeGraph):
    """Build a networkx DiGraph from the knowledge graph."""
    import networkx as nx

    g = nx.DiGraph()
    for e in graph.entities:
        g.add_node(e.name, type=e.type)
    for r in graph.relationships:
        if r.source not in g:
            g.add_node(r.source, type="unknown")
        if r.target not in g:
            g.add_node(r.target, type="unknown")
        g.add_edge(r.source, r.target, relation=r.relation)
    return g


_TYPE_COLORS = {
    "company": "#ef4444",
    "brand": "#3b82f6",
    "product": "#06b6d4",
    "community": "#a855f7",
    "trend": "#f59e0b",
    "attribute": "#10b981",
    "region": "#6366f1",
    "unknown": "#94a3b8",
}


def render_pyvis(graph: KnowledgeGraph, output_path: str) -> Optional[str]:
    """Render the knowledge graph to an interactive HTML file via pyvis.

    Returns the path on success, or None if rendering is unavailable.
    """
    try:
        from pyvis.network import Network
    except Exception:
        return None

    try:
        net = Network(
            height="600px", width="100%", directed=True, notebook=False,
            cdn_resources="in_line",
        )
    except TypeError:
        net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    for e in graph.entities:
        net.add_node(
            e.name,
            label=e.name,
            color=_TYPE_COLORS.get(e.type, _TYPE_COLORS["unknown"]),
            title=f"{e.type}",
        )
    for r in graph.relationships:
        if r.source not in net.get_nodes():
            net.add_node(r.source, label=r.source, color=_TYPE_COLORS["unknown"])
        if r.target not in net.get_nodes():
            net.add_node(r.target, label=r.target, color=_TYPE_COLORS["unknown"])
        net.add_edge(r.source, r.target, title=r.relation, label=r.relation)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        net.write_html(str(out), notebook=False)
    except Exception:
        try:
            net.save_graph(str(out))
        except Exception:
            return None
    return str(out)


def to_mermaid(graph: KnowledgeGraph, max_edges: int = 40) -> str:
    """Render the knowledge graph as a mermaid diagram (text fallback)."""
    lines = ["graph LR"]

    def _id(name: str) -> str:
        return "n_" + "".join(c if c.isalnum() else "_" for c in name)[:40]

    for r in graph.relationships[:max_edges]:
        s, t = _id(r.source), _id(r.target)
        rel = r.relation.replace('"', "'")
        lines.append(f'    {s}["{r.source}"] -->|"{rel}"| {t}["{r.target}"]')
    return "\n".join(lines)
