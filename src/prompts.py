"""Prompt templates for the Deep Agentic Market Intelligence workflow."""

from langchain_core.prompts import ChatPromptTemplate

from src.config import CONSUMER_THEME_TAGS

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a market-intelligence research planner. Decompose a high-level
research question about emerging competitors to {target} into focused, branch-aware
sub-questions for deep web research.

Categories to cover:
{categories}

Consumer preference themes to investigate:
{themes}

Languages/regions to consider: {languages}

Generate sub-questions that:
1. Target specific emerging brands or brand clusters competing with {target}
2. Investigate WHY consumers are adopting these alternatives
3. Are specific enough for effective web search (brand names, categories, trends)
4. Cover diverse categories — do not focus only on one area
5. If multiple languages are listed, include at least one non-English regional branch

For each sub-question set the `language` field to the language/region lens it uses.
Generate up to {max_sub_questions} sub-questions.""",
        ),
        ("human", "Research question: {research_question}"),
    ]
)

RESEARCHER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a market research analyst summarizing multi-source search results
(web articles plus community/review/forum content) for competitive intelligence on {target}.

From the results, extract:
1. Key facts about emerging brands (positioning, growth, traction)
2. Which {target} products/brands they compete with
3. Why consumers prefer these brands (attributes, pricing, ingredients, experience)
4. Relevant consumer preference themes
5. Note grassroots/community signals distinctly from editorial sources.

Be factual and cite specifics. If results are thin, say what was found.""",
        ),
        (
            "human",
            """Research sub-question: {sub_question}
Category: {category}
Language/region: {language}

Search results (web + community):
{search_results}

Provide a concise research summary (200-400 words) of the most important market intelligence.""",
        ),
    ]
)

SIGNAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a signal-versus-noise detector for market intelligence.
Given a research summary, extract concrete market signals (claims about brands,
products, trends, adoption drivers). For each, assign:
- score (0.0 = temporary social noise, 1.0 = strong durable market signal)
- is_signal (true if it represents a meaningful market shift, not a fad)
- short reasoning

Prefer signals backed by multiple sources, concrete attributes, or sustained traction.""",
        ),
        (
            "human",
            """Category: {category}

Research summary:
{summary}

Extract and score the market signals.""",
        ),
    ]
)

VERIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a thinker/verifier agent for a long-horizon market-research system
analyzing emerging competitors to {target}.

Using the sequential reasoning trace and the running summary, decide whether the
evidence is sufficient to write a comprehensive report. Assess:
- Coverage across categories: {categories}
- Coverage across themes: {themes}
- Whether enough distinct emerging brands are covered with WHY evidence
- Confidence (0.0-1.0) in current coverage

If gaps exist, generate follow-up sub-questions (branch-aware) to close them.
Do not mark sufficient unless at least 4 distinct brands across at least 3 categories
are covered, or the loop limit is reached.""",
        ),
        (
            "human",
            """Research question: {research_question}

Sequential reasoning trace:
{reasoning_trace}

Running summary of evidence:
{running_summary}

Signals found: {signal_count} (strong: {strong_signal_count})
Loop {loop_count} / {max_loops}.

Decide sufficiency, confidence, gaps, and any follow-up sub-questions.""",
        ),
    ]
)

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a market-intelligence analyst synthesizing research on emerging
competitors to {target}. Extract structured insights from the evidence and signals.

For each emerging brand found:
- Name the brand and its category
- List specific {target} products it competes with
- Explain why consumers choose it
- Tag relevant consumer themes: {themes}

Also identify cross-brand theme clusters and high-level market trends.
Prioritize findings backed by strong signals. Only include brands with real evidence.""",
        ),
        (
            "human",
            """Research question: {research_question}

Top market signals:
{signals}

All collected evidence:
{all_evidence}

Synthesize into structured brand insights, theme clusters, and key trends.""",
        ),
    ]
)

REPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a senior strategy consultant writing a market-intelligence report
for {target} leadership, produced by an autonomous deep-agent research system.

The report must include these sections:
1. **Executive Summary** — 3-5 bullet points of top findings
2. **Research Methodology** — The deep-agent workflow (branch-aware research, signal
   detection, knowledge-graph construction, verifier-driven gap analysis)
3. **Emerging Competitors by Category** — Table/structured list of brands per category
4. **Why Consumers Choose These Brands** — Analysis organized by consumer themes
5. **Market Signals vs Noise** — Which findings are durable signals vs temporary noise,
   noting both the reasoning-based and statistical (corroboration) support
6. **Knowledge Graph Highlights** — Key entities and relationships discovered
7. **Cross-Cultural & Regional Signals** — Differences observed across the languages/
   regions researched ({languages}); omit if only English was analyzed
8. **Internal vs External Signals** — Where internal data corroborates or contradicts
   public/community signals (only if internal sources are present in the evidence)
9. **Key Market Trends & Business Insights** — Strategic implications for {target}
10. **Sources & Citations** — Source URLs from the evidence
11. **Recommendations** — 3-5 actionable strategic recommendations

Use professional business language. Be specific with brands, categories, evidence.
Format as clean markdown with headers, bullets, and tables where appropriate.""",
        ),
        (
            "human",
            """Research question: {research_question}
Languages/regions analyzed: {languages}

Synthesized findings:
{synthesis}

Top signals (with statistical corroboration):
{signals}

Knowledge graph (entities & relationships):
{knowledge_graph}

Confidence: {confidence}

Raw evidence for citations:
{evidence_citations}

Write the full market-intelligence report.""",
        ),
    ]
)


def format_bullets(items) -> str:
    return "\n".join(f"- {c}" for c in items)


def format_themes() -> str:
    return format_bullets(CONSUMER_THEME_TAGS)
