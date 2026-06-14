"""Streamlit dashboard for Deep Agentic Market Intelligence."""

import sys
from dataclasses import replace
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config, DEFAULT_RESEARCH_QUESTION
from src.runner import run_research

st.set_page_config(
    page_title="Deep Agentic Market Intelligence",
    page_icon="🧠",
    layout="wide",
)

WORKFLOW_STEPS = [
    ("plan", "Plan", "Branch-aware sub-questions"),
    ("research", "Research", "Multi-source retrieval + signals"),
    ("knowledge_graph", "Knowledge Graph", "Entities & relationships"),
    ("verifier", "Verifier", "Sequential reasoning + gap analysis"),
    ("synthesize", "Synthesize", "Signal-ranked insights"),
    ("write_report", "Report", "Final market-intelligence report"),
]


def init_state() -> None:
    defaults = {
        "running": False,
        "completed_steps": [],
        "sub_questions": [],
        "kg_counts": (0, 0),
        "result": None,
        "error": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_results() -> None:
    st.session_state.completed_steps = []
    st.session_state.sub_questions = []
    st.session_state.kg_counts = (0, 0)
    st.session_state.result = None
    st.session_state.error = ""


def on_update(node_name: str, node_output: dict) -> None:
    if node_name not in st.session_state.completed_steps:
        st.session_state.completed_steps.append(node_name)
    if node_name == "plan" and "sub_questions" in node_output:
        st.session_state.sub_questions = [
            {"category": sq.category, "language": getattr(sq, "language", "English"),
             "question": sq.question}
            for sq in node_output["sub_questions"]
        ]
    if node_name == "knowledge_graph" and "knowledge_graph" in node_output:
        kg = node_output["knowledge_graph"]
        st.session_state.kg_counts = (
            len(kg.get("entities", [])), len(kg.get("relationships", []))
        )


def render_progress() -> None:
    cols = st.columns(len(WORKFLOW_STEPS))
    completed = set(st.session_state.completed_steps)
    for col, (node_id, label, desc) in zip(cols, WORKFLOW_STEPS):
        mark = "✅" if node_id in completed else "⬜"
        col.markdown(f"{mark} **{label}**")
        col.caption(desc)


def sidebar(config: Config) -> dict:
    with st.sidebar:
        st.header("Settings")
        try:
            config.validate()
            st.success(f"Provider: {config.llm_provider}")
        except ValueError as e:
            st.error(str(e))

        target = st.text_input("Target company", value=config.target)
        categories = st.text_input(
            "Categories (comma-separated)", value=", ".join(config.categories)
        )
        languages = st.multiselect(
            "Languages / regions",
            options=["English", "Spanish", "Hindi", "French", "German",
                     "Portuguese", "Mandarin", "Japanese"],
            default=config.languages if config.languages else ["English"],
        )
        max_loops = st.slider("Max research loops", 1, 4, config.max_loops)
        max_branches = st.slider("Max branches", 2, 8, config.max_branches)

        st.divider()
        st.subheader("Saved reports")
        reports = sorted(Path("outputs").glob("market_intel_report_*.md"), reverse=True)
        if reports:
            sel = st.selectbox("Open a report", ["(new run)"] + [p.name for p in reports[:10]])
            if sel != "(new run)" and st.button("Open report"):
                st.session_state.result = {
                    "report": (Path("outputs") / sel).read_text(encoding="utf-8"),
                    "report_path": Path("outputs") / sel,
                    "signals": [], "reasoning_trace": [], "task_records": [],
                    "knowledge_graph": {}, "confidence": 0.0, "gaps": [],
                    "kg_path": None, "evidence_count": 0,
                }
                st.session_state.completed_steps = [s[0] for s in WORKFLOW_STEPS]

    return {
        "target": target,
        "categories": [c.strip() for c in categories.split(",") if c.strip()],
        "languages": languages or ["English"],
        "max_loops": max_loops,
        "max_branches": max_branches,
    }


def render_results() -> None:
    result = st.session_state.result
    if not result:
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evidence items", result.get("evidence_count", 0))
    signals = result.get("signals", [])
    strong = sum(1 for s in signals if getattr(s, "is_signal", False))
    c2.metric("Signals (strong)", f"{len(signals)} ({strong})")
    ents, rels = (
        len(result.get("knowledge_graph", {}).get("entities", [])),
        len(result.get("knowledge_graph", {}).get("relationships", [])),
    )
    c3.metric("KG entities / edges", f"{ents} / {rels}")
    c4.metric("Confidence", f"{result.get('confidence', 0.0):.2f}")

    tabs = st.tabs(
        ["Report", "Knowledge Graph", "Signals vs Noise", "Reasoning Trace", "Research State"]
    )

    with tabs[0]:
        st.download_button(
            "Download report (.md)",
            data=result["report"],
            file_name=Path(result["report_path"]).name if result.get("report_path") else "report.md",
            mime="text/markdown",
        )
        st.markdown(result["report"])

    with tabs[1]:
        kg_path = result.get("kg_path")
        if kg_path and Path(kg_path).exists():
            html = Path(kg_path).read_text(encoding="utf-8")
            st.components.v1.html(html, height=620, scrolling=True)
        else:
            kg = result.get("knowledge_graph", {})
            rels = kg.get("relationships", [])
            if rels:
                st.table(
                    [{"source": r["source"], "relation": r["relation"], "target": r["target"]}
                     for r in rels[:40]]
                )
            else:
                st.info("No knowledge graph available.")

    with tabs[2]:
        if signals:
            rows = sorted(signals, key=lambda s: getattr(s, "score", 0), reverse=True)
            st.dataframe(
                [{"signal": s.statement, "score": round(s.score, 2),
                  "type": "signal" if s.is_signal else "noise",
                  "category": s.category} for s in rows],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No scored signals (Tavily mode or none detected).")

    with tabs[3]:
        trace = result.get("reasoning_trace", [])
        if trace:
            for step in trace:
                st.markdown(f"**Step {step.step}.** {step.thought}")
                if step.conclusion:
                    st.caption(f"↳ {step.conclusion}")
        else:
            st.info("No reasoning trace (Tavily mode).")
        gaps = result.get("gaps", [])
        if gaps:
            st.subheader("Identified gaps")
            for g in gaps:
                st.markdown(f"- {g}")

    with tabs[4]:
        tasks = result.get("task_records", [])
        if tasks:
            st.dataframe(
                [{"loop": t["loop"], "category": t["category"], "language": t["language"],
                  "status": t["status"], "question": t["question"]} for t in tasks],
                use_container_width=True, hide_index=True,
            )
            if result.get("state_path"):
                st.caption(f"Full state JSON: {result['state_path']}")
        else:
            st.info("No task records.")


def main() -> None:
    init_state()
    base_config = Config.from_env()
    overrides = sidebar(base_config)

    st.title("Deep Agentic Market Intelligence")
    st.caption(
        "Long-horizon agentic research for emerging brand discovery and competitive "
        "analysis — branch-aware orchestration, knowledge-graph construction, "
        "signal-vs-noise detection, and verifier-driven gap analysis."
    )

    question = st.text_area(
        "Research question",
        value=DEFAULT_RESEARCH_QUESTION,
        height=90,
        disabled=st.session_state.running,
    )

    col_run, col_clear, _ = st.columns([1, 1, 3])
    run_clicked = col_run.button(
        "Run research", type="primary", disabled=st.session_state.running,
        use_container_width=True,
    )
    if col_clear.button("Clear", disabled=st.session_state.running, use_container_width=True):
        reset_results()
        st.rerun()

    st.divider()
    st.subheader("Workflow")
    progress_box = st.empty()
    with progress_box.container():
        render_progress()

    if run_clicked:
        st.session_state.running = True
        reset_results()
        config = replace(
            base_config,
            target=overrides["target"],
            categories=overrides["categories"],
            languages=overrides["languages"],
            max_loops=overrides["max_loops"],
            max_branches=overrides["max_branches"],
        )
        try:
            with st.spinner("Deep agent researching across sources..."):
                result = run_research(
                    question=question, config=config,
                    target=overrides["target"], on_update=on_update,
                )
            st.session_state.result = result
        except Exception as e:
            st.session_state.error = str(e)
        finally:
            st.session_state.running = False
            st.rerun()

    if st.session_state.sub_questions:
        with st.expander("Research branches (sub-questions)", expanded=False):
            for sq in st.session_state.sub_questions:
                st.markdown(f"- **[{sq['category']} / {sq['language']}]** {sq['question']}")

    if st.session_state.error:
        st.error(st.session_state.error)

    if st.session_state.result:
        st.divider()
        render_results()


if __name__ == "__main__":
    main()
