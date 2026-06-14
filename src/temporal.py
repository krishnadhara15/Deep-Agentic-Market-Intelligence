"""Temporal perception tracking — how market perceptions evolve over time.

Each run records a snapshot of the strongest signals (per target + category) to a
persistent history file. On subsequent runs the current signals are compared against
the most recent prior snapshot for the same subject, yielding a trend direction
(new / rising / falling / stable) and a delta. This answers the spec's question
"How are customer perceptions evolving over time?" across repeated runs.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.state import Signal, SignalTrend

_STABLE_BAND = 0.08  # |delta| within this band is considered "stable"


def _subject_key(target: str, category: str, statement: str) -> str:
    """A stable-ish key for a tracked perception subject."""
    words = re.findall(r"[a-zA-Z][a-zA-Z'&-]{3,}", statement.lower())
    head = "-".join(words[:6])
    return f"{target.lower()}|{category.lower()}|{head}"


def load_history(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        return {"subjects": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"subjects": {}}


def _direction(delta: float, is_new: bool) -> str:
    if is_new:
        return "new"
    if abs(delta) <= _STABLE_BAND:
        return "stable"
    return "rising" if delta > 0 else "falling"


def compute_trends(
    target: str,
    signals: List[Signal],
    history: Dict,
    top_n: int = 12,
) -> List[SignalTrend]:
    """Compare current signals against the prior snapshot to produce trends."""
    subjects = history.get("subjects", {})
    trends: List[SignalTrend] = []
    ranked = sorted(signals, key=lambda s: s.combined_score or s.score, reverse=True)
    for s in ranked[:top_n]:
        key = _subject_key(target, s.category, s.statement)
        current = s.combined_score or s.score
        prior_entries = subjects.get(key, [])
        previous: Optional[float] = prior_entries[-1]["score"] if prior_entries else None
        is_new = previous is None
        delta = round(current - previous, 3) if previous is not None else 0.0
        trends.append(
            SignalTrend(
                subject=s.statement,
                category=s.category,
                current_score=round(current, 3),
                previous_score=previous,
                delta=delta,
                direction=_direction(delta, is_new),
                observations=len(prior_entries) + 1,
            )
        )
    return trends


def update_history(
    path: str,
    target: str,
    signals: List[Signal],
    timestamp: Optional[str] = None,
    top_n: int = 12,
) -> None:
    """Append the current run's snapshot to the perception history file."""
    history = load_history(path)
    subjects = history.setdefault("subjects", {})
    ts = timestamp or datetime.now().isoformat()
    ranked = sorted(signals, key=lambda s: s.combined_score or s.score, reverse=True)
    for s in ranked[:top_n]:
        key = _subject_key(target, s.category, s.statement)
        entry = {"ts": ts, "score": round(s.combined_score or s.score, 3),
                 "statement": s.statement, "category": s.category}
        subjects.setdefault(key, []).append(entry)
        # Cap history length per subject
        subjects[key] = subjects[key][-20:]

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def render_trends_markdown(trends: List[SignalTrend]) -> str:
    """Render a 'Perception Over Time' section for the report."""
    if not trends:
        return ""
    arrows = {"new": "🆕", "rising": "↑", "falling": "↓", "stable": "→"}
    lines = ["## Perception Over Time", "",
             "Trends are computed by comparing this run against prior runs of the same",
             "target/category (persisted across runs).", "",
             "| Trend | Signal | Category | Now | Prev | Δ | Runs |",
             "|-------|--------|----------|-----|------|---|------|"]
    for t in trends:
        prev = f"{t.previous_score:.2f}" if t.previous_score is not None else "—"
        stmt = (t.subject[:80] + "…") if len(t.subject) > 80 else t.subject
        stmt = stmt.replace("|", "/")
        lines.append(
            f"| {arrows.get(t.direction, '')} {t.direction} | {stmt} | {t.category} "
            f"| {t.current_score:.2f} | {prev} | {t.delta:+.2f} | {t.observations} |"
        )
    return "\n".join(lines) + "\n"
