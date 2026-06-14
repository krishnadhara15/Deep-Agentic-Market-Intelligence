"""Heterogeneous internal data-source connector.

Loads internal market signals from a configurable directory and exposes them as
evidence items, so the agent can fuse first-party/internal knowledge with public web
and community signals. Supports CSV, JSON, and plain-text/markdown files — a stand-in
for the heterogeneous internal sources an enterprise would connect (CRM exports, sales
notes, prior research, support tickets, etc.).

Internal records are treated as high-reliability sources.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List

from src.state import EvidenceItem

INTERNAL_RELIABILITY = 0.85
_SUPPORTED = {".csv", ".json", ".txt", ".md"}


def _records_from_csv(path: Path) -> List[Dict]:
    records = []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                records.append({k: (v or "") for k, v in row.items()})
    except Exception:
        pass
    return records


def _records_from_json(path: Path) -> List[Dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _records_from_text(path: Path) -> List[Dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return [{"title": path.stem, "note": c} for c in chunks]


def _normalize_record(rec: Dict, source_name: str) -> Dict:
    category = (rec.get("category") or rec.get("Category") or "").strip().lower()
    title = (rec.get("brand") or rec.get("title") or rec.get("Brand")
             or rec.get("name") or source_name).strip()
    note_parts = []
    for k, v in rec.items():
        if k is None or str(k).lower() == "category":
            continue
        if isinstance(v, list):
            v = " ".join(str(x) for x in v if x)
        if v:
            note_parts.append(f"{k}: {v}")
    note = "; ".join(note_parts) if note_parts else str(rec)
    return {"category": category, "title": title, "note": note}


def load_internal_records(internal_dir: str) -> List[Dict]:
    """Load and normalize all internal records from the data directory."""
    base = Path(internal_dir)
    if not base.exists():
        return []
    records: List[Dict] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED:
            continue
        if path.suffix.lower() == ".csv":
            raw = _records_from_csv(path)
        elif path.suffix.lower() == ".json":
            raw = _records_from_json(path)
        else:
            raw = _records_from_text(path)
        for rec in raw:
            records.append(_normalize_record(rec, path.stem))
    return records


def internal_evidence_for(
    category: str,
    sub_question: str,
    internal_dir: str,
    language: str = "English",
    max_items: int = 3,
) -> List[EvidenceItem]:
    """Return internal evidence items relevant to a research branch.

    Matching is by category first, then keyword overlap with the sub-question.
    """
    records = load_internal_records(internal_dir)
    if not records:
        return []

    cat = (category or "").lower()
    q_words = {w for w in sub_question.lower().split() if len(w) > 3}

    scored = []
    for rec in records:
        score = 0
        if rec["category"] and rec["category"] == cat:
            score += 3
        blob = f"{rec['title']} {rec['note']}".lower()
        score += sum(1 for w in q_words if w in blob)
        if score > 0:
            scored.append((score, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    items: List[EvidenceItem] = []
    for _, rec in scored[:max_items]:
        items.append(
            {
                "sub_question": sub_question,
                "category": category,
                "language": language,
                "source_title": f"Internal: {rec['title']}",
                "source_url": "",
                "source_type": "internal",
                "reliability": INTERNAL_RELIABILITY,
                "snippet": rec["note"][:500],
                "summary": rec["note"][:500],
            }
        )
    return items
