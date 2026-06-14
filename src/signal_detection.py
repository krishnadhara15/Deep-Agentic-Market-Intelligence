"""Statistical signal-versus-noise detection.

Complements the LLM/reasoning-based signal scoring with an evidence-derived
statistical score. For each signal we measure how well the underlying evidence
corroborates it:

- frequency: how many evidence items mention the signal's key terms
- source diversity: how many *distinct* source types corroborate it (web, community,
  internal) — cross-source agreement is a stronger statistical indicator than volume
  from a single channel
- reliability: the average reliability of the corroborating sources

These are combined into a statistical score in [0, 1] and then blended with the
reasoning score to produce a final combined score. This gives the "statistical and
reasoning-based signal detection" the platform calls for.
"""

import re
from typing import Dict, List

from src.state import EvidenceItem, Signal

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "due", "their", "they",
    "these", "such", "have", "has", "its", "consumers", "consumer", "brands", "brand",
    "products", "product", "market", "compete", "competes", "competing", "popular",
    "popularity", "growing", "growth", "because", "which", "while", "more", "than",
    "into", "over", "across", "offer", "offers", "like", "also", "their",
}


def _keywords(text: str, limit: int = 8) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z'&-]{2,}", text.lower())
    seen: List[str] = []
    for w in words:
        if w in _STOPWORDS or w in seen:
            continue
        seen.append(w)
        if len(seen) >= limit:
            break
    return seen


def _corroboration(signal: Signal, evidence: List[EvidenceItem]) -> Dict:
    keys = _keywords(signal.statement)
    if not keys:
        return {"count": 0, "types": [], "avg_reliability": 0.0}

    count = 0
    types: set = set()
    reliabilities: List[float] = []
    for item in evidence:
        haystack = f"{item.get('summary', '')} {item.get('snippet', '')} {item.get('source_title', '')}".lower()
        hits = sum(1 for k in keys if k in haystack)
        # Require at least two keyword hits to count as corroboration (reduces noise)
        if hits >= 2 or (len(keys) == 1 and hits >= 1):
            count += 1
            types.add(item.get("source_type", "web"))
            reliabilities.append(float(item.get("reliability", 0.0)))
    avg_rel = sum(reliabilities) / len(reliabilities) if reliabilities else 0.0
    return {"count": count, "types": sorted(types), "avg_reliability": avg_rel}


def _statistical_score(count: int, num_types: int, avg_reliability: float) -> float:
    # Frequency saturates quickly (diminishing returns beyond ~4 corroborations).
    freq_component = min(count / 4.0, 1.0)
    # Source diversity: 1 type -> 0.4, 2 -> 0.7, 3+ -> 1.0
    diversity_component = {0: 0.0, 1: 0.4, 2: 0.7}.get(num_types, 1.0)
    score = 0.4 * freq_component + 0.3 * diversity_component + 0.3 * avg_reliability
    return round(min(max(score, 0.0), 1.0), 3)


def score_signals_statistically(
    signals: List[Signal],
    evidence: List[EvidenceItem],
    statistical_weight: float,
    signal_threshold: float,
) -> List[Signal]:
    """Annotate signals with statistical support and a blended combined score.

    Returns the same Signal objects (mutated) sorted by combined score descending.
    """
    w = max(0.0, min(statistical_weight, 1.0))
    for s in signals:
        corr = _corroboration(s, evidence)
        s.source_count = corr["count"]
        s.source_types = corr["types"]
        s.statistical_score = _statistical_score(
            corr["count"], len(corr["types"]), corr["avg_reliability"]
        )
        s.combined_score = round((1 - w) * s.score + w * s.statistical_score, 3)
        s.is_signal = s.combined_score >= signal_threshold
    return sorted(signals, key=lambda x: x.combined_score, reverse=True)
