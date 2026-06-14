"""Multi-source retrieval via Tavily with source-reliability scoring.

Supports Specific Aim 3: discovering indirect pathways to high-value public signals
(community / review / forum content) alongside general web search, and evaluating
source reliability and coverage.
"""

from typing import List, Tuple
from urllib.parse import urlparse

from tavily import TavilyClient

from src.config import Config

# Heuristic reliability weights by source category. Editorial/market-research and
# review platforms are weighted higher; open forums lower (more noise).
_RELIABILITY_BY_DOMAIN = {
    "reddit.com": 0.45,
    "quora.com": 0.4,
    "youtube.com": 0.4,
    "trustpilot.com": 0.65,
    "influenster.com": 0.6,
}
_HIGH_TRUST_HINTS = (
    "marketdataforecast",
    "imarcgroup",
    "kenresearch",
    "businessinsider",
    "forbes",
    "modernretail",
    "cbinsights",
    "statista",
    "nielsen",
    "mckinsey",
)


def _domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


def score_reliability(url: str, source_type: str) -> float:
    """Heuristic source-reliability score in [0, 1]."""
    domain = _domain(url)
    if any(hint in domain for hint in _HIGH_TRUST_HINTS):
        return 0.9
    for known, weight in _RELIABILITY_BY_DOMAIN.items():
        if known in domain:
            return weight
    # Default: editorial web slightly more reliable than community by default
    return 0.7 if source_type == "web" else 0.5


def _normalize(items: List[dict], source_type: str) -> List[dict]:
    results = []
    for item in items:
        url = item.get("url", "")
        results.append(
            {
                "title": item.get("title", "Unknown"),
                "url": url,
                "content": item.get("content", ""),
                "source_type": source_type,
                "reliability": score_reliability(url, source_type),
            }
        )
    return results


def search_web(
    query: str, config: Config, max_results: int = 3, include_answer: bool = False
) -> Tuple[List[dict], str]:
    """General web search. Returns (results, tavily_answer). Backwards compatible."""
    client = TavilyClient(api_key=config.tavily_api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_answer=include_answer,
    )
    results = _normalize(response.get("results", []), "web")
    return results, (response.get("answer", "") or "")


def search_community(
    query: str, config: Config, max_results: int = 3
) -> List[dict]:
    """Search community/review/forum domains for grassroots consumer signals."""
    if not config.community_domains:
        return []
    client = TavilyClient(api_key=config.tavily_api_key)
    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=False,
            include_domains=config.community_domains,
        )
    except Exception:
        return []
    return _normalize(response.get("results", []), "community")


def multi_source_search(
    query: str,
    config: Config,
    web_results: int = 3,
    community_results: int = 2,
    include_answer: bool = True,
) -> Tuple[List[dict], str]:
    """
    Retrieve from multiple source types: general web + community/review/forum.

    Returns (combined_results, tavily_answer_from_web).
    """
    web, answer = search_web(query, config, max_results=web_results, include_answer=include_answer)
    community = search_community(query, config, max_results=community_results)
    return web + community, answer
