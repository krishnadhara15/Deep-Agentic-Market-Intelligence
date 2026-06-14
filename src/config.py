"""Configuration and environment loading for Deep Agentic Market Intelligence."""

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _split_env(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Config:
    """Runtime configuration loaded from environment variables."""

    llm_provider: str
    openai_api_key: str
    gemini_api_key: str
    tavily_api_key: str
    openai_model: str
    openai_model_report: str
    gemini_model: str
    gemini_model_report: str

    # Research target (generalized; defaults to the P&G case study)
    target: str
    categories: List[str]
    languages: List[str]

    # Orchestration tunables
    max_loops: int
    max_branches: int
    searches_per_question: int
    max_sub_questions: int

    # Signal-vs-noise detection
    signal_threshold: float

    # Multi-source retrieval
    community_domains: List[str] = field(default_factory=list)

    # Sequential-thinking MCP server
    use_mcp_sequential_thinking: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "gemini").lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_model_report=os.getenv("OPENAI_MODEL_REPORT", "gpt-4o"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            gemini_model_report=os.getenv("GEMINI_MODEL_REPORT", "gemini-2.5-flash"),
            target=os.getenv("TARGET", "Procter & Gamble (P&G)"),
            categories=_split_env("CATEGORIES", DEFAULT_CATEGORIES_STR),
            languages=_split_env("LANGUAGES", "English"),
            max_loops=int(os.getenv("MAX_LOOPS", "2")),
            max_branches=int(os.getenv("MAX_BRANCHES", "6")),
            searches_per_question=int(os.getenv("SEARCHES_PER_QUESTION", "3")),
            max_sub_questions=int(os.getenv("MAX_SUB_QUESTIONS", "6")),
            signal_threshold=float(os.getenv("SIGNAL_THRESHOLD", "0.5")),
            community_domains=_split_env("COMMUNITY_DOMAINS", DEFAULT_COMMUNITY_DOMAINS_STR),
            use_mcp_sequential_thinking=os.getenv(
                "USE_MCP_SEQUENTIAL_THINKING", "true"
            ).lower()
            in ("1", "true", "yes"),
        )

    def validate(self) -> None:
        """Raise ValueError if required API keys are missing."""
        missing = []
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")

        if self.llm_provider == "openai" and not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        elif self.llm_provider == "gemini" and not self.gemini_api_key:
            missing.append("GEMINI_API_KEY (or GOOGLE_API_KEY)")
        elif self.llm_provider not in ("openai", "gemini", "tavily"):
            raise ValueError(
                f"Invalid LLM_PROVIDER '{self.llm_provider}'. "
                "Use: openai, gemini, or tavily"
            )

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and add your API keys."
            )

    @property
    def uses_llm(self) -> bool:
        return self.llm_provider in ("openai", "gemini")


DEFAULT_CATEGORIES = [
    "personal care",
    "grooming",
    "oral care",
    "household cleaning",
    "baby care",
    "beauty",
]
DEFAULT_CATEGORIES_STR = ",".join(DEFAULT_CATEGORIES)

# Community / review / forum domains used for multi-source "indirect pathway" retrieval
DEFAULT_COMMUNITY_DOMAINS = [
    "reddit.com",
    "trustpilot.com",
    "influenster.com",
    "quora.com",
    "youtube.com",
]
DEFAULT_COMMUNITY_DOMAINS_STR = ",".join(DEFAULT_COMMUNITY_DOMAINS)

# Backwards-compatible alias (older modules referenced PG_CATEGORIES)
PG_CATEGORIES = DEFAULT_CATEGORIES

DEFAULT_RESEARCH_QUESTION = (
    "What are the emerging brands that compete with Procter & Gamble (P&G) "
    "products across personal care, grooming, oral care, household cleaning, "
    "baby care, and beauty — and why are they popular with consumers?"
)

CONSUMER_THEME_TAGS = [
    "sustainability",
    "natural ingredients",
    "direct-to-consumer (DTC) model",
    "product innovation",
    "affordability",
    "strong brand identity",
]

# Entity types used in the dynamic knowledge graph
ENTITY_TYPES = [
    "brand",
    "product",
    "company",
    "community",
    "trend",
    "attribute",
    "region",
]

DEFAULT_SUB_QUESTIONS = [
    {
        "question": (
            "What emerging grooming and razor brands compete with Gillette "
            "and why are consumers choosing them over P&G?"
        ),
        "category": "grooming",
        "rationale": "Gillette is a core P&G brand facing DTC disruptors.",
    },
    {
        "question": (
            "What natural personal care and deodorant brands compete with "
            "Old Spice and Secret, and why are they growing?"
        ),
        "category": "personal care",
        "rationale": "Clean-ingredient brands are reshaping personal care.",
    },
    {
        "question": (
            "What emerging oral care brands compete with Crest and Oral-B, "
            "and what consumer trends drive their popularity?"
        ),
        "category": "oral care",
        "rationale": "Sustainable and natural oral care is a fast-growing niche.",
    },
    {
        "question": (
            "What eco-friendly household cleaning brands compete with Tide "
            "and Dawn, and why do consumers prefer them?"
        ),
        "category": "household cleaning",
        "rationale": "Sustainability is disrupting home care.",
    },
    {
        "question": (
            "What premium and sustainable baby care brands compete with "
            "Pampers, and why are parents switching?"
        ),
        "category": "baby care",
        "rationale": "Premium natural baby products challenge Pampers.",
    },
    {
        "question": (
            "What DTC beauty and skincare brands compete with Olay, and why "
            "are they popular with younger consumers?"
        ),
        "category": "beauty",
        "rationale": "Digital-native beauty brands challenge legacy skincare.",
    },
]
