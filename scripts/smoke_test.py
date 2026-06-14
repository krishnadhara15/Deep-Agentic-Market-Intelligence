"""Smoke test: verify the graph compiles for each provider without API calls."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace

from src.config import Config
from src.graph import build_graph


def main() -> int:
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
    os.environ.setdefault("GEMINI_API_KEY", "gemini-test-dummy")
    os.environ.setdefault("TAVILY_API_KEY", "tvly-test-dummy")

    base = Config.from_env()
    for provider in ("gemini", "openai", "tavily"):
        config = replace(base, llm_provider=provider)
        graph = build_graph(config)
        nodes = [n for n in graph.get_graph().nodes.keys() if not n.startswith("__")]
        print(f"[{provider}] compiled OK -> nodes: {nodes}")

    print("\nAll providers compiled successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
