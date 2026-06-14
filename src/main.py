"""CLI entry point for Deep Agentic Market Intelligence."""

import argparse
import sys
from dataclasses import replace

from src.config import Config, DEFAULT_RESEARCH_QUESTION
from src.runner import run_research


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep Agentic Market Intelligence for Emerging Brand Discovery"
    )
    parser.add_argument("--question", type=str, default=DEFAULT_RESEARCH_QUESTION,
                        help="Research question to investigate")
    parser.add_argument("--target", type=str, default=None,
                        help="Company/brand under analysis (default from .env)")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated categories (overrides .env)")
    parser.add_argument("--languages", type=str, default=None,
                        help="Comma-separated languages/regions (overrides .env)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save outputs")
    parser.add_argument("--max-loops", type=int, default=None,
                        help="Override MAX_LOOPS from environment")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config.from_env()

    # CLI overrides
    if args.categories:
        config = replace(
            config,
            categories=[c.strip() for c in args.categories.split(",") if c.strip()],
        )
    if args.languages:
        config = replace(
            config,
            languages=[lang.strip() for lang in args.languages.split(",") if lang.strip()],
        )

    target = args.target or config.target

    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    max_loops = args.max_loops if args.max_loops is not None else config.max_loops

    print("=" * 64)
    print("Deep Agentic Market Intelligence (DAMI)")
    print("=" * 64)
    print(f"\nTarget: {target}")
    print(f"Question:\n{args.question}\n")
    print(f"LLM provider: {config.llm_provider}")
    if config.llm_provider == "gemini":
        print(f"Model: {config.gemini_model}")
    elif config.llm_provider == "openai":
        print(f"Model: {config.openai_model} (report: {config.openai_model_report})")
    else:
        print("Mode: Tavily web research only (no LLM)")
    print(f"Categories: {', '.join(config.categories)}")
    print(f"Languages: {', '.join(config.languages)}")
    print(f"Max loops: {max_loops} | Max branches: {config.max_branches}")
    print("-" * 64)
    print("\nStarting deep-agent workflow...\n")

    def on_update(node_name: str, node_output: dict) -> None:
        print(f"[{node_name}] completed")
        if node_name == "plan" and "sub_questions" in node_output:
            for sq in node_output["sub_questions"]:
                print(f"  -> [{sq.category}/{getattr(sq, 'language', 'English')}] {sq.question}")
        if node_name == "knowledge_graph" and "knowledge_graph" in node_output:
            kg = node_output["knowledge_graph"]
            print(f"  -> KG: {len(kg.get('entities', []))} entities, "
                  f"{len(kg.get('relationships', []))} relationships")
        if node_name == "verifier":
            print(f"  -> Loop {node_output.get('loop_count', '?')}/{max_loops}, "
                  f"confidence={node_output.get('confidence', 0):.2f}, "
                  f"sufficient={node_output.get('research_sufficient', '?')}")
        if node_name == "synthesize" and "synthesis" in node_output:
            print(f"  -> {len(node_output['synthesis'].brand_insights)} brand insights")

    try:
        result = run_research(
            question=args.question,
            config=config,
            output_dir=args.output_dir,
            max_loops=max_loops,
            target=target,
            on_update=on_update,
        )
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    strong = sum(1 for s in result["signals"] if getattr(s, "is_signal", False))
    print("\n" + "=" * 64)
    print(f"Report:          {result['report_path']}")
    print(f"Research state:  {result['state_path']}")
    if result.get("kg_path"):
        print(f"Knowledge graph: {result['kg_path']}")
    print(f"Evidence items:  {result['evidence_count']} | "
          f"Signals: {len(result['signals'])} (strong: {strong}) | "
          f"Confidence: {result['confidence']:.2f}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
