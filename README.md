# Deep Agentic Market Intelligence (DAMI)

An autonomous, long-horizon **deep-agent** platform for **emerging brand discovery and competitive analysis**. Built on LangGraph, it recursively researches the web, reasons over evidence, identifies knowledge gaps, and synthesizes market intelligence — surfacing disruptive competitors earlier than conventional, static reports.

The default case study identifies emerging brands competing with **Procter & Gamble (P&G)**, but the engine is general: set any `--target` company and `--categories`.

## What it does

Given a market question, the agent:

1. **Plans** branch-aware sub-questions across categories (and languages/regions)
2. **Researches** each branch in parallel from **multiple source types** (general web + community/review/forum)
3. Scores **signal vs noise** on every branch
4. Builds a **dynamic knowledge graph** of brands, products, communities, and trends
5. Runs a **thinker/verifier** whose **sequential reasoning** chain is tracked through a real **sequential-thinking MCP server**, detecting gaps and spawning follow-up research (bounded loop)
6. **Synthesizes** signal-ranked insights and writes a structured **market-intelligence report**

It persists a full **research-state JSON**, an interactive **knowledge-graph HTML**, and the **markdown report** to `outputs/`.

## Architecture

```
                 ┌─────────────────────────────────────────┐
                 │                  PLAN                     │
                 │   branch-aware sub-questions (+languages) │
                 └───────────────────┬───────────────────────┘
                                     │  (parallel fan-out via Send)
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                         ▼
      RESEARCH branch          RESEARCH branch           RESEARCH branch
   multi-source retrieval   + signal/noise scoring   + reliability scoring
            └────────────────────────┼────────────────────────┘
                                     ▼
                          KNOWLEDGE GRAPH
              entity + relationship extraction & merge
                          + context off-load
                                     ▼
                        VERIFIER / THINKER
            sequential reasoning · confidence · gap analysis
                       │                      │
              (gaps, loop)              (sufficient)
                       │                      ▼
                       └──► RESEARCH      SYNTHESIZE ──► REPORT
                                       signal-ranked, KG-informed
```

The verifier's reasoning chain is recorded through a real **Model Context Protocol (MCP)** sequential-thinking server (`mcp_server/sequential_thinking_server.py`), driven by a minimal stdio MCP client (`src/mcp_client.py`).

## Quick start

```bash
# 1. Setup
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: set LLM_PROVIDER=gemini, add GEMINI_API_KEY and TAVILY_API_KEY
```

Get free keys:
- Gemini (free tier): https://aistudio.google.com/apikey
- Tavily (free tier): https://tavily.com/

```bash
# 3a. Web dashboard (recommended for demos)
./run_ui.sh          # or: streamlit run app.py

# 3b. Command line
python -m src.main

# 3c. Verify the sequential-thinking MCP server (no API key needed)
python scripts/mcp_demo.py
```

## LLM providers

Set `LLM_PROVIDER` in `.env`:

| Provider | Notes |
|----------|-------|
| `gemini` | Recommended. Free tier, `gemini-2.5-flash`. Needs `GEMINI_API_KEY`. |
| `openai` | Needs `OPENAI_API_KEY` with billing. |
| `tavily` | No LLM. Runs a reduced heuristic pipeline (limited reasoning). |

## CLI options

```bash
python -m src.main \
  --target "Procter & Gamble (P&G)" \
  --categories "oral care,grooming,beauty" \
  --languages "English,Spanish" \
  --max-loops 2
```

| Flag | Description |
|------|-------------|
| `--question` | Research question (defaults to the P&G question) |
| `--target` | Company/brand under analysis |
| `--categories` | Comma-separated categories |
| `--languages` | Comma-separated languages/regions |
| `--max-loops` | Verifier follow-up loop limit |
| `--output-dir` | Output directory (default `outputs/`) |

## Outputs

Each run writes to `outputs/`:
- `market_intel_report_<ts>.md` — the report
- `research_state_<ts>.json` — tasks, signals, knowledge graph, reasoning trace, confidence, gaps
- `knowledge_graph_<ts>.html` — interactive graph (pyvis)

## Web dashboard

`streamlit run app.py` opens a dashboard with:
- Live workflow progress and research branches
- Tabs: Report · Knowledge Graph (interactive) · Signals vs Noise · Reasoning Trace · Research State
- Sidebar controls for target, categories, languages, loops, and branches

## Capabilities mapped to the Specific Aims

| Aim | Capability | Where |
|-----|-----------|-------|
| 1. Long-horizon workflows | Parallel branches, research-state tracking, context off-load | [src/graph.py](src/graph.py), [src/memory.py](src/memory.py) |
| 2. Dynamic knowledge structures | KG construction, signal/noise scoring, synthesis | [src/knowledge_graph.py](src/knowledge_graph.py), [src/nodes.py](src/nodes.py) |
| 3. Data access | Multi-source retrieval + reliability scoring | [src/tools.py](src/tools.py) |
| Reasoning | Sequential-thinking **MCP server** + verifier gap analysis | [mcp_server/sequential_thinking_server.py](mcp_server/sequential_thinking_server.py), [src/mcp_client.py](src/mcp_client.py), [src/sequential_thinking.py](src/sequential_thinking.py) |

## Project structure

```
├── app.py                     # Streamlit dashboard
├── mcp_server/
│   └── sequential_thinking_server.py  # MCP server (JSON-RPC over stdio)
├── src/
│   ├── config.py              # Config, target/categories/languages, tunables
│   ├── state.py               # Graph state + Pydantic schemas
│   ├── prompts.py             # Prompt templates
│   ├── llm.py                 # LLM factory (Gemini / OpenAI)
│   ├── tools.py               # Multi-source retrieval + reliability scoring
│   ├── memory.py              # Research-state tracking, context off-load, persistence
│   ├── mcp_client.py          # Minimal MCP stdio client
│   ├── sequential_thinking.py # Sequential reasoning via the MCP server
│   ├── knowledge_graph.py     # KG extraction, networkx, pyvis render
│   ├── nodes.py               # LLM workflow nodes
│   ├── nodes_tavily.py        # No-LLM fallback nodes
│   ├── graph.py               # LangGraph StateGraph
│   ├── runner.py              # Shared runner (CLI + UI)
│   └── main.py                # CLI
├── outputs/                   # Generated artifacts (gitignored)
├── scripts/
│   ├── smoke_test.py          # Graph-compile smoke test
│   └── mcp_demo.py            # MCP server end-to-end demo
└── requirements.txt
```

## Sequential-thinking MCP server

The verifier offloads its reasoning chain to a standalone **MCP server** that speaks
JSON-RPC 2.0 over stdio (`initialize` handshake → `tools/list` → `tools/call`). The
server exposes a `sequentialthinking` tool that records the ordered thought history and
reports whether further thinking is needed. The agent connects to it as an MCP client.
Run `python scripts/mcp_demo.py` to see the full protocol exchange (no API key needed).

## Technologies

Python · LangGraph · LangChain · Model Context Protocol (MCP) · Google Gemini (`gemini-2.5-flash`) · Tavily · networkx · pyvis · Streamlit.

## License

MIT
