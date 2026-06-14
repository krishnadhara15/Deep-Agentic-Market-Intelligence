"""Demo: exercise the sequential-thinking MCP server over the MCP protocol.

Runs with no API keys. Launches the bundled MCP server as a subprocess, performs the
initialize handshake, lists tools, and records a short reasoning chain — printing the
server's structured responses to prove the MCP server works end to end.

    python scripts/mcp_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mcp_client import MCPStdioClient, default_server_command


def main() -> int:
    thoughts = [
        "Coverage spans grooming and personal care; oral care is missing.",
        "Community sources confirm DTC traction for Harry's — a durable signal.",
        "Confidence is moderate; one more branch on oral care would close the gap.",
        "Decision: request a follow-up branch, then synthesize.",
    ]

    with MCPStdioClient(default_server_command(), timeout=8.0) as client:
        init = client.initialize()
        print("initialize ->", init.get("serverInfo"))

        tools = client.list_tools()
        print("tools/list ->", [t["name"] for t in tools])

        total = len(thoughts)
        for i, thought in enumerate(thoughts, start=1):
            result = client.call_tool(
                "sequentialthinking",
                {
                    "thought": thought,
                    "thoughtNumber": i,
                    "totalThoughts": total,
                    "nextThoughtNeeded": i < total,
                },
            )
            text = result.get("content", [{}])[0].get("text", "")
            print(f"thought {i}/{total} recorded -> {text}")

    print("\nMCP sequential-thinking server: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
