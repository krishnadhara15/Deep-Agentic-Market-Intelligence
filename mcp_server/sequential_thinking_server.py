#!/usr/bin/env python3
"""Sequential-Thinking MCP Server.

A standalone Model Context Protocol (MCP) server that speaks the MCP JSON-RPC 2.0
protocol over stdio (newline-delimited messages). It exposes a single tool,
`sequentialthinking`, that records an ordered chain of reasoning steps and returns
structured bookkeeping (thought number, total, whether another thought is needed,
branches, and the full history length) — mirroring the reference sequential-thinking
MCP server.

Run directly for a manual check:
    python mcp_server/sequential_thinking_server.py
then send a line of JSON-RPC on stdin, e.g.:
    {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}

This implementation has no third-party dependencies (works on Python 3.9), so it can
be launched as a subprocess by any MCP client (see src/mcp_client.py).
"""

import json
import sys
from typing import Any, Dict, List

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "sequential-thinking", "version": "1.0.0"}

TOOL_SCHEMA = {
    "name": "sequentialthinking",
    "description": (
        "Record one step in a sequential chain of reasoning. Maintains an ordered "
        "thought history and reports whether further thinking is needed."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "thought": {"type": "string", "description": "The current reasoning step"},
            "thoughtNumber": {"type": "integer", "description": "1-based index of this thought"},
            "totalThoughts": {"type": "integer", "description": "Estimated total thoughts"},
            "nextThoughtNeeded": {"type": "boolean", "description": "Whether another step follows"},
            "isRevision": {"type": "boolean"},
            "revisesThought": {"type": "integer"},
            "branchFromThought": {"type": "integer"},
            "branchId": {"type": "string"},
            "conclusion": {"type": "string"},
        },
        "required": ["thought", "thoughtNumber", "totalThoughts", "nextThoughtNeeded"],
    },
}


class SequentialThinkingState:
    """In-memory thought history for a server session."""

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []
        self.branches: Dict[str, List[Dict[str, Any]]] = {}

    def record(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self.history.append(args)
        branch_id = args.get("branchId")
        if branch_id:
            self.branches.setdefault(branch_id, []).append(args)
        return {
            "thoughtNumber": args.get("thoughtNumber"),
            "totalThoughts": args.get("totalThoughts"),
            "nextThoughtNeeded": args.get("nextThoughtNeeded"),
            "branches": list(self.branches.keys()),
            "thoughtHistoryLength": len(self.history),
        }


def _log(msg: str) -> None:
    print(f"[sequential-thinking] {msg}", file=sys.stderr, flush=True)


def _send(message: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _result(req_id: Any, result: Dict[str, Any]) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "result": result})


def _error(req_id: Any, code: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def handle(request: Dict[str, Any], state: SequentialThinkingState) -> None:
    method = request.get("method")
    req_id = request.get("id")

    # Notifications (no id) require no response.
    if method == "notifications/initialized" or (method and method.startswith("notifications/")):
        return

    if method == "initialize":
        _result(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        })
        return

    if method == "ping":
        _result(req_id, {})
        return

    if method == "tools/list":
        _result(req_id, {"tools": [TOOL_SCHEMA]})
        return

    if method == "tools/call":
        params = request.get("params", {}) or {}
        name = params.get("name")
        if name != "sequentialthinking":
            _error(req_id, -32602, f"Unknown tool: {name}")
            return
        args = params.get("arguments", {}) or {}
        if "thought" not in args:
            _error(req_id, -32602, "Missing required argument: thought")
            return
        summary = state.record(args)
        _result(req_id, {
            "content": [{"type": "text", "text": json.dumps(summary)}],
            "isError": False,
        })
        return

    if req_id is not None:
        _error(req_id, -32601, f"Method not found: {method}")


def main() -> int:
    _log("server started")
    state = SequentialThinkingState()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _log(f"could not parse line: {line[:120]}")
            continue
        try:
            handle(request, state)
        except Exception as e:  # pragma: no cover - defensive
            _log(f"handler error: {e}")
            if request.get("id") is not None:
                _error(request.get("id"), -32603, f"Internal error: {e}")
    _log("server stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
