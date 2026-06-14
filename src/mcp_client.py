"""Minimal Model Context Protocol (MCP) stdio client.

Speaks MCP JSON-RPC 2.0 over a child process's stdio (newline-delimited messages):
performs the `initialize` handshake, lists tools, and calls tools. Dependency-free
and synchronous, with read timeouts so a misbehaving server cannot hang the agent.

This lets the project use a real MCP server (see mcp_server/) without requiring the
`mcp` SDK (which needs Python 3.10+; this project targets 3.9).
"""

import json
import select
import subprocess
import sys
from typing import Any, Dict, List, Optional

PROTOCOL_VERSION = "2024-11-05"


class MCPClientError(RuntimeError):
    pass


class MCPStdioClient:
    """A minimal MCP client that launches a server subprocess and talks over stdio."""

    def __init__(self, command: List[str], timeout: float = 10.0):
        self.command = command
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None
        self._next_id = 0

    def __enter__(self) -> "MCPStdioClient":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _write(self, message: Dict[str, Any]) -> None:
        if not self._proc or self._proc.stdin is None:
            raise MCPClientError("server process not started")
        self._proc.stdin.write(json.dumps(message) + "\n")
        self._proc.stdin.flush()

    def _read_response(self, expected_id: int) -> Dict[str, Any]:
        if not self._proc or self._proc.stdout is None:
            raise MCPClientError("server process not started")
        while True:
            ready, _, _ = select.select([self._proc.stdout], [], [], self.timeout)
            if not ready:
                raise MCPClientError("timed out waiting for MCP server response")
            line = self._proc.stdout.readline()
            if line == "":
                raise MCPClientError("MCP server closed the connection")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == expected_id:
                if "error" in msg:
                    raise MCPClientError(str(msg["error"]))
                return msg.get("result", {})
            # ignore notifications / unrelated messages

    def _request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        req_id = self._new_id()
        self._write({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
        return self._read_response(req_id)

    def _notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def initialize(self) -> Dict[str, Any]:
        result = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "dami-agent", "version": "1.0.0"},
        })
        self._notify("notifications/initialized")
        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._request("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> None:
        if not self._proc:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=3)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None


def default_server_command() -> List[str]:
    """Command to launch the bundled sequential-thinking MCP server."""
    import os

    server = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mcp_server",
        "sequential_thinking_server.py",
    )
    return [sys.executable, server]
