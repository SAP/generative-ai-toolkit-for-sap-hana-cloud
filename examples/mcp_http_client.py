#!/usr/bin/env python3
"""
Minimal MCP HTTP client for HANAGraphTools.

Features:
- Bypasses environment proxies for localhost
- Obtains mcp-session-id via GET /mcp
- Performs initialize handshake
- Lists tools and invokes a tool

Usage:
  python examples/mcp_http_client.py --host 127.0.0.1 --port 8001 \
    --tool discovery_agent --arg query="show schema objects"

If your server uses an auth token, add:
  --auth-token YOUR_TOKEN

To bypass corporate proxy for localhost, this client disables trust_env.
"""
import argparse
import json
import os
from typing import Any, Dict, Optional

import requests


class MCPHttpClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8001, auth_token: Optional[str] = None):
        self.base_url = f"http://{host}:{port}/mcp"
        self.auth_token = auth_token
        self.session_id: Optional[str] = None
        self.s = requests.Session()
        # Avoid using system proxy settings (to prevent localhost requests going through corporate proxy)
        self.s.trust_env = False

    def _headers(self, accept: str = "application/json") -> Dict[str, str]:
        headers = {
            "Accept": accept,
        }
        if accept == "application/json":
            headers["Content-Type"] = "application/json"
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def obtain_session(self) -> str:
        # GET /mcp with Accept application/json returns 406 but includes mcp-session-id
        resp = self.s.get(self.base_url, headers=self._headers(accept="application/json"))
        # Accept 406/200; extract header either way
        sid = resp.headers.get("mcp-session-id")
        if not sid:
            # Try event-stream Accept as fallback
            resp2 = self.s.get(self.base_url, headers=self._headers(accept="text/event-stream"))
            sid = resp2.headers.get("mcp-session-id")
        if not sid:
            raise RuntimeError("Failed to obtain mcp-session-id from server")
        self.session_id = sid
        return sid

    def initialize(self, protocol_version: str = "2024-11-05") -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "initialize",
            "params": {
                "protocolVersion": protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "python-requests", "version": requests.__version__},
            },
        }
        resp = self.s.post(self.base_url, headers=self._headers(), data=json.dumps(payload))
        resp.raise_for_status()
        return resp.json()

    def tools_list(self) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/list",
            "params": {},
        }
        resp = self.s.post(self.base_url, headers=self._headers(), data=json.dumps(payload))
        resp.raise_for_status()
        return resp.json()

    def tools_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        resp = self.s.post(self.base_url, headers=self._headers(), data=json.dumps(payload))
        resp.raise_for_status()
        return resp.json()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP HTTP client tester")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--tool", default=None, help="Tool name to call, e.g., discovery_agent or data_agent")
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        help="Tool argument in key=value form. Repeat for multiple args.",
    )
    return parser.parse_args()


def kv_pairs_to_dict(pairs: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        # Attempt JSON decode for structured values; fallback to string
        try:
            out[k] = json.loads(v)
        except Exception:
            out[k] = v
    return out


def main():
    args = parse_arguments()
    client = MCPHttpClient(host=args.host, port=args.port, auth_token=args.auth_token)
    try:
        sid = client.obtain_session()
        print(f"mcp-session-id: {sid}")
        init = client.initialize()
        print("initialize:", json.dumps(init, ensure_ascii=False))

        lst = client.tools_list()
        print("tools/list:", json.dumps(lst, ensure_ascii=False))

        if args.tool:
            arguments = kv_pairs_to_dict(args.arg)
            result = client.tools_call(args.tool, arguments)
            print("tools/call:", json.dumps(result, ensure_ascii=False))
        else:
            print("No --tool provided; skipping tools/call.")
    except requests.HTTPError as e:
        print("HTTP error:", e, getattr(e, "response", None) and getattr(e.response, "text", ""))
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
