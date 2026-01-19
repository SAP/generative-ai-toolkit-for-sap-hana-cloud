#!/usr/bin/env python3
"""
Unittest: HTTP transport MCP server tool schema validation using TestML_BaseTestClass.
- Start server with only `fetch_data` tool
- Use HTTP client to list tools and validate input schema
"""
from __future__ import annotations

import unittest
import time
import socket
from typing import Dict, Any

try:
    from testML_BaseTestClass import TestML_BaseTestClass
except ImportError:
    import os, sys
    here = os.path.dirname(__file__)
    sys.path.append(here)
    sys.path.append(os.path.join(here, ".."))
    sys.path.append(os.path.join(here, "..", ".."))
    from testML_BaseTestClass import TestML_BaseTestClass


def _find_free_port(start: int = 8000, end: int = 8100) -> int:
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestMCPServerToolSchemaHTTP(TestML_BaseTestClass):
    def setUp(self):
        super().setUp()
        from hana_ai.tools.toolkit import HANAMLToolkit

        self.tk = HANAMLToolkit(connection_context=self.conn, used_tools=["fetch_data"])  # only register fetch_data
        self.port = _find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}/mcp"
        self.tk.launch_mcp_server(transport="http", host="127.0.0.1", port=self.port, max_retries=5)
        time.sleep(1.0)

    def tearDown(self):
        try:
            self.tk.stop_mcp_server(host="127.0.0.1", port=self.port, transport="http", force=True, timeout=3.0)
        finally:
            super().tearDown()

    def _get_tools_via_http_client(self) -> Dict[str, Any]:
        from hana_ai.client.mcp_client import HTTPMCPClient
        import asyncio

        client = HTTPMCPClient(base_url=self.base_url, timeout=10)
        try:
            asyncio.run(client.initialize())
            tools = asyncio.run(client.list_tools())
        finally:
            try:
                asyncio.run(client.close())
            except Exception:
                pass

        return {t.name: t for t in tools}

    def test_fetch_data_schema(self):
        tools_by_name = self._get_tools_via_http_client()
        self.assertIn("fetch_data", tools_by_name, "fetch_data tool not found in tools list")
        fetch_tool = tools_by_name["fetch_data"]

        schema = fetch_tool.inputSchema or {}
        props = schema.get("properties", {})
        required = schema.get("required", []) or []

        self.assertIn("table_name", required, "'table_name' should be in required list")

        # HTTP wrapper may omit per-parameter descriptions; validate presence of keys instead
        for key in ("table_name", "schema_name", "top_n", "last_n"):
            self.assertIn(key, props, f"Missing parameter key: {key}")


if __name__ == "__main__":
    unittest.main()
