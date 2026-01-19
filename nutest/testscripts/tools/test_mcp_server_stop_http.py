#!/usr/bin/env python3
"""
Unittest: Stop MCP server (HTTP) using TestML_BaseTestClass and verify registry cleanup; optional client check.
"""
from __future__ import annotations

import unittest
import time
import socket

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


def _try_client_list_tools(base_url: str) -> bool:
    try:
        from hana_ai.client.mcp_client import HTTPMCPClient
        import asyncio
        client = HTTPMCPClient(base_url=base_url, timeout=5)
        try:
            asyncio.run(client.initialize())
            _ = asyncio.run(client.list_tools())
            return True
        finally:
            try:
                asyncio.run(client.close())
            except Exception:
                pass
    except Exception:
        return False


class TestMCPStopHTTP(TestML_BaseTestClass):
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

    def test_stop_server_registry_cleanup(self):
        key = ("127.0.0.1", self.port, "http")
        self.assertIn(key, self.tk.mcp_servers, "Server not registered after launch")

        # Optional client reachability before stop
        _ = _try_client_list_tools(self.base_url)

        _ = self.tk.stop_mcp_server(host="127.0.0.1", port=self.port, transport="http", force=True, timeout=3.0)

        self.assertNotIn(key, self.tk.mcp_servers, "Server registry was not cleaned up")


if __name__ == "__main__":
    unittest.main()
