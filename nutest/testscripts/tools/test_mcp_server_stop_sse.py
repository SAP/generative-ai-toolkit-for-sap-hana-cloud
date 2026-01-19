#!/usr/bin/env python3
"""
Unittest: Stop MCP server (SSE) using TestML_BaseTestClass and verify registry cleanup.
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


def _find_free_port(start: int = 9001, end: int = 9100) -> int:
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


class TestMCPStopSSE(TestML_BaseTestClass):
    def setUp(self):
        super().setUp()
        from hana_ai.tools.toolkit import HANAMLToolkit

        self.tk = HANAMLToolkit(connection_context=self.conn, used_tools=["fetch_data"])  # only register fetch_data
        self.port = _find_free_port()
        self.tk.launch_mcp_server(transport="sse", host="127.0.0.1", port=self.port, max_retries=30)
        time.sleep(1.0)

    def tearDown(self):
        # Ensure cleanup even if test fails midway
        try:
            self.tk.stop_mcp_server(host="127.0.0.1", port=self.port, transport="sse", force=True, timeout=3.0)
        finally:
            super().tearDown()

    def test_stop_server_registry_cleanup(self):
        key = ("127.0.0.1", self.port, "sse")
        self.assertIn(key, self.tk.mcp_servers, "Server not registered after launch")

        _ = self.tk.stop_mcp_server(host="127.0.0.1", port=self.port, transport="sse", force=True, timeout=3.0)

        self.assertNotIn(key, self.tk.mcp_servers, "Server registry was not cleaned up")


if __name__ == "__main__":
    unittest.main()
