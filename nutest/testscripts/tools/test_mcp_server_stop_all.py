#!/usr/bin/env python3
"""
Unittest: Stop all MCP servers using TestML_BaseTestClass; verify registry cleared.
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


def _find_free_port_range(start: int, end: int) -> int:
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


class TestMCPStopAll(TestML_BaseTestClass):
    def setUp(self):
        super().setUp()
        from hana_ai.tools.toolkit import HANAMLToolkit

        self.tk = HANAMLToolkit(connection_context=self.conn, used_tools=["fetch_data"])  # only register fetch_data
        self.sse_port = _find_free_port_range(9001, 9100)
        self.http_port = _find_free_port_range(8000, 8100)
        self.tk.launch_mcp_server(transport="sse", host="127.0.0.1", port=self.sse_port, max_retries=15)
        self.tk.launch_mcp_server(transport="http", host="127.0.0.1", port=self.http_port, max_retries=5)
        time.sleep(1.0)

    def tearDown(self):
        try:
            # Attempt to stop any remaining servers
            _ = self.tk.stop_all_mcp_servers(force=True, timeout=3.0)
        finally:
            super().tearDown()

    def test_stop_all_registry_cleanup(self):
        expected_keys = {
            ("127.0.0.1", self.sse_port, "sse"),
            ("127.0.0.1", self.http_port, "http"),
        }
        for k in expected_keys:
            self.assertIn(k, self.tk.mcp_servers, f"Missing server in registry: {k}")

        # Stop all (graceful then force), treat registry cleanup as success
        _ = self.tk.stop_all_mcp_servers(force=False, timeout=3.0)
        _ = self.tk.stop_all_mcp_servers(force=True, timeout=3.0)

        self.assertFalse(self.tk.mcp_servers, "Registry not empty after stopping all servers")


if __name__ == "__main__":
    unittest.main()
