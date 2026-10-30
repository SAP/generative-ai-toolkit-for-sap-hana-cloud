#!/usr/bin/env python3
"""E2E: Reload HANA ConnectionContext from environment without restart.

This test validates that the MCP server exposes `admin_reload_connection_context_from_env`
and that calling it succeeds while the server stays running.

Notes
- The server is started in-process (background thread) via `HANAMLToolkit.launch_mcp_server`.
- For safety, we do NOT assert a real credential switch against a second HANA instance.
  Instead we assert:
  1) tool exists
  2) calling it returns ok=True when env vars match the already-working connection
"""

from __future__ import annotations

import os
import time
import socket
import unittest
import json
from typing import Dict, Any

try:
    from testML_BaseTestClass import TestML_BaseTestClass
except ImportError:
    import sys
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


class TestMCPReloadConnectionContextFromEnvHTTP(TestML_BaseTestClass):
    def setUp(self):
        if not os.environ.get("HANA_USERKEY"):
            self.skipTest("HANA_USERKEY is required for this live reload test")
        self._old_runtime_update = os.environ.get("HANA_ALLOW_RUNTIME_CONN_UPDATE")
        os.environ["HANA_ALLOW_RUNTIME_CONN_UPDATE"] = "true"
        super().setUp()
        from hana_ai.tools.toolkit import HANAMLToolkit

        # Start server with a minimal tool set; admin tools are registered automatically.
        self.tk = HANAMLToolkit(connection_context=self.conn, used_tools=["fetch_data"])
        self.port = _find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}/mcp"
        self.auth_token = "xin-key2-reload-e2e-token"

        self.tk.launch_mcp_server(
            transport="http", host="127.0.0.1", port=self.port,
            auth_token=self.auth_token, max_retries=5,
        )
        time.sleep(1.0)

    def tearDown(self):
        try:
            self.tk.stop_mcp_server(host="127.0.0.1", port=self.port, transport="http", force=True, timeout=3.0)
        finally:
            new_context = getattr(self.tk, "connection_context", None)
            if new_context is not None and new_context is not self.conn:
                try:
                    new_context.close()
                except Exception:
                    pass
            super().tearDown()
            if self._old_runtime_update is None:
                os.environ.pop("HANA_ALLOW_RUNTIME_CONN_UPDATE", None)
            else:
                os.environ["HANA_ALLOW_RUNTIME_CONN_UPDATE"] = self._old_runtime_update

    def _list_tools(self) -> Dict[str, Any]:
        from hana_ai.client.mcp_client import HTTPMCPClient
        import asyncio

        async def _main():
            client = HTTPMCPClient(
                base_url=self.base_url, timeout=10,
                extra_headers={"Authorization": f"Bearer {self.auth_token}"},
            )
            try:
                await client.initialize()
                return await client.list_tools()
            finally:
                try:
                    await client.close()
                except Exception:
                    pass

        tools = asyncio.run(_main())
        return {t.name: t for t in (tools or [])}

    def test_reload_connection_context_from_env(self):
        from hana_ai.client.mcp_client import HTTPMCPClient
        import asyncio

        tools_by_name = self._list_tools()
        self.assertIn(
            "admin_reload_connection_context_from_env",
            tools_by_name,
            "admin tool not found in tools list",
        )

        async def _main():
            client = HTTPMCPClient(
                base_url=self.base_url, timeout=120,
                extra_headers={"Authorization": f"Bearer {self.auth_token}"},
            )
            try:
                await client.initialize()
                return await client.call_tool(
                    "admin_reload_connection_context_from_env",
                    {"test_connection": False},
                )
            finally:
                try:
                    await client.close()
                except Exception:
                    pass

        res = asyncio.run(_main())

        self.assertTrue(res.success, f"Call failed: {res.error}")
        data: Any = res.data
        if isinstance(data, str):
            # Some servers/clients may return JSON string payloads
            try:
                data = json.loads(data)
            except Exception:
                pass
        self.assertIsInstance(data, dict, f"Expected dict response, got: {type(data)} -> {data}")
        self.assertTrue(data.get("ok"), f"Expected ok=True, got: {data}")


if __name__ == "__main__":
    unittest.main()
