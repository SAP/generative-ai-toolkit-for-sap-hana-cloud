#!/usr/bin/env python3
"""E2E: Reload HANA ConnectionContext from a file without restart (HTTP transport).

This test validates that the MCP server exposes `admin_reload_connection_context_from_file`
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
import tempfile
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


class TestMCPReloadConnectionContextFromFileHTTP(TestML_BaseTestClass):
    def setUp(self):
        super().setUp()
        from hana_ai.tools.toolkit import HANAMLToolkit

        # Start server with a minimal tool set; admin tools are registered automatically.
        self.tk = HANAMLToolkit(connection_context=self.conn, used_tools=["fetch_data"])
        self.port = _find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}/mcp"

        # Prepare a temporary VCAP_SERVICES file for reload call.
        # We reuse the current working connection so reload can succeed without needing a second HANA.
        # Address/port/user can often be introspected; password cannot, so it must be provided via env.
        try:
            addr = getattr(self.conn, "address", None) or getattr(self.conn, "_address", None)
            port = getattr(self.conn, "port", None) or getattr(self.conn, "_port", None)
            user = getattr(self.conn, "user", None) or getattr(self.conn, "_user", None)
        except Exception:
            addr = port = user = None

        if not os.environ.get("HANA_PASSWORD"):
            raise AssertionError(
                "Missing required env var HANA_PASSWORD for this e2e test. "
                "Export it in the same shell before running pytest, e.g. 'export HANA_PASSWORD=...'."
            )

        if not (addr and port and user):
            raise AssertionError("Failed to introspect address/port/user from connection context")

        vcap_payload = {
            "hana": [
                {
                    "credentials": {
                        "host": str(addr),
                        "port": str(port),
                        "user": str(user),
                        "password": str(os.environ.get("HANA_PASSWORD")),
                        "url": f"jdbc:sap://{addr}:{port}?encrypt=true&validateCertificate=true",
                    }
                }
            ]
        }

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
            tf.write("VCAP_SERVICES='" + json.dumps(vcap_payload) + "'\n")
            self._vcap_file = tf.name

        self.tk.launch_mcp_server(transport="http", host="127.0.0.1", port=self.port, max_retries=5)
        time.sleep(1.0)

    def tearDown(self):
        try:
            self.tk.stop_mcp_server(host="127.0.0.1", port=self.port, transport="http", force=True, timeout=3.0)
        finally:
            try:
                if getattr(self, "_vcap_file", None):
                    os.unlink(self._vcap_file)
            except Exception:
                pass
            super().tearDown()

    def _list_tools(self) -> Dict[str, Any]:
        from hana_ai.client.mcp_client import HTTPMCPClient
        import asyncio

        async def _main():
            client = HTTPMCPClient(base_url=self.base_url, timeout=10)
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

    def test_reload_connection_context_from_file(self):
        from hana_ai.client.mcp_client import HTTPMCPClient
        import asyncio

        tools_by_name = self._list_tools()
        self.assertIn(
            "admin_reload_connection_context_from_file",
            tools_by_name,
            "admin tool not found in tools list",
        )

        async def _main():
            client = HTTPMCPClient(base_url=self.base_url, timeout=20)
            try:
                await client.initialize()
                return await client.call_tool(
                    "admin_reload_connection_context_from_file",
                    {"file_path": self._vcap_file, "test_connection": False},
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
