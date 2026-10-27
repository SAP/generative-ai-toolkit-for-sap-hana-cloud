#!/usr/bin/env python3
"""E2E: Reload HANA ConnectionContext from environment over stdio.

This test starts the stdio MCP server as a subprocess and uses StdioMCPClient
to call `admin_reload_connection_context_from_file`.
"""

from __future__ import annotations

import os
import json
import sys
import unittest


class TestMCPReloadConnectionContextFromFileSTDIO(unittest.TestCase):
    def test_reload_connection_context_from_file_stdio(self):
        if not os.environ.get("HANA_USERKEY"):
            self.skipTest("HANA_USERKEY is required for this live reload test")
        old_runtime_update = os.environ.get("HANA_ALLOW_RUNTIME_CONN_UPDATE")
        old_profiles = os.environ.get("HANA_MCP_STDIO_ALLOWED_PROFILES")
        os.environ["HANA_ALLOW_RUNTIME_CONN_UPDATE"] = "true"
        os.environ["HANA_MCP_STDIO_ALLOWED_PROFILES"] = json.dumps([
            {"command": sys.executable, "args_prefix": ["examples/mcp_stdio_server.py"]}
        ])

        import asyncio
        from hana_ai.client.mcp_client import StdioMCPClient

        async def _main():
            client = StdioMCPClient(
                command=sys.executable,
                args=["examples/mcp_stdio_server.py"],
                server_name="HANATools",
                timeout=120,
            )
            try:
                await client.initialize()
                tools = await client.list_tools()
                names = {t.name for t in tools}
                if "admin_reload_connection_context_from_env" not in names:
                    raise AssertionError("admin_reload_connection_context_from_env not found in stdio tools/list")

                res = await client.call_tool(
                    "admin_reload_connection_context_from_env",
                    {"test_connection": True},
                )
                if not res.success:
                    raise AssertionError(f"Tool call failed: {res.error}")
                data = res.data
                # stdio client may return dict or string depending on server
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except Exception:
                        pass
                if not isinstance(data, dict) or not data.get("ok"):
                    raise AssertionError(f"Expected ok=True, got: {data}")
            finally:
                await client.close()

        try:
            asyncio.run(_main())
        finally:
            if old_runtime_update is None:
                os.environ.pop("HANA_ALLOW_RUNTIME_CONN_UPDATE", None)
            else:
                os.environ["HANA_ALLOW_RUNTIME_CONN_UPDATE"] = old_runtime_update
            if old_profiles is None:
                os.environ.pop("HANA_MCP_STDIO_ALLOWED_PROFILES", None)
            else:
                os.environ["HANA_MCP_STDIO_ALLOWED_PROFILES"] = old_profiles


if __name__ == "__main__":
    unittest.main()
