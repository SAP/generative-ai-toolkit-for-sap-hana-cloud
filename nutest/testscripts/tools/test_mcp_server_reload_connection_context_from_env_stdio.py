#!/usr/bin/env python3
"""E2E: Reload HANA ConnectionContext from server env without restart (stdio transport).

This test starts the stdio MCP server as a subprocess and uses StdioMCPClient
to call `admin_reload_connection_context_from_env`.
"""

from __future__ import annotations

import os
import unittest


class TestMCPReloadConnectionContextFromEnvSTDIO(unittest.TestCase):
    def test_reload_connection_context_from_env_stdio(self):
        # Ensure required env vars are present for the server process.
        for k in ("HANA_ADDRESS", "HANA_PORT", "HANA_USER", "HANA_PASSWORD"):
            if not os.environ.get(k):
                raise AssertionError(f"Missing required env var {k}. Ensure you exported HANA_* before running tests.")

        import asyncio
        from hana_ai.client.mcp_client import StdioMCPClient

        async def _main():
            client = StdioMCPClient(
                command="python",
                args=["examples/mcp_stdio_server.py"],
                server_name="HANATools",
            )
            try:
                await client.initialize()
                tools = await client.list_tools()
                names = {t.name for t in tools}
                if "admin_reload_connection_context_from_env" not in names:
                    raise AssertionError("admin_reload_connection_context_from_env not found in stdio tools/list")

                res = await client.call_tool(
                    "admin_reload_connection_context_from_env",
                    {"test_connection": False},
                )
                if not res.success:
                    raise AssertionError(f"Tool call failed: {res.error}")
                data = res.data
                # stdio client may return dict or string depending on server
                if isinstance(data, str):
                    import json
                    try:
                        data = json.loads(data)
                    except Exception:
                        pass
                if not isinstance(data, dict) or not data.get("ok"):
                    raise AssertionError(f"Expected ok=True, got: {data}")
            finally:
                await client.close()

        asyncio.run(_main())


if __name__ == "__main__":
    unittest.main()
