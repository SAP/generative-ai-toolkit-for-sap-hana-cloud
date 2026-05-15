#!/usr/bin/env python3
"""E2E: Reload HANA ConnectionContext from a file without restart (stdio transport).

This test starts the stdio MCP server as a subprocess and uses StdioMCPClient
to call `admin_reload_connection_context_from_file`.
"""

from __future__ import annotations

import os
import json
import tempfile
import unittest


class TestMCPReloadConnectionContextFromFileSTDIO(unittest.TestCase):
    def test_reload_connection_context_from_file_stdio(self):
        # Ensure required env vars are present for the server process.
        # We reuse these values to build a temporary VCAP_SERVICES file.
        for k in ("HANA_ADDRESS", "HANA_PORT", "HANA_USER", "HANA_PASSWORD"):
            if not os.environ.get(k):
                raise AssertionError(f"Missing required env var {k}. Ensure you exported HANA_* before running tests.")

        vcap_payload = {
            "hana": [
                {
                    "credentials": {
                        "host": os.environ.get("HANA_ADDRESS"),
                        "port": str(os.environ.get("HANA_PORT")),
                        "user": os.environ.get("HANA_USER"),
                        "password": os.environ.get("HANA_PASSWORD"),
                        "url": f"jdbc:sap://{os.environ.get('HANA_ADDRESS')}:{os.environ.get('HANA_PORT')}?encrypt=true&validateCertificate=true",
                    }
                }
            ]
        }

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
            tf.write("VCAP_SERVICES='" + json.dumps(vcap_payload) + "'\n")
            vcap_file = tf.name

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
                if "admin_reload_connection_context_from_file" not in names:
                    raise AssertionError("admin_reload_connection_context_from_file not found in stdio tools/list")

                res = await client.call_tool(
                    "admin_reload_connection_context_from_file",
                    {"file_path": vcap_file, "test_connection": False},
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

        asyncio.run(_main())

        try:
            os.unlink(vcap_file)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
