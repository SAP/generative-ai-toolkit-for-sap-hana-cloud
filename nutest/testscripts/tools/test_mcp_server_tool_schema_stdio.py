"""Real stdio schema + HANA fetch using an explicitly approved server process."""

import asyncio
import json
import os
from pathlib import Path
import sys


def test_stdio_schema_and_fetch(monkeypatch):
    from hana_ai.client.mcp_client import StdioMCPClient

    script = str(Path(__file__).resolve().parents[3] / "examples/mcp_stdio_server.py")
    monkeypatch.setenv("HANA_USERKEY", os.environ.get("HANA_USERKEY", "XinKey2"))
    monkeypatch.setenv("HANA_ENCRYPT", "true")
    monkeypatch.setenv("HANA_SSL_VALIDATE", "true")
    monkeypatch.delenv("HANA_ALLOW_RUNTIME_CONN_UPDATE", raising=False)
    monkeypatch.setenv("HANA_MCP_STDIO_ALLOWED_PROFILES", json.dumps([
        {"command": sys.executable, "args_prefix": [script]},
    ]))

    async def scenario():
        client = StdioMCPClient(command=sys.executable, args=[script], timeout=120)
        try:
            await client.initialize()
            tools = {tool.name: tool for tool in await client.list_tools()}
            assert not any(name.startswith("admin_") for name in tools)
            schema = tools["fetch_data"].inputSchema
            assert "table_name" in schema["required"]
            for name in ("table_name", "schema_name", "top_n", "last_n"):
                assert schema["properties"][name]["description"]
            result = await client.call_tool(
                "fetch_data", {"table_name": "DUMMY", "schema_name": "SYS", "top_n": 1},
            )
            assert result.success, result.error
            assert "DUMMY" in str(result.data) and "X" in str(result.data)
        finally:
            await client.close()

    asyncio.run(scenario())
