#!/usr/bin/env python3
"""
Unittest: SSE transport MCP server tool schema and direct call validation using TestML_BaseTestClass.
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


class TestMCPServerToolSchemaSSE(TestML_BaseTestClass):
    def setUp(self):
        super().setUp()
        from hana_ai.tools.toolkit import HANAMLToolkit

        self.tk = HANAMLToolkit(connection_context=self.conn, used_tools=["fetch_data"])  # only register fetch_data
        self.port = _find_free_port()
        self.tk.launch_mcp_server(transport="sse", host="127.0.0.1", port=self.port, max_retries=30)
        time.sleep(1.0)

    def tearDown(self):
        try:
            self.tk.stop_mcp_server(host="127.0.0.1", port=self.port, transport="sse", force=True, timeout=3.0)
        finally:
            super().tearDown()

    def test_schema_descriptions(self):
        from hana_ai.tools.hana_ml_tools.fetch_tools import FetchDataTool, FetchDataInput

        tool = FetchDataTool(connection_context=self.conn)
        self.assertTrue(isinstance(tool.description, str) and len(tool.description.strip()) > 0,
                        "Tool description missing or empty for fetch_data")

        try:
            schema = FetchDataInput.model_json_schema(by_alias=True)
        except Exception:
            schema = FetchDataInput.schema(by_alias=True)

        props = schema.get("properties", {})
        checks = {
            "table_name": "the name of the table",
            "schema_name": "the schema name of the table",
            "top_n": "the number of rows to fetch",
            "last_n": "the number of rows to fetch from the end",
        }
        for k, phrase in checks.items():
            desc = props.get(k, {}).get("description")
            self.assertTrue(isinstance(desc, str) and phrase in desc,
                            f"Missing or incorrect description for parameter: {k}")

    def test_direct_tool_call(self):
        from hana_ai.tools.hana_ml_tools.fetch_tools import FetchDataTool

        tool = FetchDataTool(connection_context=self.conn)
        res = tool._run(kwargs={"table_name": "DUMMY", "schema_name": "SYS", "top_n": 1})
        self.assertIn("DUMMY", res)
        self.assertIn("X", res)

    def test_sse_protocol_schema_and_fetch(self):
        import asyncio
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        async def scenario():
            async with sse_client(f"http://127.0.0.1:{self.port}/sse") as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                    self.assertFalse(any(name.startswith("admin_") for name in tools))
                    self.assertIn("table_name", tools["fetch_data"].inputSchema["required"])
                    result = await session.call_tool(
                        "fetch_data", {"table_name": "DUMMY", "schema_name": "SYS", "top_n": 1},
                    )
                    self.assertFalse(result.isError)
                    content = "\n".join(item.text for item in result.content if item.type == "text")
                    self.assertIn("DUMMY", content)
                    self.assertIn("X", content)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
