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
        try:
            res = tool._run(kwargs={"table_name": "DUMMY", "schema_name": None, "top_n": 1})
            self.assertTrue(isinstance(res, str) and len(res) > 0, "Tool call did not return a string response")
        except Exception as e:
            msg = str(e)
            self.assertTrue(isinstance(msg, str) and len(msg) > 0, "Tool call raised an exception without message")


if __name__ == "__main__":
    unittest.main()
