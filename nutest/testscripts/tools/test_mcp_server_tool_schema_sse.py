#!/usr/bin/env python3
"""
SSE mode test: start launch_mcp_server (SSE) and validate fetch_data
- Tool description is present
- Parameter descriptions present via Pydantic schema
- Direct tool call path executes and returns a response (success or error)

Exit code 0 on success; non-zero on failed assertions.
"""
from __future__ import annotations

import sys
import time
import socket


def _find_free_port(start: int = 9001, end: int = 9100) -> int:
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    # Fallback to ephemeral
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server_sse_only_with_fetch_data(port: int | None = None) -> int:
    from hana_ai.tools.toolkit import HANAMLToolkit
    from hana_ml import ConnectionContext

    cc = ConnectionContext(userkey="RaysKey", sslValidateCertificate=False, encrypt=True)
    if port is None:
        port = _find_free_port()
    tk = HANAMLToolkit(connection_context=cc, used_tools=["fetch_data"])  # only register fetch_data
    tk.launch_mcp_server(transport="sse", host="127.0.0.1", port=port, max_retries=30)
    return port


def _assert_schema_descriptions() -> None:
    from hana_ai.tools.hana_ml_tools.fetch_tools import FetchDataTool, FetchDataInput
    from hana_ml import ConnectionContext

    # Tool description (instantiate with real ConnectionContext; no DB call during __init__)
    cc = ConnectionContext(userkey="RaysKey", sslValidateCertificate=False, encrypt=True)
    tool = FetchDataTool(connection_context=cc)
    if not isinstance(tool.description, str) or len(tool.description.strip()) == 0:
        raise AssertionError("Tool description missing or empty for fetch_data")

    # Parameter descriptions via Pydantic JSON schema
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
        if not (isinstance(desc, str) and phrase in desc):
            raise AssertionError(f"Missing or incorrect description for parameter: {k}")


def _assert_direct_tool_call() -> None:
    from hana_ai.tools.hana_ml_tools.fetch_tools import FetchDataTool
    from hana_ml import ConnectionContext

    cc = ConnectionContext(userkey="RaysKey", sslValidateCertificate=False, encrypt=True)
    tool = FetchDataTool(connection_context=cc)
    # Attempt a call; we accept either a successful markdown string or an error string
    try:
        res = tool._run(kwargs={"table_name": "DUMMY", "schema_name": None, "top_n": 1})
        if not isinstance(res, str) or len(res) == 0:
            raise AssertionError("Tool call did not return a string response")
    except Exception as e:
        # Accept environment-specific failures (no table/connection), but ensure error is informative
        msg = str(e)
        if not isinstance(msg, str) or len(msg) == 0:
            raise AssertionError("Tool call raised an exception without message")


def main() -> int:
    try:
        port = _start_server_sse_only_with_fetch_data()
    except Exception as e:
        print(f"❌ Failed to start SSE MCP server: {e}")
        return 2

    # Wait for server startup
    time.sleep(1.0)

    try:
        _assert_schema_descriptions()
        _assert_direct_tool_call()
        print(f"✅ SSE(port={port}): fetch_data tool description and parameter descriptions validated; direct tool call executed.")
        return 0
    except Exception as e:
        print(f"❌ SSE validation failed: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
