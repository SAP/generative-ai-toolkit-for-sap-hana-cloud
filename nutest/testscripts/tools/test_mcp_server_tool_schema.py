#!/usr/bin/env python3
"""
End-to-end test: start launch_mcp_server (HTTP), then call tools/list and
validate that "fetch_data" exposes per-parameter descriptions and required fields.

Assumptions:
- Default HTTP port 8000 is available (FastMCP default for HTTP).
- Only registers the minimal tool set (fetch_data) to speed up startup.

Exit code 0 on success; non-zero on failed assertions.
"""
from __future__ import annotations

import sys
import time
from typing import Dict, Any

def _start_server_http_only_with_fetch_data() -> None:
    from hana_ai.tools.toolkit import HANAMLToolkit
    from hana_ml import ConnectionContext

    # Use user store key for local environment; avoids hardcoding credentials.
    cc = ConnectionContext(userkey="RaysKey", sslValidateCertificate=False, encrypt=True)

    tk = HANAMLToolkit(connection_context=cc, used_tools=["fetch_data"])  # only register fetch_data
    # Use HTTP transport; ensure server port matches client (8000)
    tk.launch_mcp_server(transport="http", host="127.0.0.1", port=8000)


def _get_tools_via_http_client(base_url: str = "http://127.0.0.1:8000/mcp") -> Dict[str, Any]:
    from hana_ai.client.mcp_client import HTTPMCPClient

    client = HTTPMCPClient(base_url=base_url, timeout=10)
    try:
        # initialize()/list_tools() include JSON-RPC calls to fetch tool list
        import asyncio
        asyncio.run(client.initialize())
        tools = asyncio.run(client.list_tools())
    finally:
        try:
            asyncio.run(client.close())
        except Exception:
            pass

    # convert to dict by name for quick lookup
    by_name = {t.name: t for t in tools}
    return by_name


def _assert_fetch_data_schema(fetch_tool) -> None:
    schema = fetch_tool.inputSchema or {}
    props = schema.get("properties", {})
    required = schema.get("required", []) or []

    # Check required
    if "table_name" not in required:
        raise AssertionError("'table_name' should be in required list")

    # Check descriptions
    tn = props.get("table_name", {})
    sn = props.get("schema_name", {})
    topn = props.get("top_n", {})
    lastn = props.get("last_n", {})

    def _has_desc(node, key: str):
        d = node.get("description") if isinstance(node, dict) else None
        if not (isinstance(d, str) and len(d) > 0):
            raise AssertionError(f"Missing or empty description for parameter: {key}")

    _has_desc(tn, "table_name")
    _has_desc(sn, "schema_name")
    _has_desc(topn, "top_n")
    _has_desc(lastn, "last_n")


def main() -> int:
    try:
        _start_server_http_only_with_fetch_data()
    except Exception as e:
        print(f"❌ Failed to start MCP server: {e}")
        return 2

    # Wait for server to be ready
    time.sleep(2.0)

    # Retry a few times to fetch tool list
    last_err = None
    for attempt in range(5):
        try:
            tools_by_name = _get_tools_via_http_client()
            if "fetch_data" not in tools_by_name:
                raise RuntimeError("fetch_data tool not found in tools list")
            fetch_tool = tools_by_name["fetch_data"]
            # Debug: print the raw inputSchema to inspect properties
            try:
                import json
                print("\n📄 Raw inputSchema for fetch_data:\n" + json.dumps(fetch_tool.inputSchema, ensure_ascii=False, indent=2))
            except Exception:
                pass
            _assert_fetch_data_schema(fetch_tool)
            print("✅ MCP server exposes fetch_data with per-parameter descriptions and required fields.")
            return 0
        except Exception as e:
            last_err = e
            time.sleep(0.8)

    print(f"❌ Validation failed: {last_err}")
    return 3


if __name__ == "__main__":
    sys.exit(main())
