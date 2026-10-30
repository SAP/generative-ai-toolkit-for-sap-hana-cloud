"""Live ContextAgent -> MCP -> HANA reload, over HTTP and real stdio.

Replaces the obsolete file-reload scenarios: clients must not select a server
credential file. Reload uses trusted server environment and HANA_USERKEY
(default XinKey2), with TLS verification enabled. No VCAP file is required.
Enable HTTP with RUN_CONTEXT_AGENT_MCP_E2E=1 and stdio with
RUN_CONTEXT_AGENT_MCP_STDIO_E2E=1. A live tool-calling LLM is also required.
"""

import asyncio
import json
import os
from pathlib import Path
import secrets
import socket
import sys

import httpx
import pytest


def _payload(result):
    assert result.success, result.error
    return json.loads(result.data) if isinstance(result.data, str) else result.data


@pytest.mark.parametrize(
    ("transport", "opt_in"),
    [("http", "RUN_CONTEXT_AGENT_MCP_E2E"),
     ("stdio", "RUN_CONTEXT_AGENT_MCP_STDIO_E2E")],
)
def test_context_agent_reload_from_env(transport, opt_in, monkeypatch, tmp_path):
    if os.environ.get(opt_in) != "1":
        pytest.skip(f"Set {opt_in}=1 to run live ContextAgent + MCP + HANA")

    from gen_ai_hub.proxy.langchain import init_llm
    from langchain_core.tools import StructuredTool
    from hana_ml import ConnectionContext
    from hana_ai.client.mcp_client import HTTPMCPClient, StdioMCPClient
    from hana_ai.iagents.context_agent import AgentConfig, ContextAgent
    from hana_ai.tools.toolkit import HANAMLToolkit

    userkey = os.environ.get("HANA_USERKEY", "XinKey2")
    monkeypatch.setenv("HANA_USERKEY", userkey)
    monkeypatch.setenv("HANA_ENCRYPT", "true")
    monkeypatch.setenv("HANA_SSL_VALIDATE", "true")
    monkeypatch.setenv("HANA_ALLOW_RUNTIME_CONN_UPDATE", "true")
    llm = init_llm(
        os.environ.get("CONTEXT_AGENT_E2E_LLM_MODEL", "gpt-4.1"),
        temperature=0.0, max_tokens=800,
    )

    async def scenario():
        toolkit = None
        connection = None
        client = None
        port = None
        try:
            if transport == "http":
                connection = ConnectionContext(
                    userkey=userkey, encrypt=True, sslValidateCertificate=True,
                )
                toolkit = HANAMLToolkit(connection_context=connection, used_tools=["fetch_data"])
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                token = secrets.token_urlsafe(32)
                toolkit.launch_mcp_server(
                    transport="http", host="127.0.0.1", port=port,
                    auth_token=token, max_retries=1,
                )
                base_url = f"http://127.0.0.1:{port}/mcp"
                client = HTTPMCPClient(
                    base_url, timeout=120,
                    extra_headers={"Authorization": f"Bearer {token}"},
                )
                async with httpx.AsyncClient(trust_env=False) as unauthenticated:
                    for headers in ({}, {"Authorization": "Bearer incorrect-test-token"}):
                        response = await unauthenticated.post(base_url, headers=headers, json={
                            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": "admin_reload_connection_context_from_env",
                                       "arguments": {"test_connection": True}},
                        })
                        assert response.status_code == 401
                        assert toolkit.connection_context is connection
            else:
                script = str(Path(__file__).resolve().parents[2] / "examples/mcp_stdio_server.py")
                monkeypatch.setenv("HANA_MCP_STDIO_ALLOWED_PROFILES", json.dumps([
                    {"command": sys.executable, "args_prefix": [script]},
                ]))
                client = StdioMCPClient(command=sys.executable, args=[script], timeout=120)

            await client.initialize()
            schemas = {tool.name: tool.inputSchema for tool in await client.list_tools()}
            assert "admin_reload_connection_context_from_file" not in schemas
            assert set(schemas["admin_reload_connection_context_from_env"]["properties"]) == {"test_connection"}

            async def fetch_dummy():
                result = await client.call_tool(
                    "fetch_data", {"table_name": "DUMMY", "schema_name": "SYS", "top_n": 1},
                )
                assert result.success, result.error
                assert "DUMMY" in str(result.data) and "X" in str(result.data)

            await fetch_dummy()
            loop = asyncio.get_running_loop()
            calls = []

            def reload_environment(test_connection: bool = True) -> str:
                """Reload HANA from trusted MCP server environment; validate the new connection."""
                result = asyncio.run_coroutine_threadsafe(client.call_tool(
                    "admin_reload_connection_context_from_env",
                    {"test_connection": test_connection},
                ), loop).result(timeout=130)
                data = _payload(result)
                calls.append((test_connection, data))
                return json.dumps(data)

            tool = StructuredTool.from_function(
                reload_environment, name="mcp_reload_connection_context_from_env",
                return_direct=True,
            )
            agent = ContextAgent(
                llm=llm, tools=[tool], storage_dir=str(tmp_path / transport),
                config=AgentConfig(
                    skills_use_llm_selector=False, enable_structured_notes=False,
                    enable_compaction=False, enable_session_summary=False,
                ),
                progress_bar=False,
            )
            output = await asyncio.to_thread(
                agent.chat,
                "Call mcp_reload_connection_context_from_env exactly once with "
                "test_connection=true. Return the tool result.",
            )
            assert isinstance(output, str)
            assert len(calls) == 1, "The LLM must actually invoke the reload tool exactly once"
            test_connection, data = calls[0]
            assert test_connection is True
            assert data["ok"] is True
            assert data["connection"]["userkey"] == "***"
            assert data["connection"]["encrypt"] is True
            assert data["connection"]["sslValidateCertificate"] is True
            assert data["tools_updated_in_place"] > 0
            if toolkit is not None:
                assert toolkit.connection_context is not connection
            await fetch_dummy()
        finally:
            try:
                if client is not None:
                    await client.close()
            finally:
                try:
                    if toolkit is not None:
                        toolkit.stop_mcp_server(
                            host="127.0.0.1", port=port, transport="http", force=True, timeout=3.0,
                        )
                finally:
                    try:
                        if toolkit is not None and toolkit.connection_context is not connection:
                            toolkit.connection_context.close()
                    finally:
                        if connection is not None:
                            connection.close()

    asyncio.run(scenario())
