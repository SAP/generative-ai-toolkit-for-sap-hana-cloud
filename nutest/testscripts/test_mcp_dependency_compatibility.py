"""Exercise the launcher's HTTP settings, tool calls, and audit boundary."""

import json
import os
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, Field
from starlette.testclient import TestClient

from hana_ai.langchain_compat import BaseTool
from hana_ai.tools import toolkit as toolkit_module


class EchoInput(BaseModel):
    value: int = Field(description="Value returned by the test tool")
    fail: bool = False


class EchoTool(BaseTool):
    name: str = "dependency_echo"
    description: str = "Return an integer to verify the MCP tool boundary."
    args_schema: type[BaseModel] = EchoInput

    def _run(self, value: int, fail: bool = False) -> str:
        if fail:
            raise ValueError("deliberate test failure")
        return json.dumps({"value": value})


def launch_http_app(monkeypatch, toolkit, **options):
    """Use the real launcher; replace only the background network listener."""
    launched = []

    def capture_run(server, **settings):
        launched.append((server, settings))

    monkeypatch.delenv("HANA_ALLOW_RUNTIME_CONN_UPDATE", raising=False)
    monkeypatch.setattr(toolkit_module.HANAMLToolkit, "_global_mcp_servers", {})
    monkeypatch.setattr(toolkit_module.HANAMLToolkit, "is_port_available", lambda *_: True)
    with monkeypatch.context() as startup:
        startup.setattr(toolkit_module.FastMCPHTTP, "run", capture_run)
        startup.setattr(toolkit_module.threading.Thread, "start", lambda thread: thread.run())
        toolkit.launch_mcp_server(
            transport="http", host="127.0.0.1", port=18937,
            auth_token="dependency-test-token", max_retries=1, **options,
        )
    assert len(launched) == 1, "the real launcher must reach HTTP startup"
    server, settings = launched[0]
    assert (settings["host"], settings["port"], settings["path"]) == ("127.0.0.1", 18937, "/mcp")
    # Feed the real startup options into FastMCP's real ASGI transport.
    app_options = {key: settings[key] for key in ("path", "json_response", "stateless_http") if key in settings}
    return server.http_app(**app_options)


@pytest.mark.parametrize("mode", [None, False, True], ids=["default-stateful", "stateful", "stateless"])
def test_http_tool_calls_preserve_auth_schema_sessions_and_audit(monkeypatch, mode):
    events = []
    prepared = []
    finished = []
    toolkit = toolkit_module.HANAMLToolkit(
        connection_context=Mock(spec=toolkit_module.ConnectionContext), used_tools=[],
    )
    toolkit.add_custom_tool(EchoTool())
    monkeypatch.setattr(toolkit, "_emit_audit_event", lambda event: events.append(event))
    monkeypatch.setattr(toolkit, "_ensure_hana_execution_context",
                        lambda *args, **kwargs: prepared.append((args, kwargs)))
    monkeypatch.setattr(toolkit, "_update_hana_session_event", lambda event: finished.append(event))
    # A host-wide default must not change this library's default session mode.
    monkeypatch.setattr(__import__("fastmcp").settings, "stateless_http", True)
    options = {} if mode is None else {"stateless_http": mode}
    app = launch_http_app(monkeypatch, toolkit, **options)
    headers = {"Accept": "application/json, text/event-stream",
               "Authorization": "Bearer dependency-test-token"}
    initialize = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                  "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                             "clientInfo": {"name": "dependency-regression", "version": "1"}}}
    with TestClient(app, base_url="http://127.0.0.1") as client:
        for authorization in (None, "Bearer invalid"):
            unauthorized = {"Accept": headers["Accept"]}
            if authorization:
                unauthorized["Authorization"] = authorization
            assert client.post("/mcp", headers=unauthorized, json=initialize).status_code == 401
        assert not prepared
        response = client.post("/mcp", headers=headers, json=initialize)
        assert response.status_code == 200
        assert "result" in response.json()
        session_id = response.headers.get("mcp-session-id")
        assert bool(session_id) is (mode is not True)
        if session_id:
            headers["mcp-session-id"] = session_id
        headers["MCP-Protocol-Version"] = response.json()["result"]["protocolVersion"]
        listing = client.post("/mcp", headers=headers,
                              json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = listing.json()["result"]["tools"]
        assert [tool["name"] for tool in tools] == ["dependency_echo"]
        schema = tools[0]["inputSchema"]
        assert schema["properties"]["value"]["type"] == "integer"
        assert schema["required"] == ["value"]
        assert "ctx" not in schema["properties"]
        for fail in (False, True):
            call = client.post("/mcp", headers=headers, json={
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "dependency_echo", "arguments": {"value": 7, "fail": fail}},
            })
            assert call.status_code == 200
            result = call.json()["result"]
            assert result.get("isError", False) is fail
            if not fail:
                assert json.loads(result["content"][0]["text"]) == {"value": 7}
        stale_headers = dict(headers, **{"mcp-session-id": "unknown-session"})
        stale = client.post("/mcp", headers=stale_headers,
                            json={"jsonrpc": "2.0", "id": 4, "method": "tools/list"})
        assert (stale.status_code == 200) is (mode is True)
    assert len(prepared) == 2, "both calls must establish HANA audit context"
    assert [event["event_type"] for event in finished] == [
        "mcp.tool.invocation.succeeded", "mcp.tool.invocation.failed",
    ]
    invocations = [event for event in events if event["event_type"].startswith("mcp.tool.invocation.")]
    assert [event["event_type"] for event in invocations] == [
        "mcp.tool.invocation.started", "mcp.tool.invocation.succeeded",
        "mcp.tool.invocation.started", "mcp.tool.invocation.failed",
    ]
    for start, end in zip(invocations[::2], invocations[1::2]):
        assert start["correlation"]["invocation_id"] == end["correlation"]["invocation_id"]


@pytest.mark.skipif(not os.environ.get("HANA_USERKEY"), reason="HANA_USERKEY is required for the live HANA check")
@pytest.mark.parametrize("stateless", [False, True], ids=["stateful", "stateless"])
def test_http_fetches_real_hana_data_and_preserves_audit(monkeypatch, stateless):
    """Read SYS.DUMMY through the actual HTTP-to-tool-to-HANA boundary."""
    cc = toolkit_module.ConnectionContext(
        userkey=os.environ["HANA_USERKEY"], encrypt=True, sslValidateCertificate=True,
    )
    try:
        toolkit = toolkit_module.HANAMLToolkit(cc, used_tools=["fetch_data"])
        events = []
        monkeypatch.setattr(toolkit, "_emit_audit_event", lambda event: events.append(event))
        app = launch_http_app(monkeypatch, toolkit, stateless_http=stateless)
        headers = {"Authorization": "Bearer dependency-test-token",
                   "Accept": "application/json, text/event-stream"}
        with TestClient(app, base_url="http://127.0.0.1") as client:
            initialized = client.post("/mcp", headers=headers, json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                           "clientInfo": {"name": "live-hana-regression", "version": "1"}},
            })
            assert initialized.status_code == 200
            session_id = initialized.headers.get("mcp-session-id")
            if session_id:
                headers["mcp-session-id"] = session_id
            headers["MCP-Protocol-Version"] = initialized.json()["result"]["protocolVersion"]
            response = client.post("/mcp", headers=headers, json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "fetch_data",
                           "arguments": {"table_name": "DUMMY", "schema_name": "SYS", "top_n": 1}},
            })
            assert response.status_code == 200
            result = response.json()["result"]
            assert not result.get("isError", False), result
            text = "\n".join(block.get("text", "") for block in result["content"])
            assert "DUMMY" in text and "X" in text
        tool_events = [event for event in events if event["event_type"].startswith("mcp.tool.invocation.")]
        assert [event["event_type"] for event in tool_events] == [
            "mcp.tool.invocation.started", "mcp.tool.invocation.succeeded",
        ]
        assert tool_events[0]["correlation"]["invocation_id"] == tool_events[1]["correlation"]["invocation_id"]
        assert tool_events[1]["payload"]["tool_name"] == "fetch_data"
    finally:
        cc.close()
