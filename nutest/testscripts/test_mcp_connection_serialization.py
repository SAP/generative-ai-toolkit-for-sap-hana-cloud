"""Protect shared HANA attribution through completion, errors, and cancellation."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import Mock

import pytest
from starlette.testclient import TestClient

from hana_ai.tools import toolkit as toolkit_module
from test_mcp_dependency_compatibility import EchoTool, launch_http_app


class ObservedLock:
    """Signal an actual contention attempt instead of relying on timing."""

    def __init__(self, on_contention):
        self.lock = threading.Lock()
        self.on_contention = on_contention

    def acquire(self, blocking=False):
        acquired = self.lock.acquire(blocking=blocking)
        if not acquired:
            self.on_contention()
        return acquired

    def release(self):
        self.lock.release()


def make_toolkit():
    return toolkit_module.HANAMLToolkit(
        connection_context=Mock(spec=toolkit_module.ConnectionContext), used_tools=[],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True], ids=["success", "tool-error"])
async def test_cancelled_request_keeps_connection_until_worker_and_audit_finish(fail):
    toolkit = make_toolkit()
    loop = asyncio.get_running_loop()
    worker_started = asyncio.Event()
    contended = asyncio.Event()
    audit_started = asyncio.Event()
    finish_audit = asyncio.Event()
    finish_worker = threading.Event()
    sequence = []
    toolkit._mcp_request_lock = ObservedLock(contended.set)

    def worker():
        sequence.append("worker-start")
        loop.call_soon_threadsafe(worker_started.set)
        assert finish_worker.wait(10), "test did not release worker"
        sequence.append("worker-end")
        if fail:
            raise ValueError("expected worker error")

    async def first_handler(_):
        try:
            await asyncio.to_thread(worker)
        finally:
            audit_started.set()
            await finish_audit.wait()
            sequence.append("audit-end")

    async def second_handler(_):
        sequence.append("next-request")
        return "next-result"

    first = asyncio.create_task(toolkit._run_mcp_request(first_handler, None))
    second = None
    try:
        await asyncio.wait_for(worker_started.wait(), 5)
        first.cancel()
        second = asyncio.create_task(toolkit._run_mcp_request(second_handler, None))
        await asyncio.wait_for(contended.wait(), 5)
        assert sequence == ["worker-start"]
        assert not first.done() and not second.done()
        finish_worker.set()
        await asyncio.wait_for(audit_started.wait(), 5)
        # Repeated cancellation must not abandon the completion audit either.
        first.cancel()
        contended.clear()
        await asyncio.wait_for(contended.wait(), 5)
        assert sequence == ["worker-start", "worker-end"]
        assert not second.done()
        finish_audit.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, 5)
        assert await asyncio.wait_for(second, 5) == "next-result"
        assert sequence == ["worker-start", "worker-end", "audit-end", "next-request"]
    finally:
        finish_worker.set()
        finish_audit.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_release_active_connection():
    toolkit = make_toolkit()
    contended = asyncio.Event()
    toolkit._mcp_request_lock = ObservedLock(contended.set)
    toolkit._mcp_request_lock.acquire()
    entered = []

    async def handler(_):
        entered.append(True)

    waiter = asyncio.create_task(toolkit._run_mcp_request(handler, None))
    try:
        await asyncio.wait_for(contended.wait(), 5)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert toolkit._mcp_request_lock.lock.locked()
        assert not entered
    finally:
        toolkit._mcp_request_lock.release()
    await asyncio.wait_for(toolkit._run_mcp_request(handler, None), 5)
    assert entered == [True]


def test_servers_on_separate_event_loops_share_connection_guard():
    toolkit = make_toolkit()
    entered = threading.Event()
    contended = threading.Event()
    release = threading.Event()
    sequence = []
    toolkit._mcp_request_lock = ObservedLock(contended.set)

    async def first_handler(_):
        sequence.append("first-start")
        entered.set()
        assert await asyncio.to_thread(release.wait, 10)
        sequence.append("first-end")
        raise ValueError("expected error")

    async def second_handler(_):
        sequence.append("second")

    def run(handler):
        asyncio.run(toolkit._run_mcp_request(handler, None))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, first_handler)
        try:
            assert entered.wait(5)
            second = pool.submit(run, second_handler)
            assert contended.wait(5)
            assert sequence == ["first-start"]
        finally:
            release.set()
        with pytest.raises(ValueError, match="expected error"):
            first.result(timeout=5)
        second.result(timeout=5)
    assert sequence == ["first-start", "first-end", "second"]


@pytest.mark.parametrize("following_request", ["initialize", "reload"])
def test_http_initialize_and_reload_wait_for_tool_completion(monkeypatch, following_request):
    toolkit = make_toolkit()
    original_connection = toolkit.connection_context
    replacement_connection = Mock(spec=toolkit_module.ConnectionContext)
    entered = threading.Event()
    release = threading.Event()
    contended = threading.Event()
    sequence = []
    toolkit._mcp_request_lock = ObservedLock(contended.set)
    monkeypatch.setattr(toolkit, "_fetch_hana_identity", lambda: {})
    monkeypatch.setattr(toolkit, "_set_hana_session_variables", lambda *args, **kwargs: None)
    monkeypatch.setattr(toolkit, "_ensure_hana_execution_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(toolkit, "_update_hana_session_event", lambda event: None)
    monkeypatch.setattr(toolkit, "_emit_audit_event",
                        lambda event: sequence.append(event["event_type"]))

    def blocked_echo(_self, value, fail=False):
        entered.set()
        assert release.wait(10)
        assert toolkit.connection_context is original_connection
        return str(value)

    def new_connection(**_kwargs):
        assert sequence[-1] == "mcp.tool.invocation.started"  # The admin call's start.
        assert "mcp.tool.invocation.succeeded" in sequence  # Previous tool completed.
        sequence.append("connection-replaced")
        return replacement_connection

    monkeypatch.setattr(EchoTool, "_run", blocked_echo)
    toolkit.add_custom_tool(EchoTool())
    app = launch_http_app(monkeypatch, toolkit, stateless_http=True, runtime_update=True)
    monkeypatch.setattr(toolkit_module, "_build_cc_params_from_env",
                        lambda: {"userkey": "test-only", "encrypt": True, "sslValidateCertificate": True})
    monkeypatch.setattr(toolkit_module, "ConnectionContext", new_connection)
    headers = {"Accept": "application/json, text/event-stream",
               "Authorization": "Bearer dependency-test-token"}
    tool_request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "dependency_echo", "arguments": {"value": 1}}}
    if following_request == "initialize":
        next_request = {"jsonrpc": "2.0", "id": 2, "method": "initialize",
                        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                   "clientInfo": {"name": "next-client", "version": "1"}}}
    else:
        next_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {"name": "admin_reload_connection_context_from_env",
                                   "arguments": {"test_connection": False}}}
    with TestClient(app, base_url="http://127.0.0.1") as client, ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.post, "/mcp", headers=headers, json=tool_request)
        try:
            assert entered.wait(5)
            second = pool.submit(client.post, "/mcp", headers=headers, json=next_request)
            assert contended.wait(5)
            assert not second.done()
            assert toolkit.connection_context is original_connection
        finally:
            release.set()
        for request in (first, second):
            response = request.result(timeout=10)
            assert response.status_code == 200
            assert not response.json()["result"].get("isError", False)
        if following_request == "reload":
            assert toolkit.connection_context is not original_connection
            assert sequence.count("connection-replaced") == 1


@pytest.mark.parametrize("next_identity", [{}, {"x-mcp-client-name": "second-client"}],
                         ids=["anonymous", "partial-identity"])
def test_stateless_request_clears_previous_client_identity(monkeypatch, next_identity):
    toolkit = make_toolkit()
    client_info = {}
    snapshots = []
    events = []
    connection = Mock()
    connection.setclientinfo.side_effect = lambda key, value: client_info.update({key: value})
    toolkit.connection_context.connection = connection
    monkeypatch.setattr(toolkit, "_fetch_hana_identity", lambda: {})
    monkeypatch.setattr(toolkit, "_set_hana_session_variables", lambda *args, **kwargs: None)
    monkeypatch.setattr(toolkit, "_emit_audit_event", events.append)

    def snapshot(_self, value, fail=False):
        snapshots.append(client_info.copy())
        return str(value)

    monkeypatch.setattr(EchoTool, "_run", snapshot)
    toolkit.add_custom_tool(EchoTool())
    app = launch_http_app(monkeypatch, toolkit, stateless_http=True)
    headers = {"Accept": "application/json, text/event-stream",
               "Authorization": "Bearer dependency-test-token"}
    first_identity = {
        "x-mcp-client-name": "first-client", "x-mcp-client-id": "first-user",
        "x-mcp-client-version": "first-version", "x-ai-agent-name": "first-agent",
        "x-ai-model-name": "first-model", "x-ai-model-version": "first-model-version",
    }
    with TestClient(app, base_url="http://127.0.0.1") as client:
        for index, identity in enumerate((first_identity, next_identity), 1):
            response = client.post("/mcp", headers={**headers, **identity}, json={
                "jsonrpc": "2.0", "id": index, "method": "tools/call",
                "params": {"name": "dependency_echo", "arguments": {"value": index}},
            })
            assert response.status_code == 200
            assert not response.json()["result"].get("isError", False)
    assert len(snapshots) == 2
    assert snapshots[0]["APPLICATIONUSER"] == "first-user"
    assert snapshots[1]["APPLICATIONUSER"] == next_identity.get("x-mcp-client-name", "")
    assert not any("first-" in str(value) for value in snapshots[1].values())
    starts = [event for event in events if event["event_type"] == "mcp.tool.invocation.started"]
    assert len(starts) == 2
    assert starts[0]["session"]["mcp_session_id"] == starts[1]["session"]["mcp_session_id"]
    for key, value in starts[1]["session"].items():
        if key.startswith("client_declared_"):
            expected = next_identity.get("x-mcp-client-name") if key == "client_declared_name" else None
            assert value == expected, key
