"""Actual sockets, two clients, disk audit, and live HANA attribution."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from typing import Any, Type

import pytest


def _probe_tool(cc, snapshots):
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel

    class ProbeInput(BaseModel):
        marker: str
        fail: bool = False

    class AuditProbe(BaseTool):
        name: str = "dependency_audit_probe"
        description: str = "Read live HANA invocation attribution for acceptance testing."
        args_schema: Type[BaseModel] = ProbeInput
        connection_context: Any = None

        def _run(self, marker: str, fail: bool = False):
            assert re.fullmatch(r"DEPE2E_[A-F0-9]{12}_[AB]_[0-9]+", marker)
            cursor = self.connection_context.connection.cursor()
            try:
                # Unique SQL avoids HANA's documented first-plan-wins behavior.
                cursor.execute(
                    f"SELECT /* {marker} */ SESSION_CONTEXT('MCP_SESSION_ID'), "
                    "SESSION_CONTEXT('INVOCATION_ID'), SESSION_CONTEXT('CLIENT_DECLARED_NAME'), "
                    "SESSION_CONTEXT('TOOL_NAME'), SESSION_CONTEXT('APPLICATIONSOURCE') FROM DUMMY"
                )
                row = dict(zip(("session", "invocation", "client", "tool", "appsource"), cursor.fetchone()))
                snapshots[marker] = row
            finally:
                cursor.close()
            if fail:
                raise RuntimeError("Deliberate acceptance failure: " + marker)
            return json.dumps(dict(marker=marker, **row))

    return AuditProbe(connection_context=cc)


@pytest.mark.parametrize("stateless", [False, True], ids=["stateful", "stateless"])
def test_two_clients_preserve_real_hana_and_disk_audit(hana_case, tmp_path, stateless):
    _, prefix = hana_case
    # Isolate the server's background threads even when its public stop API
    # cannot join them. Exiting this owned process guarantees port cleanup.
    log_path = tmp_path / "scenario.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), prefix, str(tmp_path), str(int(stateless))],
            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
            stdout=log, stderr=subprocess.STDOUT, timeout=600, check=False,
        )
    assert completed.returncode == 0, f"MCP acceptance failed; see {log_path}"


def _run_scenario(cc, prefix, tmp_path, stateless):
    import requests
    from hana_ai.tools.toolkit import HANAMLToolkit
    from hana_ai.tools.hana_ml_tools.utility import fetch_hana_mcp_audit_view, parse_appsource_pack
    audit_path = tmp_path / "audit.jsonl"
    snapshots = {}
    toolkit = HANAMLToolkit(cc, used_tools=["fetch_data"], audit_log_path=str(audit_path))
    toolkit.add_custom_tool(_probe_tool(cc, snapshots))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/mcp"
    token = prefix + "-test-token"
    clients = []
    started = False
    try:
        toolkit.launch_mcp_server(transport="http", host="127.0.0.1", port=port,
                                  auth_token=token, max_retries=1, stateless_http=stateless)
        started = True
        for label in ("A", "B"):
            client = requests.Session()
            client.trust_env = False
            clients.append(client)
            client.headers.update({
                "Accept": "application/json, text/event-stream",
                "Authorization": "Bearer " + token,
                "x-mcp-client-name": prefix + "_" + label,
            })
        deadline = time.monotonic() + 15
        while True:
            try:
                clients[0].get(url, timeout=1)
                break
            except requests.ConnectionError:
                assert time.monotonic() < deadline, "MCP server did not start"
                time.sleep(0.1)

        def post(client, method, params, request_id=1):
            response = client.post(url, json=dict(jsonrpc="2.0", id=request_id, method=method, params=params), timeout=90)
            assert response.status_code == 200, response.text
            payload = response.json()
            assert "error" not in payload, payload
            return response, payload["result"]

        sessions = []
        for client, label in zip(clients, ("A", "B")):
            response, initialized = post(client, "initialize", dict(
                protocolVersion="2025-06-18", capabilities={},
                clientInfo=dict(name=prefix + "_" + label, version="1")))
            session = response.headers.get("mcp-session-id")
            assert bool(session) is (not stateless)
            sessions.append(session)
            if session:
                client.headers["mcp-session-id"] = session
            client.headers["MCP-Protocol-Version"] = initialized["protocolVersion"]
            # Also exercise an unchanged production business tool.
            _, result = post(client, "tools/call", dict(name="fetch_data", arguments=dict(
                table_name="DUMMY", schema_name="SYS", top_n=1)))
            assert not result.get("isError", False), result
            assert "X" in str(result) and "DUMMY" in str(result)
        if not stateless:
            assert sessions[0] != sessions[1]

        def calls(client, label):
            responses = []
            for number, fail in enumerate((False, True, False)):
                marker = f"{prefix}_{label}_{number}"
                _, result = post(client, "tools/call", dict(name="dependency_audit_probe",
                                 arguments=dict(marker=marker, fail=fail)), number + 10)
                responses.append((marker, fail, result))
            return responses

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(calls, client, label) for client, label in zip(clients, ("A", "B"))]
            responses = [response for future in futures for response in future.result(timeout=300)]

        # Preserve independent database evidence before making assertions, so
        # an attribution regression does not prevent readback of the other calls.
        rows = fetch_hana_mcp_audit_view(cc.connection, tool_name="dependency_audit_probe",
                                        since_seconds=600, limit=1000)
        rows.to_json(tmp_path / "hana-audit.json", orient="records", indent=2, date_format="iso")
        # The public audit view intentionally omits SQL text. Query raw plan
        # cache separately to associate each unique probe marker with its pack.
        cursor = cc.connection.cursor()
        try:
            cursor.execute("SELECT STATEMENT_STRING, APPLICATION_SOURCE FROM M_SQL_PLAN_CACHE "
                           "WHERE STATEMENT_STRING LIKE ? AND APPLICATION_SOURCE LIKE ?",
                           ("%/* " + prefix + "_%", "mcp=hana-ai/%"))
            raw_plans = [dict(statement=str(row[0]), pack=parse_appsource_pack(str(row[1])))
                         for row in cursor.fetchall()]
        finally:
            cursor.close()
        (tmp_path / "hana-plans.json").write_text(json.dumps(raw_plans, indent=2), encoding="utf-8")
        (tmp_path / "responses.json").write_text(json.dumps(responses, indent=2), encoding="utf-8")
        for marker, fail, result in responses:
            assert result.get("isError", False) is fail, result
            if not fail:
                payload = json.loads(result["content"][0]["text"])
                assert payload["marker"] == marker
                assert payload["client"] == marker.rsplit("_", 1)[0], (marker, payload)

        # Read the actual disk sink: no replacement of _emit_audit_event or HANA writers.
        events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        by_invocation = defaultdict(list)
        for event in events:
            if event["event_type"].startswith("mcp.tool.invocation."):
                by_invocation[event["correlation"]["invocation_id"]].append(event)
        assert len(by_invocation) == 8  # 2 fetch_data + 6 probe invocations
        expected_markers = {f"{prefix}_{label}_{n}" for label in ("A", "B") for n in range(3)}
        assert set(snapshots) == expected_markers
        seen = set()
        for invocation, pair in by_invocation.items():
            assert len(pair) == 2, pair
            begin, end = pair
            assert begin["event_type"] == "mcp.tool.invocation.started"
            assert begin["session"]["mcp_session_id"] == end["session"]["mcp_session_id"]
            assert begin["correlation"] == end["correlation"]
            if begin["payload"]["tool_name"] == "fetch_data":
                assert end["event_type"] == "mcp.tool.invocation.succeeded"
                continue
            args = begin["payload"]["input_redacted"]
            marker = args["marker"]
            seen.add(marker)
            expected_client = marker.rsplit("_", 1)[0]
            outcome = "failed" if args["fail"] else "succeeded"
            assert end["event_type"] == "mcp.tool.invocation." + outcome
            snapshot = snapshots[marker]
            assert snapshot["invocation"] == invocation
            assert snapshot["client"] == expected_client
            assert snapshot["tool"] == "dependency_audit_probe"
            assert snapshot["session"] == begin["session"]["mcp_session_id"]
            assert begin["session"]["client_declared_name"] == expected_client
            assert end["session"]["client_declared_name"] == expected_client
            pack = parse_appsource_pack(snapshot["appsource"])
            assert pack["inv"] == invocation and pack["sess"] == snapshot["session"]
        assert seen == expected_markers
        assert len({row["session"] for row in snapshots.values()}) == (1 if stateless else 2)
        session_starts = [event for event in events if event["event_type"] == "mcp.session.started"]
        assert len(session_starts) == (1 if stateless else 2)

        # Independent database readback of the unique probe statements. This is
        # the documented plan-cache audit view, not the optional DBA AUDIT_LOG.
        for marker, snapshot in snapshots.items():
            matched = [plan["pack"] for plan in raw_plans if marker in plan["statement"]]
            assert matched, f"No HANA audit evidence for {marker}"
            assert {pack["inv"] for pack in matched} == {snapshot["invocation"]}
            assert {pack["sess"] for pack in matched} == {snapshot["session"]}
            assert snapshot["invocation"] in set(rows["INVOCATION_ID"])
    finally:
        (tmp_path / "hana-context-snapshots.json").write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
        for client in clients:
            client.close()
        if started:
            stopped = toolkit.stop_mcp_server(host="127.0.0.1", port=port, transport="http", timeout=5)
            (tmp_path / "shutdown.json").write_text(json.dumps(dict(
                graceful_stop=stopped, port=port, isolated_process=True)), encoding="utf-8")


if __name__ == "__main__":
    import importlib.metadata
    from hana_ai.tools import toolkit as toolkit_module
    from hana_ml import dataframe
    Path(sys.argv[2], "runtime.json").write_text(json.dumps(dict(
        python=sys.executable, toolkit_source=toolkit_module.__file__,
        fastmcp=importlib.metadata.version("fastmcp"),
    ), indent=2), encoding="utf-8")
    connection = dataframe.ConnectionContext(userkey=os.environ["HANA_USERKEY"],
                                             encrypt=True, sslValidateCertificate=True)
    try:
        _run_scenario(connection, sys.argv[1], Path(sys.argv[2]), bool(int(sys.argv[3])))
    finally:
        connection.close()
