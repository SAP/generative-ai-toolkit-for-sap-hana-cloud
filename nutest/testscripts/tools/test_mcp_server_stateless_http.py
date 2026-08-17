#!/usr/bin/env python3
"""
Unittest: HTTP transport MCP server ``stateless_http`` option.

Verifies:
1. **Default behavior is unchanged (stateful).** ``launch_mcp_server`` without
   ``stateless_http`` still issues an ``mcp-session-id`` on GET and requires
   subsequent POSTs to carry it. This is the pre-existing legacy behavior; the
   test exists to catch any accidental default flip.
2. **Stateless mode accepts fresh POSTs with no prior handshake / no session
   header.** This is the Joule Desktop failure mode: a client cached a stale
   session id and the server restarted. In stateless mode there is no session
   registry to reject the caller.
3. **Audit side effects are neutralized.** Across N stateless requests:
   - ``toolkit.mcp_session_metadata`` holds exactly ONE entry (the stable,
     process-lifetime session id), not N per-request phantoms.
   - ``mcp.session.started`` fires exactly ONCE, not per request.
   - The stable id has shape ``stateless-http-<pid>-<hex>``.

The HANA connection is opened via ``userkey`` (``HANA_USERKEY`` env, default
``RaysKey2``) so this can run against any team member's hdbuserstore entry.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
import threading
import unittest
from typing import Any, Optional

import requests


HANA_USERKEY = os.environ.get("HANA_USERKEY", "RaysKey3")


def _find_free_port(start: int = 8100, end: int = 8300) -> int:
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


def _post(base_url: str, payload: dict, headers: Optional[dict] = None) -> requests.Response:
    hdrs = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if headers:
        hdrs.update(headers)
    s = requests.Session()
    s.trust_env = False  # bypass corporate proxy for localhost
    return s.post(base_url, headers=hdrs, data=json.dumps(payload), timeout=15)


def _wait_ready(base_url: str, timeout: float = 8.0) -> None:
    """Poll the endpoint until it responds to a GET (any status)."""
    deadline = time.time() + timeout
    last_exc: Optional[Exception] = None
    while time.time() < deadline:
        try:
            requests.get(base_url, timeout=1.0)
            return
        except Exception as e:  # pragma: no cover — expected during boot
            last_exc = e
            time.sleep(0.15)
    raise RuntimeError(f"MCP server not ready at {base_url}: {last_exc}")


class TestMCPServerStatelessHTTP(unittest.TestCase):
    """End-to-end test of the ``stateless_http`` option."""

    @classmethod
    def setUpClass(cls):
        import certifi
        from hana_ml import dataframe
        cls.conn = dataframe.ConnectionContext(
            userkey=HANA_USERKEY,
            sslTrustStore=certifi.where(),
            sslValidateCertificate=False,
            encrypt=True,
        )

    @classmethod
    def tearDownClass(cls):
        try:
            cls.conn.connection.close()
        except Exception:
            pass

    # --- helpers ---------------------------------------------------------

    def _launch(self, *, stateless_http: bool) -> tuple[Any, int, str, list]:
        """Boot a fresh HTTP MCP server on a free port and return the toolkit,
        port, base_url, and the audit-events sink we install for verification.
        """
        from hana_ai.tools.toolkit import HANAMLToolkit

        tk = HANAMLToolkit(connection_context=self.conn, used_tools=["fetch_data"])

        # Capture audit events so we can assert "started only once".
        audit_events: list[dict] = []
        original_emit = tk._emit_audit_event

        def _capture(event: dict) -> None:
            audit_events.append(dict(event))
            return original_emit(event)

        tk._emit_audit_event = _capture  # type: ignore[method-assign]

        port = _find_free_port()
        base_url = f"http://127.0.0.1:{port}/mcp"

        tk.launch_mcp_server(
            transport="http",
            host="127.0.0.1",
            port=port,
            max_retries=3,
            stateless_http=stateless_http,
        )
        _wait_ready(base_url)
        return tk, port, base_url, audit_events

    def _stop(self, tk: Any, port: int) -> None:
        try:
            tk.stop_mcp_server(
                host="127.0.0.1", port=port, transport="http",
                force=True, timeout=3.0,
            )
        except Exception as e:
            print(f"warn: stop_mcp_server failed: {e}", file=sys.stderr)

    def _initialize_payload(self) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "stateless-test", "version": "1"},
            },
        }

    def _tools_list_payload(self) -> dict:
        return {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

    # --- (1) legacy default is still stateful -----------------------------

    def test_default_is_stateful(self):
        """A default (stateful) server MUST reject a POST that carries a
        never-issued ``mcp-session-id``. This is the pre-patch behavior that
        broke Joule Desktop on server restart; we assert we did NOT change it.
        """
        tk, port, base_url, _ = self._launch(stateless_http=False)
        try:
            # Fabricate a stale session id — as if a client cached one from a
            # prior server run. Stateful server should reject.
            resp = _post(
                base_url,
                self._initialize_payload(),
                headers={"mcp-session-id": "stale-from-previous-run"},
            )
            self.assertNotEqual(
                resp.status_code, 200,
                f"Stateful server accepted an unknown session id "
                f"(status={resp.status_code}, body={resp.text[:200]}); "
                f"default behavior may have flipped to stateless.",
            )
        finally:
            self._stop(tk, port)

    # --- (2) stateless mode accepts fresh POSTs with no handshake --------

    def test_stateless_accepts_unknown_session_id(self):
        """The Joule Desktop repro: client sends a POST with either NO
        session id or a stale one; stateless server must accept both and
        return a valid JSON-RPC result.
        """
        tk, port, base_url, _ = self._launch(stateless_http=True)
        try:
            # (a) no session header at all
            r1 = _post(base_url, self._initialize_payload())
            self.assertEqual(
                r1.status_code, 200,
                f"stateless server rejected header-less POST: "
                f"status={r1.status_code} body={r1.text[:200]}",
            )
            # (b) fabricated stale session id — server should ignore it
            r2 = _post(
                base_url,
                self._tools_list_payload(),
                headers={"mcp-session-id": "stale-from-previous-run"},
            )
            self.assertEqual(
                r2.status_code, 200,
                f"stateless server rejected stale session id: "
                f"status={r2.status_code} body={r2.text[:200]}",
            )
            # Response must actually be a real tools list, not an error.
            body = r2.text
            self.assertIn("fetch_data", body,
                          f"tools/list did not surface fetch_data: {body[:400]}")
        finally:
            self._stop(tk, port)

    # --- (3) audit side effects are neutralized --------------------------

    def test_stateless_audit_side_effects(self):
        """After many stateless requests:
        - metadata dict holds exactly one entry (the stable id)
        - ``mcp.session.started`` audit event fires exactly once
        - stable id has the expected shape
        """
        tk, port, base_url, events = self._launch(stateless_http=True)
        try:
            # Fire a burst of independent requests — each with a different
            # (or missing) session id, as a real stateless client would.
            for i in range(5):
                headers = None
                if i % 2 == 0:
                    headers = {"mcp-session-id": f"phantom-{i}"}
                _post(base_url, self._initialize_payload(), headers=headers)
                _post(base_url, self._tools_list_payload(), headers=headers)

            # Give the audit thread a moment to flush.
            time.sleep(0.3)

            # (a) metadata dict has exactly ONE stable id
            sids = list(tk.mcp_session_metadata.keys())
            self.assertEqual(
                len(sids), 1,
                f"stateless metadata dict has {len(sids)} entries "
                f"(expected 1): {sids}",
            )
            stable = sids[0]
            self.assertRegex(
                stable, r"^stateless-http-\d+-[0-9a-f]{8}$",
                f"stable session id has wrong shape: {stable!r}",
            )

            # (b) mcp.session.started emitted exactly ONCE
            starts = [e for e in events if e.get("event_type") == "mcp.session.started"]
            self.assertEqual(
                len(starts), 1,
                f"expected 1 mcp.session.started event, got {len(starts)}. "
                f"session ids: {[e.get('session', {}).get('mcp_session_id') for e in starts]}",
            )
            self.assertEqual(
                starts[0].get("session", {}).get("mcp_session_id"),
                stable,
            )
        finally:
            self._stop(tk, port)


if __name__ == "__main__":
    unittest.main(verbosity=2)
