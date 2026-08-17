#!/usr/bin/env python3
"""
Stress test: MCP audit event emission under concurrent load.

Bombards the HTTP MCP server with many concurrent workers, each doing
initialize + tools/list + tools/call cycles, then inspects the captured
audit event stream to verify correctness invariants that must hold in
BOTH stateful and stateless modes:

Stateful invariants (per unique client session id):
- exactly 1 ``mcp.session.started`` event
- every ``mcp.tool.invocation.started`` has a matching ``.succeeded`` or
  ``.failed`` sibling with the same ``invocation_id``
- no orphan events, no duplicate ``invocation_id``s
- total tool.started count == WORKERS × ROUNDS (no lost events)

Stateless invariants (across the WHOLE load — this is the code path we
added and where locking/dedup can regress):
- exactly 1 stable session id in ``toolkit.mcp_session_metadata``
- exactly 1 ``mcp.session.started`` event, regardless of concurrency
- every tool.invocation event carries the stable session id — no
  transport-level per-request id leaks through
- tool.invocation pairs still balance (no orphans, no both succeeded+failed)
- the once-only session-start guard survives a thundering-herd of concurrent
  initialize requests (specifically stresses the ``started_lock`` in the
  middleware — separate test)

Uses HANA userkey ``HANA_USERKEY`` (default ``RaysKey3``) with certifi
trust store, encrypted, no cert validation — matching the notebook
conventions in this repo.

The audit path itself normally writes to HANA (SESSION_VARIABLES,
setclientinfo, beacon SQL), but those share a single connection cursor
that serialises every concurrent call. Since we're stressing the
middleware's locking + event bookkeeping — NOT the HANA-side channel —
we stub the HANA-write helpers to no-op. ``_emit_audit_event`` (the
actual event stream) is untouched.

Load parameters (env-tunable):
- ``STRESS_WORKERS``  (default 32): concurrent worker threads
- ``STRESS_ROUNDS``   (default 8):  init+list+call cycles per worker
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import socket
import sys
import threading
import time
import unittest
from collections import Counter, defaultdict
from typing import Any, Optional

import requests


HANA_USERKEY = os.environ.get("HANA_USERKEY", "RaysKey3")
WORKERS = int(os.environ.get("STRESS_WORKERS", "32"))
ROUNDS = int(os.environ.get("STRESS_ROUNDS", "8"))


def _find_free_port(start: int = 8300, end: int = 8500) -> int:
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


def _wait_ready(base_url: str, timeout: float = 8.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(base_url, timeout=1.0)
            return
        except Exception:
            time.sleep(0.15)
    raise RuntimeError(f"server not ready: {base_url}")


class _Client:
    """Minimal HTTP MCP client.

    - Stateful mode: obtains a session id via GET, echoes it on every
      subsequent POST.
    - Stateless mode: never carries a session id; every POST is bare —
      this mirrors how a Joule-Desktop-style client with a stale cached
      session behaves after a server restart.
    """

    def __init__(self, base_url: str, *, stateless: bool):
        self.base_url = base_url
        self.stateless = stateless
        self.session_id: Optional[str] = None
        self.s = requests.Session()
        self.s.trust_env = False

    def _headers(self) -> dict:
        h = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self.session_id and not self.stateless:
            h["mcp-session-id"] = self.session_id
        return h

    def obtain_session(self) -> None:
        if self.stateless:
            return
        r = self.s.get(self.base_url, headers=self._headers(), timeout=10)
        sid = r.headers.get("mcp-session-id")
        if not sid:
            r2 = self.s.get(self.base_url,
                            headers={**self._headers(),
                                     "Accept": "text/event-stream"},
                            timeout=10)
            sid = r2.headers.get("mcp-session-id")
        self.session_id = sid

    def _rpc(self, method: str, params: dict, rid: int) -> dict:
        r = self.s.post(
            self.base_url,
            headers=self._headers(),
            data=json.dumps({
                "jsonrpc": "2.0", "id": rid, "method": method, "params": params,
            }),
            timeout=30,
        )
        try:
            return r.json()
        except Exception:
            return {"_raw_status": r.status_code, "_raw_text": r.text[:500]}

    def initialize(self, rid: int) -> dict:
        return self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "stress", "version": "1"},
        }, rid)

    def tools_list(self, rid: int) -> dict:
        return self._rpc("tools/list", {}, rid)

    def tools_call(self, name: str, arguments: dict, rid: int) -> dict:
        return self._rpc("tools/call",
                         {"name": name, "arguments": arguments}, rid)


def _boot_server(*, stateless: bool):
    """Boot a fresh HTTP MCP server on a free port with only fetch_data
    registered and an audit-event sink installed.

    ``fetch_data._run`` is monkey-patched to a ~0ms no-op so we can stress
    the audit middleware itself without serialising on the (single-threaded)
    HANA connection. The audit path treats every ``tools/call`` identically
    regardless of what the tool actually does — the middleware runs before
    and after the tool body — so a fast in-memory tool exercises exactly the
    same emit paths as a real fetch, just N× faster.

    The HANA-side audit writers (session variables, setclientinfo, beacon
    SQL) are also stubbed for the same serialisation reason. We keep
    ``_emit_audit_event`` intact — the event stream itself is what we're
    validating.

    Returns (toolkit, port, base_url, events, events_lock).
    """
    import certifi
    from hana_ml import dataframe
    from hana_ai.tools.toolkit import HANAMLToolkit

    cc = dataframe.ConnectionContext(
        userkey=HANA_USERKEY,
        sslTrustStore=certifi.where(),
        sslValidateCertificate=False,
        encrypt=True,
    )
    tk = HANAMLToolkit(connection_context=cc, used_tools=["fetch_data"])

    for tool in tk.get_tools():
        if getattr(tool, "name", None) == "fetch_data":
            tool._run = lambda **kwargs: "ok"  # type: ignore[assignment]
            break

    tk._ensure_hana_execution_context = lambda *a, **kw: None  # type: ignore[assignment]
    tk._update_hana_session_event = lambda *a, **kw: None  # type: ignore[assignment]
    tk._set_hana_session_variables = lambda *a, **kw: None  # type: ignore[assignment]
    tk._fetch_hana_identity = lambda *a, **kw: {}  # type: ignore[assignment]
    tk._emit_beacon_sql = lambda *a, **kw: None  # type: ignore[assignment]

    events: list[dict] = []
    events_lock = threading.Lock()
    original_emit = tk._emit_audit_event

    def _capture(event: dict) -> None:
        with events_lock:
            events.append(dict(event))
        return original_emit(event)

    tk._emit_audit_event = _capture  # type: ignore[method-assign]

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}/mcp"
    tk.launch_mcp_server(
        transport="http",
        host="127.0.0.1",
        port=port,
        max_retries=3,
        stateless_http=stateless,
    )
    _wait_ready(base_url)
    return tk, port, base_url, events, events_lock


def _stop_server(tk: Any, port: int) -> None:
    try:
        tk.stop_mcp_server(host="127.0.0.1", port=port,
                           transport="http", force=True, timeout=3.0)
    except Exception as e:
        print(f"warn: stop_mcp_server: {e}", file=sys.stderr)
    try:
        tk.connection_context.connection.close()
    except Exception:
        pass


def _bucket_events(events: list[dict]) -> dict:
    """Group events by type, and pair up tool invocation started/finished."""
    by_type: Counter = Counter()
    by_session_started: Counter = Counter()
    invocations: dict[str, dict] = defaultdict(dict)  # inv_id -> {phase: event}

    for e in events:
        et = e.get("event_type")
        by_type[et] += 1
        sid = e.get("session", {}).get("mcp_session_id")
        if et == "mcp.session.started":
            by_session_started[sid] += 1
        if et and et.startswith("mcp.tool.invocation."):
            inv_id = e.get("correlation", {}).get("invocation_id")
            phase = et.rsplit(".", 1)[-1]
            invocations[inv_id][phase] = e
    return {
        "by_type": dict(by_type),
        "by_session_started": dict(by_session_started),
        "invocations": invocations,
    }


class TestAuditStress(unittest.TestCase):
    """Stress the audit path under concurrent load."""

    def _run_workers(self, base_url: str, stateless: bool):
        """Spin up WORKERS threads; each does ROUNDS init+list+call cycles.
        Returns (successful_call_count, seen_session_ids)."""
        successful_calls = 0
        seen_sessions: set[str] = set()
        counter_lock = threading.Lock()
        rid_counter = {"n": 0}
        rid_lock = threading.Lock()

        def _next_rid() -> int:
            with rid_lock:
                rid_counter["n"] += 1
                return rid_counter["n"]

        def _worker(_worker_idx: int) -> tuple[int, Optional[str]]:
            client = _Client(base_url, stateless=stateless)
            client.obtain_session()
            successes = 0
            for _ in range(ROUNDS):
                client.initialize(_next_rid())
                client.tools_list(_next_rid())
                resp = client.tools_call(
                    "fetch_data",
                    {"table_name": "DUMMY", "schema_name": "SYS",
                     "top_n": 1},
                    _next_rid(),
                )
                if resp.get("result") is not None:
                    successes += 1
            return successes, client.session_id

        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(_worker, i) for i in range(WORKERS)]
            for f in concurrent.futures.as_completed(futures):
                s, sid = f.result()
                with counter_lock:
                    successful_calls += s
                    if sid:
                        seen_sessions.add(sid)
        time.sleep(0.5)  # let the audit thread drain
        return successful_calls, seen_sessions

    # --- stateful stress -------------------------------------------------

    def test_stateful_concurrent_load(self):
        """WORKERS×ROUNDS concurrent tool calls with stateful sessions:
        - each session sees exactly 1 mcp.session.started
        - every tool.invocation.started has a matching finish event
        - no invocation both succeeds and fails
        - total tool.started == WORKERS × ROUNDS (no lost events)
        """
        tk, port, base_url, events, events_lock = _boot_server(stateless=False)
        try:
            successes, seen_sessions = self._run_workers(
                base_url, stateless=False,
            )
            with events_lock:
                snapshot = list(events)
            b = _bucket_events(snapshot)

            for sid, count in b["by_session_started"].items():
                self.assertEqual(
                    count, 1,
                    f"session {sid!r} got {count} session.started events "
                    f"(should be 1). Duplicate emission = audit corruption.",
                )

            missing = seen_sessions - set(b["by_session_started"].keys())
            self.assertFalse(
                missing,
                f"clients had session ids that never got a session.started "
                f"event: {missing}",
            )

            orphans, both = [], []
            for inv_id, phases in b["invocations"].items():
                has_start = "started" in phases
                has_ok = "succeeded" in phases
                has_err = "failed" in phases
                if has_start and not (has_ok or has_err):
                    orphans.append(inv_id)
                if has_ok and has_err:
                    both.append(inv_id)
            self.assertFalse(
                orphans,
                f"{len(orphans)} tool invocations started but never "
                f"terminated (first: {orphans[:3]})",
            )
            self.assertFalse(
                both,
                f"{len(both)} tool invocations emitted BOTH succeeded and "
                f"failed (first: {both[:3]})",
            )

            total_starts = b["by_type"].get("mcp.tool.invocation.started", 0)
            self.assertEqual(
                total_starts, WORKERS * ROUNDS,
                f"expected {WORKERS * ROUNDS} tool.started events, "
                f"got {total_starts}. Audit lost events under load.",
            )
            print(f"[stateful] workers={WORKERS} rounds={ROUNDS} "
                  f"sessions={len(b['by_session_started'])} "
                  f"tool_calls={total_starts} successes={successes}",
                  file=sys.stderr)
        finally:
            _stop_server(tk, port)

    # --- stateless stress ------------------------------------------------

    def test_stateless_concurrent_load(self):
        """Same load as the stateful test, but every request is bare (no
        mcp-session-id). Extra invariants specific to the stateless path
        we added:
        - exactly 1 stable session id in metadata (no phantom leak)
        - exactly 1 mcp.session.started event across the whole load
        - every tool.invocation event carries the stable session id
        """
        tk, port, base_url, events, events_lock = _boot_server(stateless=True)
        try:
            successes, _ = self._run_workers(base_url, stateless=True)
            with events_lock:
                snapshot = list(events)
            b = _bucket_events(snapshot)

            sids = list(tk.mcp_session_metadata.keys())
            self.assertEqual(
                len(sids), 1,
                f"stateless metadata leaked: {len(sids)} entries: {sids[:5]}",
            )
            stable = sids[0]
            self.assertRegex(
                stable, r"^stateless-http-\d+-[0-9a-f]{8}$",
                f"stable session id has wrong shape: {stable!r}",
            )

            starts_for_stable = b["by_session_started"].get(stable, 0)
            other_starts = {sid: n for sid, n in b["by_session_started"].items()
                            if sid != stable}
            self.assertEqual(
                starts_for_stable, 1,
                f"expected 1 session.started under stable id {stable!r}, "
                f"got {starts_for_stable}. started_lock race?",
            )
            self.assertFalse(
                other_starts,
                f"stateless mode emitted session.started under non-stable "
                f"session ids: {other_starts}",
            )

            orphans = [i for i, p in b["invocations"].items()
                       if "started" in p
                       and not (("succeeded" in p) or ("failed" in p))]
            self.assertFalse(orphans,
                             f"orphan invocations under stateless load: "
                             f"{len(orphans)}")

            leaked = [e for e in snapshot
                      if e.get("event_type", "").startswith("mcp.tool.invocation.")
                      and e.get("session", {}).get("mcp_session_id") != stable]
            self.assertFalse(
                leaked,
                f"{len(leaked)} tool events leaked a non-stable session id "
                f"(first: {leaked[0].get('session', {}) if leaked else None})",
            )

            total_starts = b["by_type"].get("mcp.tool.invocation.started", 0)
            self.assertEqual(
                total_starts, WORKERS * ROUNDS,
                f"expected {WORKERS * ROUNDS} tool.started events under "
                f"stateless load, got {total_starts}. Audit lost events.",
            )
            print(f"[stateless] workers={WORKERS} rounds={ROUNDS} "
                  f"tool_calls={total_starts} successes={successes} "
                  f"stable_id={stable}",
                  file=sys.stderr)
        finally:
            _stop_server(tk, port)

    # --- thundering-herd initialize (stateless once-only guard) ----------

    def test_stateless_thundering_herd_initialize(self):
        """Pathological once-only-guard test: N workers each blast M
        initialize requests with a shared barrier so they all fire
        together. If ``started_lock`` is missing or narrower than the
        set-membership check, we would see > 1 session.started."""
        tk, port, base_url, events, events_lock = _boot_server(stateless=True)
        try:
            N = max(WORKERS, 16)
            M = 20
            barrier = threading.Barrier(N)

            def _blast() -> None:
                client = _Client(base_url, stateless=True)
                barrier.wait()
                for i in range(M):
                    client.initialize(rid=1000 + i)

            with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
                list(pool.map(lambda _: _blast(), range(N)))

            time.sleep(0.4)
            with events_lock:
                snapshot = list(events)
            starts = [e for e in snapshot
                      if e.get("event_type") == "mcp.session.started"]
            self.assertEqual(
                len(starts), 1,
                f"thundering-herd init produced {len(starts)} "
                f"session.started events (expected 1). "
                f"session ids: "
                f"{[e.get('session', {}).get('mcp_session_id') for e in starts]}",
            )
            print(f"[herd] workers={N} inits/worker={M} "
                  f"total_inits={N*M} session_started={len(starts)}",
                  file=sys.stderr)
        finally:
            _stop_server(tk, port)


if __name__ == "__main__":
    unittest.main(verbosity=2)
