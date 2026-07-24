"""Unit tests for fetch_hana_mcp_audit_view row-mapping logic.

The function issues one SELECT and parses each row's APPLICATION_SOURCE with
``parse_appsource_pack``. We stub the cursor so we can assert on the
resulting DataFrame shape without needing a HANA connection.
"""

import unittest

from hana_ai.tools.hana_ml_tools.utility import (
    HANA_MCP_AUDIT_VIEW_COLUMNS,
    MCP_BEACON_SQL_MARKER,
    build_appsource_pack,
    fetch_hana_mcp_audit_view,
)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []      # (sql, params)

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return list(self._rows)

    def close(self):
        pass


class _FakeConnection:
    def __init__(self, rows):
        self._rows = rows
        self.cursors: list[_FakeCursor] = []

    def cursor(self):
        cur = _FakeCursor(self._rows)
        self.cursors.append(cur)
        return cur


def _pc_row(pack, *, user_name="DBTECH", session_user="DBTECH",
            app_name="test-cli", stmt_hash="hash-1", exec_count=3,
            ts="2026-07-24T10:00:00", stmt_string="SELECT 1 FROM T"):
    """Fake M_SQL_PLAN_CACHE row in the exact column order the SELECT emits."""
    return (ts, user_name, session_user, app_name, stmt_hash, exec_count,
            pack, stmt_string)


class TestFetchHanaMcpAuditView(unittest.TestCase):

    def test_maps_pack_to_view_columns(self):
        pack = build_appsource_pack(
            mcp_version="0.3",
            mcp_session_id="s" * 32,
            client_declared_agent_name="langchain",
            client_declared_model_name="claude-opus-4-8",
            client_declared_name="cli-x",
            client_ip="10.20.30.40",
            tool_name="fetch_data",
            invocation_id="i" * 16,
            hana_correlation_id="c" * 16,
        )
        conn = _FakeConnection([
            _pc_row(pack, user_name="I516820", app_name="cli-x"),
        ])
        df = fetch_hana_mcp_audit_view(conn)

        self.assertEqual(list(df.columns), HANA_MCP_AUDIT_VIEW_COLUMNS)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["MCP_SESSION_ID"], "s" * 32)
        self.assertEqual(row["TOOL_NAME"], "fetch_data")
        self.assertEqual(row["INVOCATION_ID"], "i" * 16)
        self.assertEqual(row["HANA_CORRELATION_ID"], "c" * 16)
        self.assertEqual(row["AGENT_NAME"], "langchain")
        self.assertEqual(row["MODEL_NAME"], "claude-opus-4-8")
        self.assertEqual(row["CLIENT_DECLARED_NAME"], "cli-x")
        self.assertEqual(row["CLIENT_IP"], "10.20.30.40")
        self.assertEqual(row["MCP_VERSION"], "hana-ai/0.3")
        self.assertEqual(row["HANA_AUTHENTICATED_USER"], "I516820")
        self.assertEqual(row["APPLICATION_NAME"], "cli-x")
        self.assertEqual(row["EXECUTION_COUNT"], 3)
        # No beacon row present -> RESPONSE_SIZE stays None.
        self.assertIsNone(row["RESPONSE_SIZE"])

    def test_response_size_propagates_from_beacon_to_tool_rows(self):
        # Two rows share INVOCATION_ID: one real tool SQL (no resp) and one
        # beacon SQL (carries resp=). The view must propagate resp onto the
        # tool row AND filter out the beacon row itself.
        inv = "inv-" + "a" * 32
        started_pack = build_appsource_pack(
            mcp_version="0.3",
            mcp_session_id="s" * 32,
            tool_name="fetch",
            invocation_id=inv,
        )
        beacon_pack = build_appsource_pack(
            mcp_version="0.3",
            mcp_session_id="s" * 32,
            tool_name="fetch",
            invocation_id=inv,
            response_size=7,
        )
        beacon_stmt = f"SELECT /* {MCP_BEACON_SQL_MARKER} inv={inv} */ 1 FROM DUMMY"
        conn = _FakeConnection([
            _pc_row(started_pack, stmt_string="SELECT * FROM TARGET_TABLE"),
            _pc_row(beacon_pack, stmt_string=beacon_stmt),
        ])
        df = fetch_hana_mcp_audit_view(conn)

        # Beacon row is filtered out — auditor sees only real tool rows.
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["INVOCATION_ID"], inv)
        self.assertEqual(row["RESPONSE_SIZE"], 7)
        # And it is the real SQL row, not the beacon.
        self.assertEqual(row["APPLICATION_SOURCE"], started_pack)

    def test_response_size_fans_out_to_all_rows_of_same_invocation(self):
        # A single tool call may cache >1 SQL plan (different statements),
        # each landing its own M_SQL_PLAN_CACHE row with the same inv=. The
        # beacon still fires just once — all rows of that invocation must
        # receive RESPONSE_SIZE.
        inv = "inv-" + "b" * 32
        pack_stmt_a = build_appsource_pack(
            mcp_version="0.3", tool_name="load", invocation_id=inv,
        )
        pack_stmt_b = build_appsource_pack(
            mcp_version="0.3", tool_name="load", invocation_id=inv,
        )
        beacon_pack = build_appsource_pack(
            mcp_version="0.3", tool_name="load", invocation_id=inv, response_size=42,
        )
        beacon_stmt = f"SELECT /* {MCP_BEACON_SQL_MARKER} inv={inv} */ 1 FROM DUMMY"
        conn = _FakeConnection([
            _pc_row(pack_stmt_a, stmt_string="SELECT A FROM T"),
            _pc_row(pack_stmt_b, stmt_string="SELECT B FROM T"),
            _pc_row(beacon_pack, stmt_string=beacon_stmt),
        ])
        df = fetch_hana_mcp_audit_view(conn)
        self.assertEqual(len(df), 2)
        self.assertTrue((df["RESPONSE_SIZE"] == 42).all())

    def test_empty_result_returns_all_columns(self):
        df = fetch_hana_mcp_audit_view(_FakeConnection([]))
        self.assertEqual(list(df.columns), HANA_MCP_AUDIT_VIEW_COLUMNS)
        self.assertEqual(len(df), 0)

    def test_filters_translate_into_where_and_params(self):
        pack = build_appsource_pack(
            mcp_version="0.3",
            mcp_session_id="sess-xyz",
            tool_name="fetch",
            invocation_id="i" * 16,
        )
        conn = _FakeConnection([_pc_row(pack, app_name="cli-x")])
        fetch_hana_mcp_audit_view(
            conn,
            mcp_session_id="sess-xyz",
            application_name="cli-x",
            tool_name="fetch",
            since_seconds=3600,
            limit=42,
        )
        sql, params = conn.cursors[0].executed[0]
        self.assertIn("APPLICATION_SOURCE LIKE 'mcp=hana-ai/%'", sql)
        self.assertIn("APPLICATION_NAME = ?", sql)
        self.assertIn("LAST_EXECUTION_TIMESTAMP > ADD_SECONDS(CURRENT_TIMESTAMP, ?)", sql)
        self.assertIn("LIMIT  42", sql)
        # STATEMENT_STRING is projected so the beacon filter can run.
        self.assertIn("STATEMENT_STRING", sql)
        # 3 LIKE-style filters (sess/app/tool) + 1 since_seconds negative int
        self.assertEqual(params, ["%sess=sess-xyz%", "cli-x", "%tool=fetch%", -3600])

    def test_bytes_application_source_is_decoded(self):
        pack = build_appsource_pack(
            mcp_version="0.3",
            mcp_session_id="s" * 32,
            tool_name="fetch",
            invocation_id="i" * 16,
        )
        # Simulate a driver that hands us memoryview instead of str.
        row = list(_pc_row(pack))
        row[6] = memoryview(pack.encode("ascii"))
        conn = _FakeConnection([tuple(row)])
        df = fetch_hana_mcp_audit_view(conn)
        self.assertEqual(df.iloc[0]["TOOL_NAME"], "fetch")


if __name__ == "__main__":
    unittest.main()
