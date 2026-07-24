"""Pure unit tests for build_appsource_pack — no HANA connection needed.

Validates the byte-budget contract that the HANA 4.00 setclientinfo cap
requires:
  * output is ASCII-only
  * byte length <= APPLICATIONSOURCE_MAX_BYTES (254)
  * ``resp=<int>`` (post-completion "beacon" pack, tool response_size) sits
    at the tail so tail-first truncation drops it before dropping identity
"""

import unittest

from hana_ai.tools.hana_ml_tools.utility import (
    APPLICATIONSOURCE_MAX_BYTES,
    build_appsource_pack,
    parse_appsource_pack,
)


class TestBuildAppsourcePack(unittest.TestCase):

    def _assert_pack_valid(self, pack: str) -> None:
        # ASCII-only, respects byte budget.
        pack.encode("ascii")   # raises UnicodeEncodeError if not
        self.assertLessEqual(len(pack.encode("ascii")), APPLICATIONSOURCE_MAX_BYTES)

    def test_stable_layer_only_within_budget(self):
        pack = build_appsource_pack(
            mcp_version="0.3.1",
            mcp_session_id="a" * 32,
            client_declared_agent_name="langchain",
            client_declared_model_name="claude-opus-4-8",
            client_declared_name="mcp-python-client",
            client_ip="10.20.30.40",
        )
        self._assert_pack_valid(pack)
        self.assertIn("mcp=hana-ai/0.3.1", pack)
        self.assertIn("sess=" + "a" * 32, pack)
        self.assertIn("agent=langchain", pack)
        self.assertIn("model=claude-opus-4-8", pack)
        self.assertIn("cli=mcp-python-client", pack)
        self.assertIn("mcp_ip=10.20.30.40", pack)
        # No volatile fields set — none should appear.
        self.assertNotIn("|tool=", pack)
        self.assertNotIn("|inv=", pack)
        self.assertNotIn("|resp=", pack)

    def test_started_path_pack_has_no_resp_segment(self):
        # Started path: response_size is unknown, so builder must skip resp=.
        pack = build_appsource_pack(
            mcp_version="0.3",
            mcp_session_id="s" * 32,
            tool_name="fetch_data",
            invocation_id="i" * 16,
            hana_correlation_id="c" * 16,
        )
        self._assert_pack_valid(pack)
        self.assertNotIn("|resp=", pack)
        parsed = parse_appsource_pack(pack)
        self.assertNotIn("response_size", parsed)

    def test_beacon_path_pack_carries_resp_segment(self):
        # Success path: response_size fires from _update_hana_session_event.
        pack = build_appsource_pack(
            mcp_version="0.3",
            mcp_session_id="s" * 32,
            tool_name="fetch_data",
            invocation_id="i" * 16,
            hana_correlation_id="c" * 16,
            response_size=42,
        )
        self._assert_pack_valid(pack)
        self.assertIn("|resp=42", pack)
        # resp is packed last so cheap SUBSTR_BEFORE parsing works.
        self.assertTrue(pack.endswith("|resp=42"))
        parsed = parse_appsource_pack(pack)
        self.assertEqual(parsed["response_size"], 42)

    def test_response_size_zero_still_emitted(self):
        # response_size=0 is legitimate (tool returned an empty result).
        pack = build_appsource_pack(
            mcp_version="0.3",
            tool_name="noop",
            invocation_id="i" * 16,
            response_size=0,
        )
        self._assert_pack_valid(pack)
        self.assertIn("|resp=0", pack)
        self.assertEqual(parse_appsource_pack(pack)["response_size"], 0)

    def test_negative_and_non_numeric_response_size_is_dropped(self):
        for bad in (-5, "not-a-number", True, False, 3.14):
            pack = build_appsource_pack(
                mcp_version="0.3",
                tool_name="x",
                invocation_id="i" * 16,
                response_size=bad,  # type: ignore[arg-type]
            )
            self._assert_pack_valid(pack)
            self.assertNotIn("|resp=", pack, f"bad value {bad!r} must be dropped")

    def test_identity_alone_overflow_falls_back_to_pipe_truncation(self):
        # Feed identity fields that alone already blow the budget.
        long = "z" * 200
        pack = build_appsource_pack(
            mcp_version="0.3",
            mcp_session_id=long,
            client_declared_agent_name=long,
            client_declared_model_name=long,
            client_declared_name=long,
            client_ip="1.2.3.4",
        )
        self._assert_pack_valid(pack)
        # Last segment must still be a well-formed K=V (no half-written key).
        for segment in pack.split("|"):
            self.assertIn("=", segment, f"segment {segment!r} broken")

    def test_field_value_sanitization_blocks_pipe_injection(self):
        # A malicious tool name trying to inject fake fields must be neutered.
        pack = build_appsource_pack(
            mcp_version="0.3",
            tool_name="evil|sess=stolen|",
        )
        self._assert_pack_valid(pack)
        # The pack must contain only ONE sess= (from mcp_session_id, which
        # here is None so it should be absent).
        self.assertEqual(pack.count("sess="), 0)
        # And the tool name's '|' should have been sanitized to '_'.
        self.assertIn("tool=evil_sess_stolen_", pack)

    def test_empty_all_fields_returns_empty(self):
        self.assertEqual(build_appsource_pack(), "")

    def test_identity_plus_resp_never_exceeds_budget(self):
        # Regression: the worst-case production identity from context-agent.
        # Identity alone is ~247 bytes, so a large `resp=` might overflow —
        # in that case the builder must drop `resp=` (tail-first) rather than
        # emit a mangled pack. Auditors still get the value via the JSONL sink.
        pack = build_appsource_pack(
            mcp_version="1.1.26072000",
            mcp_session_id="00a65bdca1204dd99e8f8679fcf7d0cb",
            client_declared_agent_name="context-agent",
            client_declared_model_name="gpt-4.1",
            client_declared_name="context-agent-notebook",
            client_ip="127.0.0.1",
            tool_name="list_models",
            invocation_id="inv-e21ca14d6e824a73a0309a4157929c60",
            hana_correlation_id="hana-corr-b737aed26cf74fbc8191d9f1707c6d22",
            response_size=99999,
        )
        self._assert_pack_valid(pack)
        # Every emitted segment is a complete K=V pair (no half-written tail).
        for segment in pack.split("|"):
            self.assertIn("=", segment, f"segment {segment!r} broken")
        # `resp=99999` fits (8 bytes) only if there's room after identity;
        # if it was dropped, the pack must NOT contain a partial "resp".
        if "resp=" in pack:
            self.assertIn("resp=99999", pack)

    def test_resp_present_when_identity_leaves_room(self):
        # A more typical identity (~150 bytes) leaves plenty of room for resp.
        pack = build_appsource_pack(
            mcp_version="0.3",
            mcp_session_id="s" * 32,
            client_declared_agent_name="langchain",
            client_declared_model_name="claude-opus-4-8",
            client_declared_name="cli",
            tool_name="fetch",
            invocation_id="i" * 16,
            hana_correlation_id="c" * 16,
            response_size=12345,
        )
        self._assert_pack_valid(pack)
        self.assertIn("|resp=12345", pack)


if __name__ == "__main__":
    unittest.main()
