"""Unit tests for the opt-in AutoML fit-tool config_dict guard.

The AutoML fit tools (``automatic_timeseries_fit_and_save`` and its massive
variant) can be configured to refuse a call with a missing ``config_dict``,
so an agent that skips the ``ts_check`` -> ``get_automl_config_dict`` ->
(optional ``modify_automl_config_dict``) -> fit flow is routed back through
it. The guard is off by default so existing callers that always ran the PAL
default pipeline keep working; setting ``HANA_AI_AUTOML_REQUIRE_CONFIG_DICT=1``
in the environment turns it on.
"""

import json
import os
import unittest
from unittest.mock import patch

from hana_ml import ConnectionContext

from hana_ai.tools.hana_ml_tools.automatic_timeseries_tools import (
    AutomaticTimeSeriesFitAndSave,
)
from hana_ai.tools.hana_ml_tools.massive_automatic_timeseries_tools import (
    MassiveAutomaticTimeSeriesFitAndSave,
)
from hana_ai.tools.hana_ml_tools import config_dict_validator_tools as cfg_module


def _stub_connection_context():
    """Build an un-connected ConnectionContext so the tool can be instantiated
    without opening a HANA session. When the guard fires it returns before any
    connection work, so the stub is never dereferenced."""
    with patch.object(ConnectionContext, "__init__", lambda self, *a, **k: None):
        return ConnectionContext()


class TestAutoMLFitGuard(unittest.TestCase):
    def setUp(self):
        # Every test starts with a clean environment so opt-in state is explicit.
        self._saved = os.environ.pop("HANA_AI_AUTOML_REQUIRE_CONFIG_DICT", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("HANA_AI_AUTOML_REQUIRE_CONFIG_DICT", None)
        else:
            os.environ["HANA_AI_AUTOML_REQUIRE_CONFIG_DICT"] = self._saved

    # -- guard OFF (default) --------------------------------------------------

    def test_timeseries_fit_defaults_to_not_guarding(self):
        # No env var -> the tool must NOT short-circuit. It will fail later on
        # the missing HANA connection, but crucially it must not return the
        # `config_dict_required` payload — that would break existing callers.
        tool = AutomaticTimeSeriesFitAndSave(connection_context=_stub_connection_context())
        try:
            raw = tool._run(fit_table="T", key="K", endog="E", name="M")
        except Exception:
            # Downstream code touches the stubbed connection; that's fine, the
            # point is we got past the guard check.
            return
        # If _run happened to return (e.g. connection stub extended), it must
        # not carry the guard error.
        try:
            parsed = json.loads(raw)
        except Exception:
            return
        self.assertNotEqual(parsed.get("error"), "config_dict_required", parsed)

    def test_massive_fit_defaults_to_not_guarding(self):
        tool = MassiveAutomaticTimeSeriesFitAndSave(
            connection_context=_stub_connection_context()
        )
        try:
            raw = tool._run(fit_table="T", key="K", group_key="G", endog="E", name="M")
        except Exception:
            return
        try:
            parsed = json.loads(raw)
        except Exception:
            return
        self.assertNotEqual(parsed.get("error"), "config_dict_required", parsed)

    # -- guard ON -------------------------------------------------------------

    def test_timeseries_fit_guard_fires_when_opted_in(self):
        os.environ["HANA_AI_AUTOML_REQUIRE_CONFIG_DICT"] = "1"
        tool = AutomaticTimeSeriesFitAndSave(connection_context=_stub_connection_context())
        raw = tool._run(fit_table="T", key="K", endog="E", name="M")
        parsed = json.loads(raw)
        self.assertEqual(parsed["error"], "config_dict_required")
        self.assertEqual(parsed["next_action"]["tool"], "get_automl_config_dict")
        self.assertEqual(parsed["next_action"]["pipeline_type"], "timeseries")
        self.assertIn("get_automl_config_dict", parsed["message"])
        self.assertIn("modify_automl_config_dict", parsed["message"])
        self.assertIn("ts_check", parsed["message"])

    def test_massive_fit_guard_fires_when_opted_in(self):
        os.environ["HANA_AI_AUTOML_REQUIRE_CONFIG_DICT"] = "1"
        tool = MassiveAutomaticTimeSeriesFitAndSave(
            connection_context=_stub_connection_context()
        )
        raw = tool._run(
            fit_table="T", key="K", group_key="G", endog="E", name="M"
        )
        parsed = json.loads(raw)
        self.assertEqual(parsed["error"], "config_dict_required")
        self.assertEqual(parsed["next_action"]["tool"], "get_automl_config_dict")
        self.assertIn("massive_ts_check", parsed["message"])

    def test_guard_recognises_various_truthy_values(self):
        for value in ("1", "true", "TRUE", "yes", "on", "True"):
            os.environ["HANA_AI_AUTOML_REQUIRE_CONFIG_DICT"] = value
            tool = AutomaticTimeSeriesFitAndSave(
                connection_context=_stub_connection_context()
            )
            raw = tool._run(fit_table="T", key="K", endog="E", name="M")
            parsed = json.loads(raw)
            self.assertEqual(parsed["error"], "config_dict_required", (value, parsed))

    def test_guard_stays_off_for_falsy_values(self):
        for value in ("0", "false", "no", "", "off"):
            os.environ["HANA_AI_AUTOML_REQUIRE_CONFIG_DICT"] = value
            tool = AutomaticTimeSeriesFitAndSave(
                connection_context=_stub_connection_context()
            )
            try:
                raw = tool._run(fit_table="T", key="K", endog="E", name="M")
            except Exception:
                continue
            try:
                parsed = json.loads(raw)
            except Exception:
                continue
            self.assertNotEqual(
                parsed.get("error"), "config_dict_required", (value, parsed)
            )


class TestAutoMLFitConfigDictPreflight(unittest.TestCase):
    """The fit tools must catch schema-invalid config_dicts before PAL does.

    PAL emits errors like "Illegal operator for pipeline type timeseries:
    Seasonality" when the agent hands it an unknown operator. The fit tool
    runs the authoritative PAL_AUTOML_CONFIG(VERIFY_CONFIG=1) check so the
    AutoML search never starts with an unusable config. These tests patch the
    ``_call_pal_automl_config`` helper so the guard exercises the same code
    path without needing a live HANA connection.
    """

    def _mock_pal_reject(self, message):
        """Return a stand-in for ``_call_pal_automl_config`` that emulates PAL
        raising with ``message``. The helper contract returns
        ``(result_json, info_rows, error)`` — the third slot is the surfaced
        error string."""
        return lambda *args, **kwargs: ("", [], message)

    def test_timeseries_fit_rejects_unknown_operator(self):
        tool = AutomaticTimeSeriesFitAndSave(connection_context=_stub_connection_context())
        bad_config = {"Seasonality": {"PERIOD": [7]}, "ARIMA": {"MAX_D": [2]}}
        pal_err = "Illegal operator for pipeline type timeseries: Seasonality"
        with patch.object(
            cfg_module,
            "_call_pal_automl_config",
            self._mock_pal_reject(pal_err),
        ):
            raw = tool._run(
                fit_table="T", key="K", endog="E", name="M", config_dict=bad_config,
            )
        parsed = json.loads(raw)
        self.assertEqual(parsed["error"], "invalid_config_dict", parsed)
        self.assertFalse(parsed["verdict"]["valid"])
        self.assertTrue(
            any("Seasonality" in e for e in parsed["verdict"]["errors"]),
            parsed["verdict"]["errors"],
        )

    def test_timeseries_fit_rejects_unknown_parameter(self):
        tool = AutomaticTimeSeriesFitAndSave(connection_context=_stub_connection_context())
        pal_err = "Illegal parameter for operator ARIMA: NOT_A_PARAM"
        with patch.object(
            cfg_module,
            "_call_pal_automl_config",
            self._mock_pal_reject(pal_err),
        ):
            raw = tool._run(
                fit_table="T", key="K", endog="E", name="M",
                config_dict={"ARIMA": {"NOT_A_PARAM": [1]}},
            )
        parsed = json.loads(raw)
        self.assertEqual(parsed["error"], "invalid_config_dict", parsed)
        self.assertTrue(
            any("NOT_A_PARAM" in e for e in parsed["verdict"]["errors"]),
            parsed["verdict"]["errors"],
        )

    def test_massive_fit_rejects_unknown_operator(self):
        tool = MassiveAutomaticTimeSeriesFitAndSave(
            connection_context=_stub_connection_context()
        )
        pal_err = "Illegal operator for pipeline type timeseries: Seasonality"
        with patch.object(
            cfg_module,
            "_call_pal_automl_config",
            self._mock_pal_reject(pal_err),
        ):
            raw = tool._run(
                fit_table="T", key="K", group_key="G", endog="E", name="M",
                config_dict={"Seasonality": {"PERIOD": [7]}},
            )
        parsed = json.loads(raw)
        self.assertEqual(parsed["error"], "invalid_config_dict", parsed)

    def test_pal_template_strings_pass_preflight(self):
        # Templates 'default' / 'light' / 'empty' are consumed by PAL directly
        # and are not subject to the VERIFY_CONFIG round-trip. The pre-flight
        # guard must short-circuit before touching ``_call_pal_automl_config``;
        # the fit will then fail on the connection stub, not on the validator.
        tool = AutomaticTimeSeriesFitAndSave(connection_context=_stub_connection_context())
        for template in ("default", "light", "empty"):
            call_log = []

            def _tracked(*args, **kwargs):
                call_log.append((args, kwargs))
                return ("{}", [], None)

            with patch.object(cfg_module, "_call_pal_automl_config", _tracked):
                try:
                    raw = tool._run(
                        fit_table="T", key="K", endog="E", name="M",
                        config_dict=template,
                    )
                except Exception:
                    self.assertEqual(call_log, [], (template, call_log))
                    continue
            self.assertEqual(call_log, [], (template, call_log))
            try:
                parsed = json.loads(raw)
            except Exception:
                continue
            self.assertNotEqual(
                parsed.get("error"), "invalid_config_dict", (template, parsed)
            )


if __name__ == "__main__":
    unittest.main()
