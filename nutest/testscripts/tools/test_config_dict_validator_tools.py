"""Unit tests for the HANA-backed PAL config tools.

The three tools (``GetPALPipelineInfo``, ``GetAutoMLConfigDict``,
``ModifyAutoMLConfigDict``) delegate to PAL SQL procedures via a shared
``_call_pal_automl_config`` helper. These tests patch that helper (or the
underlying ``PALBase._call_pal_auto``) so no HANA connection is required.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from hana_ml import ConnectionContext

from hana_ai.tools.hana_ml_tools import config_dict_validator_tools as cfg_module
from hana_ai.tools.hana_ml_tools.config_dict_validator_tools import (
    GetAutoMLConfigDict,
    GetPALPipelineInfo,
    ModifyAutoMLConfigDict,
    _call_pal_automl_config,
    _encode_config_payload,
)


def _stub_connection_context():
    """Instantiate a ConnectionContext without opening a session — mirrors the
    helper in test_automl_fit_guard.py."""
    with patch.object(ConnectionContext, "__init__", lambda self, *a, **k: None):
        return ConnectionContext()


class TestEncodeConfigPayload(unittest.TestCase):
    """The client-side payload coercion used before shipping a row to PAL."""

    def test_none_passes_through(self):
        self.assertIsNone(_encode_config_payload(None))

    def test_template_keywords_are_lowercased(self):
        for template in ("default", "Default", "LIGHT", "empty"):
            self.assertEqual(_encode_config_payload(template), template.strip().lower())

    def test_dict_is_json_encoded(self):
        encoded = _encode_config_payload({"ARIMA": {"MAX_D": [2]}})
        self.assertEqual(json.loads(encoded), {"ARIMA": {"MAX_D": [2]}})

    def test_json_string_passes_through(self):
        raw = '{"ARIMA": {"MAX_D": [2]}}'
        self.assertEqual(_encode_config_payload(raw), raw)


class TestCallPalAutomlConfigHelper(unittest.TestCase):
    """The helper builds the ParameterTable rows and re-assembles the result."""

    def _patched_call(self, param_rows_sink, *, raise_on_call=False):
        """Return a stand-in for ``PALBase._call_pal_auto`` that captures the
        ParameterTable data rows into ``param_rows_sink``."""

        def _fake_call_pal_auto(pal_base, connection_context, proc_name, param_table, *outputs):
            # ParameterTable stores rows on ``.data`` when built via with_data.
            data = getattr(param_table, "data", None)
            if data is None:
                # Fall back to iterating the object if the API changes.
                data = list(param_table)
            param_rows_sink.append(list(data))
            if raise_on_call:
                from hdbcli import dbapi
                raise dbapi.Error("simulated PAL failure: Illegal operator")

        return _fake_call_pal_auto

    def _patched_table_reader(self, *, result_content="{}", info_rows=None):
        """Return a stand-in for ``ConnectionContext.table`` that yields fake
        RESULT / INFO frames depending on the requested table name."""
        info_rows = info_rows or []

        def _fake_table(name, *args, **kwargs):
            frame = MagicMock()
            if "RESULT" in name:
                frame.collect.return_value = pd.DataFrame(
                    [{"ROW_INDEX": 0, "CONTENT": result_content}]
                )
            elif "INFO" in name:
                if info_rows:
                    frame.collect.return_value = pd.DataFrame(info_rows)
                else:
                    frame.collect.return_value = pd.DataFrame(
                        columns=["NAME", "TYPE", "CONFIG"]
                    )
            else:
                frame.collect.return_value = pd.DataFrame()
            return frame

        return _fake_table

    def test_param_row_order_and_verify_flag(self):
        """PAL cares about row ordering: PIPELINE_TYPE, CONFIG_DICT, CONFIG_REMOVE,
        CONFIG_ADD, CONFIG_REPLACE, CONFIG_MODIFY, VERIFY_CONFIG."""
        captured = []
        cc = _stub_connection_context()
        with patch(
            "hana_ai.tools.hana_ml_tools.config_dict_validator_tools.PALBase._call_pal_auto",
            self._patched_call(captured),
        ), patch.object(cc, "table", self._patched_table_reader(result_content='{"ARIMA":{}}')), patch(
            "hana_ai.tools.hana_ml_tools.config_dict_validator_tools.try_drop"
        ):
            _call_pal_automl_config(
                cc,
                pipeline_type="timeseries",
                config_dict="default",
                config_remove={"BSTS": []},
                config_add={"ARIMA": {"MAX_D": [3]}},
                verify=True,
            )
        self.assertEqual(len(captured), 1)
        rows = captured[0]
        names_in_order = [row[0] for row in rows]
        self.assertEqual(
            names_in_order,
            ["PIPELINE_TYPE", "CONFIG_DICT", "CONFIG_REMOVE", "CONFIG_ADD", "VERIFY_CONFIG"],
        )
        # Values map to the string slot for the JSON rows, int slot for VERIFY.
        by_name = {row[0]: row for row in rows}
        self.assertEqual(by_name["PIPELINE_TYPE"][3], "timeseries")
        self.assertEqual(by_name["CONFIG_DICT"][3], "default")
        self.assertEqual(json.loads(by_name["CONFIG_REMOVE"][3]), {"BSTS": []})
        self.assertEqual(json.loads(by_name["CONFIG_ADD"][3]), {"ARIMA": {"MAX_D": [3]}})
        self.assertEqual(by_name["VERIFY_CONFIG"][1], 1)

    def test_none_rows_are_skipped(self):
        captured = []
        cc = _stub_connection_context()
        with patch(
            "hana_ai.tools.hana_ml_tools.config_dict_validator_tools.PALBase._call_pal_auto",
            self._patched_call(captured),
        ), patch.object(cc, "table", self._patched_table_reader()), patch(
            "hana_ai.tools.hana_ml_tools.config_dict_validator_tools.try_drop"
        ):
            _call_pal_automl_config(cc, pipeline_type="classifier", config_dict="light")
        rows = captured[0]
        names_in_order = [row[0] for row in rows]
        # Only PIPELINE_TYPE + CONFIG_DICT: no modifier rows, no VERIFY_CONFIG.
        self.assertEqual(names_in_order, ["PIPELINE_TYPE", "CONFIG_DICT"])

    def test_pal_error_becomes_third_slot(self):
        captured = []
        cc = _stub_connection_context()
        with patch(
            "hana_ai.tools.hana_ml_tools.config_dict_validator_tools.PALBase._call_pal_auto",
            self._patched_call(captured, raise_on_call=True),
        ), patch.object(cc, "table", self._patched_table_reader()), patch(
            "hana_ai.tools.hana_ml_tools.config_dict_validator_tools.try_drop"
        ):
            result_json, info_rows, err = _call_pal_automl_config(
                cc,
                pipeline_type="timeseries",
                config_dict={"Seasonality": {"PERIOD": [7]}},
                verify=True,
            )
        self.assertEqual(result_json, "")
        self.assertEqual(info_rows, [])
        self.assertIn("simulated PAL failure", err)

    def test_info_rows_are_reshaped(self):
        cc = _stub_connection_context()
        info = [
            {"NAME": "ARIMA", "TYPE": "estimator", "CONFIG": '{"MAX_D": [2]}'},
            {"NAME": "Outlier", "TYPE": "transformer", "CONFIG": "{}"},
        ]
        with patch(
            "hana_ai.tools.hana_ml_tools.config_dict_validator_tools.PALBase._call_pal_auto",
            self._patched_call([]),
        ), patch.object(
            cc,
            "table",
            self._patched_table_reader(result_content='{"ARIMA":{}}', info_rows=info),
        ), patch(
            "hana_ai.tools.hana_ml_tools.config_dict_validator_tools.try_drop"
        ):
            _, info_rows, err = _call_pal_automl_config(
                cc, pipeline_type="timeseries", config_dict="default"
            )
        self.assertIsNone(err)
        self.assertEqual(info_rows[0]["operator"], "ARIMA")
        self.assertEqual(info_rows[0]["type"], "estimator")
        self.assertEqual(info_rows[1]["operator"], "Outlier")


class TestGetPALPipelineInfoTool(unittest.TestCase):
    def test_metadata(self):
        tool = GetPALPipelineInfo(connection_context=_stub_connection_context())
        self.assertEqual(tool.name, "get_pal_pipeline_info")
        self.assertEqual(
            sorted(tool.args_schema.model_fields.keys()),
            ["category", "include_parameters", "operator"],
        )

    def test_pal_procedure_missing(self):
        tool = GetPALPipelineInfo(connection_context=_stub_connection_context())
        with patch.object(cfg_module, "get_pipeline_info", return_value=False):
            payload = json.loads(tool._run())
        self.assertIn("error", payload)

    def test_operator_filter_and_include_parameters(self):
        rows = pd.DataFrame(
            [
                {"NAME": "ARIMA", "CATEGORY": "estimator", "PARAMETER": "MAX_D..."},
                {"NAME": "BSTS", "CATEGORY": "estimator", "PARAMETER": "BURN_IN..."},
                {"NAME": "ImputeTS", "CATEGORY": "transformer", "PARAMETER": "..."},
            ]
        )
        info_df = MagicMock()
        info_df.collect.return_value = rows

        tool = GetPALPipelineInfo(connection_context=_stub_connection_context())
        with patch.object(cfg_module, "get_pipeline_info", return_value=info_df):
            filtered = json.loads(tool._run(operator="ARIMA"))
            by_cat = json.loads(tool._run(category="transformer"))
            no_params = json.loads(tool._run(include_parameters=False))
        self.assertEqual(filtered["count"], 1)
        self.assertEqual(filtered["operators"][0]["NAME"], "ARIMA")
        self.assertEqual(by_cat["count"], 1)
        self.assertEqual(by_cat["operators"][0]["NAME"], "ImputeTS")
        for row in no_params["operators"]:
            self.assertNotIn("PARAMETER", row)


class TestGetAutoMLConfigDictTool(unittest.TestCase):
    def test_metadata(self):
        tool = GetAutoMLConfigDict(connection_context=_stub_connection_context())
        self.assertEqual(tool.name, "get_automl_config_dict")
        self.assertEqual(
            sorted(tool.args_schema.model_fields.keys()),
            ["config_dict", "pipeline_type"],
        )

    def test_forwards_kwargs_and_parses_result(self):
        tool = GetAutoMLConfigDict(connection_context=_stub_connection_context())
        seen = {}

        def _fake_helper(**kwargs):
            seen.update(kwargs)
            return ('{"ARIMA":{"MAX_D":[2]}}', [{"operator": "ARIMA"}], None)

        with patch.object(cfg_module, "_call_pal_automl_config", side_effect=lambda cc, **kw: _fake_helper(**kw)):
            payload = json.loads(
                tool._run(pipeline_type="timeseries", config_dict="default")
            )
        self.assertEqual(seen["pipeline_type"], "timeseries")
        self.assertEqual(seen["config_dict"], "default")
        self.assertFalse(seen["verify"])
        self.assertEqual(payload["pipeline_type"], "timeseries")
        self.assertEqual(payload["config_dict"], {"ARIMA": {"MAX_D": [2]}})
        self.assertEqual(payload["operators"], [{"operator": "ARIMA"}])

    def test_invalid_pipeline_type_short_circuits(self):
        tool = GetAutoMLConfigDict(connection_context=_stub_connection_context())
        with patch.object(cfg_module, "_call_pal_automl_config") as mock_call:
            payload = json.loads(tool._run(pipeline_type="not-a-thing"))
        mock_call.assert_not_called()
        self.assertIn("error", payload)

    def test_helper_error_surfaces(self):
        tool = GetAutoMLConfigDict(connection_context=_stub_connection_context())
        with patch.object(
            cfg_module,
            "_call_pal_automl_config",
            return_value=("", [], "PAL blew up"),
        ):
            payload = json.loads(tool._run(pipeline_type="timeseries"))
        self.assertEqual(payload["error"], "pal_automl_config_failed")
        self.assertEqual(payload["detail"], "PAL blew up")


class TestModifyAutoMLConfigDictTool(unittest.TestCase):
    def test_metadata(self):
        tool = ModifyAutoMLConfigDict(connection_context=_stub_connection_context())
        self.assertEqual(tool.name, "modify_automl_config_dict")
        self.assertEqual(
            sorted(tool.args_schema.model_fields.keys()),
            [
                "config_add",
                "config_dict",
                "config_modify",
                "config_remove",
                "config_replace",
                "pipeline_type",
                "verify",
            ],
        )

    def test_default_verify_is_true(self):
        tool = ModifyAutoMLConfigDict(connection_context=_stub_connection_context())
        seen = {}

        def _fake_helper(cc, **kwargs):
            seen.update(kwargs)
            return ('{}', [], None)

        with patch.object(cfg_module, "_call_pal_automl_config", side_effect=_fake_helper):
            tool._run(pipeline_type="timeseries")
        self.assertTrue(seen["verify"])

    def test_verify_false_surfaces_helper_error_generically(self):
        tool = ModifyAutoMLConfigDict(connection_context=_stub_connection_context())
        with patch.object(
            cfg_module,
            "_call_pal_automl_config",
            return_value=("", [], "connection died"),
        ):
            payload = json.loads(
                tool._run(pipeline_type="timeseries", verify=False)
            )
        self.assertEqual(payload["error"], "pal_automl_config_failed")

    def test_verify_true_returns_invalid_config_dict(self):
        tool = ModifyAutoMLConfigDict(connection_context=_stub_connection_context())
        pal_err = "Illegal operator for pipeline type timeseries: Seasonality"
        with patch.object(
            cfg_module,
            "_call_pal_automl_config",
            return_value=("", [], pal_err),
        ):
            payload = json.loads(
                tool._run(
                    pipeline_type="timeseries",
                    config_dict={"Seasonality": {"PERIOD": [7]}},
                    verify=True,
                )
            )
        self.assertEqual(payload["error"], "invalid_config_dict")
        self.assertIn("Seasonality", payload["detail"])

    def test_forwards_all_modifier_kwargs(self):
        tool = ModifyAutoMLConfigDict(connection_context=_stub_connection_context())
        seen = {}

        def _fake_helper(cc, **kwargs):
            seen.update(kwargs)
            return ('{"ARIMA":{}}', [], None)

        with patch.object(cfg_module, "_call_pal_automl_config", side_effect=_fake_helper):
            tool._run(
                pipeline_type="timeseries",
                config_dict="default",
                config_add={"ARIMA": {"MAX_D": [3]}},
                config_remove={"BSTS": []},
                config_replace={"Outlier": {}},
                config_modify={"ARIMA": {"MAX_D": [2]}},
                verify=True,
            )
        self.assertEqual(seen["config_dict"], "default")
        self.assertEqual(seen["config_add"], {"ARIMA": {"MAX_D": [3]}})
        self.assertEqual(seen["config_remove"], {"BSTS": []})
        self.assertEqual(seen["config_replace"], {"Outlier": {}})
        self.assertEqual(seen["config_modify"], {"ARIMA": {"MAX_D": [2]}})
        self.assertTrue(seen["verify"])


if __name__ == "__main__":
    unittest.main()
