import json
import unittest
from unittest.mock import patch

import pandas as pd

from hana_ai.tools.hana_ml_tools.additive_model_forecast_tools import AdditiveModelForecastLoadModelAndPredict
from hana_ai.tools.hana_ml_tools.additive_model_forecast_tools import MassiveAdditiveModelForecastLoadModelAndPredict
from hana_ai.tools.hana_ml_tools.automatic_timeseries_tools import AutomaticTimeSeriesLoadModelAndPredict
from hana_ai.tools.hana_ml_tools.massive_automatic_timeseries_tools import MassiveAutomaticTimeSeriesLoadModelAndPredict
from hana_ai.tools.hana_ml_tools.utility import is_predict_feature_mismatch_error


class FakeDataFrame:
    def __init__(self, columns, collected=None):
        self.columns = list(columns)
        self._collected = collected
        self.saved_as = None
        self.smart_saved_as = None

    def select(self, *columns):
        return FakeDataFrame(columns)

    def add_constant(self, column_name, _value):
        return FakeDataFrame([*self.columns, column_name])

    def save(self, name, force=False):
        self.saved_as = (name, force)
        return self

    def smart_save(self, name, force=False):
        self.smart_saved_as = (name, force)
        return self

    def collect(self):
        return self._collected if self._collected is not None else pd.DataFrame()


class FakeConnectionContext:
    def __init__(self, tables):
        self.tables = tables

    def has_table(self, name, schema=None):
        return name in self.tables

    def table(self, name, schema=None):
        return self.tables[name]


class PredictModelStub:
    def __init__(self, mismatch_message, output_table_names, fail_first_call=True):
        self.version = 1
        self._predict_output_table_names = output_table_names
        self.calls = []
        self._mismatch_message = mismatch_message
        self._fail_first_call = fail_first_call

    def predict(self, data, **kwargs):
        self.calls.append({"columns": list(data.columns), "kwargs": kwargs})
        if self._fail_first_call and len(self.calls) == 1:
            raise Exception(self._mismatch_message)


class ModelStorageStub:
    def __init__(self, model):
        self.model = model
        self.saved_models = []

    def load_model(self, name=None, version=None):
        return self.model

    def save_model(self, model=None, if_exists=None):
        self.saved_models.append({"model": model, "if_exists": if_exists})


def make_tool(tool_cls, connection_context):
    tool = object.__new__(tool_cls)
    object.__setattr__(tool, "connection_context", connection_context)
    object.__setattr__(tool, "return_direct", False)
    return tool


class TestPredictionAutoRepair(unittest.TestCase):
    def test_mismatch_detection_is_not_triggered_by_generic_pipeline_predict_error(self):
        self.assertFalse(is_predict_feature_mismatch_error(Exception("AFL error in PIPELINE_PREDICT without schema mismatch details")))
        self.assertTrue(is_predict_feature_mismatch_error(Exception("feature number of predict table does not match the trained model")))

    def test_automatic_predict_retries_with_repaired_columns(self):
        model = PredictModelStub(
            "feature number of predict table does not match the trained model",
            ["AUTO_PRED_OUT", "AUTO_STATS_OUT"],
            fail_first_call=False,
        )
        storage = ModelStorageStub(model)
        conn = FakeConnectionContext(
            {
                "PREDICT_INPUT": FakeDataFrame(["TIMESTAMP", "VALUE"]),
                "AUTO_PRED_OUT": FakeDataFrame(["TIMESTAMP", "FORECAST"]),
                "AUTO_STATS_OUT": FakeDataFrame([], pd.DataFrame([["MSE", 1.0]], columns=["STAT_NAME", "STAT_VALUE"])),
            }
        )

        with patch("hana_ai.tools.hana_ml_tools.automatic_timeseries_tools.ModelStorage", return_value=storage):
            tool = make_tool(AutomaticTimeSeriesLoadModelAndPredict, conn)
            result = json.loads(tool._run(predict_table="PREDICT_INPUT", key="TIMESTAMP", name="MODEL", version=1))

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["columns"], ["TIMESTAMP"])
        self.assertTrue(result["auto_repaired_predict_input"])
        self.assertEqual(result["predict_table_columns_used_for_prediction"], ["TIMESTAMP"])
        self.assertEqual(result["predicted_results_table"], "PREDICT_RESULT_PREDICT_INPUT_MODEL_1")

    def test_additive_predict_retries_with_repaired_columns(self):
        model = PredictModelStub(
            "feature number of predict table does not match the trained model",
            ["ADD_PRED_OUT", "ADD_REASON_OUT"],
            fail_first_call=False,
        )
        storage = ModelStorageStub(model)
        conn = FakeConnectionContext(
            {
                "PREDICT_INPUT": FakeDataFrame(["TIMESTAMP", "VALUE"]),
                "ADD_PRED_OUT": FakeDataFrame(["TIMESTAMP", "FORECAST"]),
                "ADD_REASON_OUT": FakeDataFrame(["TIMESTAMP", "REASON"]),
            }
        )

        with patch("hana_ai.tools.hana_ml_tools.additive_model_forecast_tools.ModelStorage", return_value=storage):
            tool = make_tool(AdditiveModelForecastLoadModelAndPredict, conn)
            result = json.loads(tool._run(predict_table="PREDICT_INPUT", key="TIMESTAMP", name="MODEL", version=1))

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["columns"], ["TIMESTAMP", "PLACEHOLDER"])
        self.assertTrue(result["auto_repaired_predict_input"])
        self.assertEqual(result["predict_table_columns_used_for_prediction"], ["TIMESTAMP"])
        self.assertEqual(result["predicted_results_table"], "PREDICT_RESULT_PREDICT_INPUT_MODEL_1")

    def test_massive_predict_retries_with_group_and_key_only(self):
        model = PredictModelStub(
            "feature number of predict table does not match the trained model",
            ["MASSIVE_PRED_OUT", "MASSIVE_STATS_OUT"],
            fail_first_call=False,
        )
        storage = ModelStorageStub(model)
        conn = FakeConnectionContext(
            {
                "PREDICT_INPUT": FakeDataFrame(["GROUP_ID", "TIMESTAMP", "VALUE"]),
                "MASSIVE_PRED_OUT": FakeDataFrame(["GROUP_ID", "TIMESTAMP", "FORECAST"]),
                "MASSIVE_STATS_OUT": FakeDataFrame([], pd.DataFrame([["MSE", 1.0]], columns=["STAT_NAME", "STAT_VALUE"])),
            }
        )

        with patch("hana_ai.tools.hana_ml_tools.massive_automatic_timeseries_tools.ModelStorage", return_value=storage):
            tool = make_tool(MassiveAutomaticTimeSeriesLoadModelAndPredict, conn)
            result = json.loads(tool._run(predict_table="PREDICT_INPUT", key="TIMESTAMP", group_key="GROUP_ID", name="MODEL", version=1))

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["columns"], ["GROUP_ID", "TIMESTAMP"])
        self.assertTrue(result["auto_repaired_predict_input"])
        self.assertEqual(result["predict_table_columns_used_for_prediction"], ["GROUP_ID", "TIMESTAMP"])
        self.assertEqual(result["predicted_results_table"], "PREDICT_RESULT_PREDICT_INPUT_MODEL_1")

    def test_massive_additive_predict_retries_with_group_and_key_only(self):
        model = PredictModelStub(
            "feature number of predict table does not match the trained model",
            ["MASSIVE_ADD_PRED_OUT", "MASSIVE_ADD_REASON_OUT", "MASSIVE_ADD_ERR_OUT"],
            fail_first_call=False,
        )
        storage = ModelStorageStub(model)
        conn = FakeConnectionContext(
            {
                "PREDICT_INPUT": FakeDataFrame(["GROUP_ID", "TIMESTAMP", "VALUE"]),
                "MASSIVE_ADD_PRED_OUT": FakeDataFrame(["GROUP_ID", "TIMESTAMP", "FORECAST"]),
                "MASSIVE_ADD_REASON_OUT": FakeDataFrame(["GROUP_ID", "TIMESTAMP", "REASON"]),
                "MASSIVE_ADD_ERR_OUT": FakeDataFrame(["GROUP_ID", "MESSAGE"]),
            }
        )

        with patch("hana_ai.tools.hana_ml_tools.additive_model_forecast_tools.ModelStorage", return_value=storage):
            tool = make_tool(MassiveAdditiveModelForecastLoadModelAndPredict, conn)
            result = json.loads(tool._run(predict_table="PREDICT_INPUT", key="TIMESTAMP", group_key="GROUP_ID", name="MODEL", version=1))

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["columns"], ["GROUP_ID", "TIMESTAMP"])
        self.assertTrue(result["auto_repaired_predict_input"])
        self.assertEqual(result["predict_table_columns_used_for_prediction"], ["GROUP_ID", "TIMESTAMP"])
        self.assertEqual(result["predicted_results_table"], "PREDICT_RESULT_PREDICT_INPUT_MODEL_1")
        self.assertEqual(result["prediction_error_table"], "PREDICT_ERROR_PREDICT_INPUT_MODEL_1")

    def test_predict_still_returns_diagnostic_when_repaired_input_cannot_fix_schema(self):
        model = PredictModelStub(
            "feature number of predict table does not match the trained model",
            ["AUTO_PRED_OUT", "AUTO_STATS_OUT"],
            fail_first_call=True,
        )
        storage = ModelStorageStub(model)
        conn = FakeConnectionContext(
            {
                "PREDICT_INPUT": FakeDataFrame(["TIMESTAMP"]),
                "AUTO_PRED_OUT": FakeDataFrame(["TIMESTAMP", "FORECAST"]),
                "AUTO_STATS_OUT": FakeDataFrame([], pd.DataFrame([["MSE", 1.0]], columns=["STAT_NAME", "STAT_VALUE"])),
            }
        )

        with patch("hana_ai.tools.hana_ml_tools.automatic_timeseries_tools.ModelStorage", return_value=storage):
            tool = make_tool(AutomaticTimeSeriesLoadModelAndPredict, conn)
            result = json.loads(tool._run(predict_table="PREDICT_INPUT", key="TIMESTAMP", name="MODEL", version=1))

        self.assertEqual(result["error_category"], "predict_table_feature_mismatch")