import json
import unittest
from unittest.mock import patch

from hana_ai.tools.hana_ml_tools.additive_model_forecast_tools import MassiveAdditiveModelForecastFitAndSave
from hana_ai.tools.hana_ml_tools.additive_model_forecast_tools import MassiveAdditiveModelForecastLoadModelAndPredict


class FakeDataFrame:
    def __init__(self, columns):
        self.columns = list(columns)
        self.smart_saved_as = None

    def select(self, *columns):
        return FakeDataFrame(columns)

    def smart_save(self, name, force=False):
        self.smart_saved_as = (name, force)
        return self


class FakeConnectionContext:
    def __init__(self, tables):
        self.tables = tables

    def has_table(self, name, schema=None):
        return name in self.tables

    def table(self, name, schema=None):
        return self.tables[name]


class AdditiveModelStub:
    last_instance = None

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.fit_calls = []
        self.predict_calls = []
        self.version = None
        self.name = None
        self._predict_output_table_names = ["PRED_OUT", "REASON_OUT", "ERR_OUT"]
        AdditiveModelStub.last_instance = self

    def fit(self, **kwargs):
        self.fit_calls.append(kwargs)

    def predict(self, data, **kwargs):
        self.predict_calls.append({"columns": list(data.columns), "kwargs": kwargs})


class ModelStorageStub:
    def __init__(self, model=None):
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


class TestMassiveAdditiveModelForecastTools(unittest.TestCase):
    def test_fit_and_save_uses_massive_additive_model(self):
        conn = FakeConnectionContext({"FIT_INPUT": FakeDataFrame(["GROUP_ID", "TIMESTAMP", "VALUE"])})
        storage = ModelStorageStub()

        with patch("hana_ai.tools.hana_ml_tools.additive_model_forecast_tools.AdditiveModelForecast", AdditiveModelStub), patch(
            "hana_ai.tools.hana_ml_tools.additive_model_forecast_tools.ModelStorage", return_value=storage
        ), patch(
            "hana_ai.tools.hana_ml_tools.additive_model_forecast_tools.generate_model_storage_version", return_value=7
        ):
            tool = make_tool(MassiveAdditiveModelForecastFitAndSave, conn)
            result = json.loads(
                tool._run(
                    fit_table="FIT_INPUT",
                    key="TIMESTAMP",
                    group_key="GROUP_ID",
                    endog="VALUE",
                    name="MODEL",
                    period=30,
                )
            )

        model = AdditiveModelStub.last_instance
        self.assertIsNotNone(model)
        self.assertTrue(model.init_kwargs["massive"])
        self.assertEqual(model.fit_calls[0]["group_key"], "GROUP_ID")
        self.assertEqual(result["model_storage_version"], 7)

    def test_load_model_and_predict_saves_prediction_and_error_tables(self):
        model = AdditiveModelStub(massive=True)
        storage = ModelStorageStub(model)
        conn = FakeConnectionContext(
            {
                "PREDICT_INPUT": FakeDataFrame(["GROUP_ID", "TIMESTAMP", "VALUE"]),
                "PRED_OUT": FakeDataFrame(["GROUP_ID", "TIMESTAMP", "YHAT"]),
                "REASON_OUT": FakeDataFrame(["GROUP_ID", "TIMESTAMP", "TREND"]),
                "ERR_OUT": FakeDataFrame(["GROUP_ID", "MESSAGE"]),
            }
        )

        with patch("hana_ai.tools.hana_ml_tools.additive_model_forecast_tools.ModelStorage", return_value=storage):
            tool = make_tool(MassiveAdditiveModelForecastLoadModelAndPredict, conn)
            result = json.loads(
                tool._run(
                    predict_table="PREDICT_INPUT",
                    key="TIMESTAMP",
                    group_key="GROUP_ID",
                    name="MODEL",
                    version=3,
                    show_explainer=True,
                )
            )

        self.assertEqual(model.predict_calls[0]["columns"], ["GROUP_ID", "TIMESTAMP"])
        self.assertEqual(result["predicted_results_table"], "PREDICT_RESULT_PREDICT_INPUT_MODEL_3")
        self.assertEqual(result["decomposed_and_reason_code_table"], "REASON_CODE_PREDICT_INPUT_MODEL_3")
        self.assertEqual(result["prediction_error_table"], "PREDICT_ERROR_PREDICT_INPUT_MODEL_3")