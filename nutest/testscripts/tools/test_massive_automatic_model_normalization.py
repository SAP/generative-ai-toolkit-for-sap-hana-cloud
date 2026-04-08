import json
import unittest
from unittest.mock import patch

import pandas as pd

from hana_ai.tools.hana_ml_tools.massive_automatic_timeseries_tools import (
    MassiveAutomaticTimeSeriesLoadModelAndPredict,
    MassiveAutomaticTimeSeriesLoadModelAndScore,
    _prefer_massive_model_table_for_inference,
)


class FakeDataFrame:
    def __init__(self, columns, collected=None):
        self.columns = list(columns)
        self._collected = collected
        self.saved_as = None
        self.smart_saved_as = None

    def select(self, *columns):
        return FakeDataFrame(columns)

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
    def __init__(self):
        self.version = 3
        self.model_ = ["MODEL_TABLE", "PIPELINE_TABLE"]
        self._predict_output_table_names = ["PRED_OUT", "PRED_STATS"]
        self.predict_model_state = None

    def predict(self, data, **kwargs):
        self.predict_model_state = list(self.model_) if isinstance(self.model_, list) else self.model_


class ScoreModelStub:
    def __init__(self):
        self.version = 5
        self.model_ = ["MODEL_TABLE", "PIPELINE_TABLE"]
        self._score_output_table_names = ["SCORE_OUT", "SCORE_STATS"]
        self.score_model_state = None

    def score(self, data, **kwargs):
        self.score_model_state = list(self.model_) if isinstance(self.model_, list) else self.model_


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


class TestMassiveAutomaticModelNormalization(unittest.TestCase):
    def test_helper_drops_pipeline_table_from_loaded_model_list(self):
        class Model:
            def __init__(self):
                self.model_ = ["MODEL_TABLE", "PIPELINE_TABLE"]

        model = Model()

        normalized = _prefer_massive_model_table_for_inference(model)

        self.assertIs(normalized, model)
        self.assertEqual(model.model_, ["MODEL_TABLE", None])

    def test_predict_tool_normalizes_loaded_model_before_predict(self):
        model = PredictModelStub()
        storage = ModelStorageStub(model)
        conn = FakeConnectionContext(
            {
                "PREDICT_INPUT": FakeDataFrame(["GROUP_ID", "TIMESTAMP"]),
                "PRED_OUT": FakeDataFrame(["GROUP_ID", "TIMESTAMP", "FORECAST"]),
                "PRED_STATS": FakeDataFrame([], pd.DataFrame([["RMSE", 1.23]], columns=["STAT_NAME", "STAT_VALUE"])),
            }
        )

        with patch("hana_ai.tools.hana_ml_tools.massive_automatic_timeseries_tools.ModelStorage", return_value=storage):
            tool = make_tool(MassiveAutomaticTimeSeriesLoadModelAndPredict, conn)
            result = json.loads(
                tool._run(
                    predict_table="PREDICT_INPUT",
                    key="TIMESTAMP",
                    group_key="GROUP_ID",
                    name="MODEL",
                    version=1,
                )
            )

        self.assertEqual(model.predict_model_state, ["MODEL_TABLE", None])
        self.assertEqual(result["predicted_results_table"], "PREDICT_RESULT_PREDICT_INPUT_MODEL_3")

    def test_score_tool_normalizes_loaded_model_before_score(self):
        model = ScoreModelStub()
        storage = ModelStorageStub(model)
        conn = FakeConnectionContext(
            {
                "SCORE_INPUT": FakeDataFrame(["GROUP_ID", "TIMESTAMP", "VALUE"]),
                "SCORE_OUT": FakeDataFrame(["GROUP_ID", "TIMESTAMP", "VALUE", "FORECAST"]),
                "SCORE_STATS": FakeDataFrame([], pd.DataFrame([["RMSE", 0.91]], columns=["STAT_NAME", "STAT_VALUE"])),
            }
        )

        with patch("hana_ai.tools.hana_ml_tools.massive_automatic_timeseries_tools.ModelStorage", return_value=storage):
            tool = make_tool(MassiveAutomaticTimeSeriesLoadModelAndScore, conn)
            result = json.loads(
                tool._run(
                    score_table="SCORE_INPUT",
                    key="TIMESTAMP",
                    group_key="GROUP_ID",
                    endog="VALUE",
                    name="MODEL",
                    version=1,
                )
            )

        self.assertEqual(model.score_model_state, ["MODEL_TABLE", None])
        self.assertEqual(result["scored_results_table"], "SCORE_RESULT_SCORE_INPUT_MODEL_5")


if __name__ == "__main__":
    unittest.main()