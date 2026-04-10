import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from hana_ai.tools.hana_ml_tools.ts_visualizer_tools import ForecastLinePlot


class FakeDataFrame:
    def __init__(self, columns, row_count=1, collected=None):
        self.columns = list(columns)
        self._row_count = row_count
        self._collected = collected or {}

    def count(self):
        return self._row_count

    def collect(self):
        if isinstance(self._collected, pd.DataFrame):
            return self._collected
        return pd.DataFrame(self._collected)

    def __getitem__(self, column_name):
        column_values = self._collected.get(column_name, [None])
        return FakeDataFrame([column_name], row_count=self._row_count, collected={column_name: column_values})


class FakeConnectionContext:
    def __init__(self, tables):
        self.tables = tables

    def has_table(self, name, schema=None):
        return name in self.tables

    def table(self, name, schema=None):
        return self.tables[name]


class FigureStub:
    def to_html(self, full_html=True):
        return "<html><body>plot</body></html>"

    def show(self):
        return None


def make_tool(connection_context):
    tool = object.__new__(ForecastLinePlot)
    object.__setattr__(tool, "connection_context", connection_context)
    object.__setattr__(tool, "return_direct", False)
    object.__setattr__(tool, "bas", True)
    return tool


class TestForecastLinePlotValidation(unittest.TestCase):
    def test_actual_table_name_alias_is_supported(self):
        predict_df = FakeDataFrame(["BOOKING_DATE", "YHAT"], row_count=2)
        actual_df = FakeDataFrame(["BOOKING_DATE", "REFUNDS"], row_count=2)
        tool = make_tool(FakeConnectionContext({"PREDICT_RESULT": predict_df, "ACTUALS": actual_df}))

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "hana_ai.tools.hana_ml_tools.ts_visualizer_tools.forecast_line_plot",
            return_value=FigureStub(),
        ) as plot_mock:
            result = json.loads(
                tool._run(
                    predict_result="PREDICT_RESULT",
                    actual_table_name="ACTUALS",
                    output_dir=tmpdir,
                )
            )

            self.assertTrue(Path(result["html_file"]).exists())

        self.assertIs(plot_mock.call_args[0][0], predict_df)
        self.assertIs(plot_mock.call_args[0][1], actual_df)

    def test_empty_prediction_result_returns_clear_error(self):
        predict_df = FakeDataFrame(["BOOKING_DATE", "YHAT"], row_count=0)
        tool = make_tool(FakeConnectionContext({"PREDICT_RESULT": predict_df}))

        result = json.loads(tool._run(predict_result="PREDICT_RESULT"))

        self.assertIn("is empty", result["error"])
        self.assertEqual(result["predict_result"], "PREDICT_RESULT")

    def test_non_standard_prediction_column_name_is_accepted(self):
        predict_df = FakeDataFrame(["BOOKING_DATE", "Predicted REFUNDS"], row_count=2)
        tool = make_tool(FakeConnectionContext({"PREDICT_RESULT": predict_df}))

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "hana_ai.tools.hana_ml_tools.ts_visualizer_tools.forecast_line_plot",
            return_value=FigureStub(),
        ):
            result = json.loads(tool._run(predict_result="PREDICT_RESULT", output_dir=tmpdir))
            self.assertTrue(Path(result["html_file"]).exists())

    def test_only_key_column_returns_clear_error(self):
        predict_df = FakeDataFrame(["BOOKING_DATE"], row_count=2)
        tool = make_tool(FakeConnectionContext({"PREDICT_RESULT": predict_df}))

        result = json.loads(tool._run(predict_result="PREDICT_RESULT"))

        self.assertIn("does not contain any value columns", result["error"])
        self.assertIn("suggested_fix", result)


if __name__ == "__main__":
    unittest.main()