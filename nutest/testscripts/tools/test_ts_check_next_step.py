"""Unit tests for the next-step guidance in ``ts_check`` outputs.

We don't hit HANA; instead we mock the PAL/hana_ml calls that ``ts_char`` /
``ts_char_massive`` make and assert the appended guidance points the agent
at ``get_automl_config_dict`` / ``modify_automl_config_dict`` with the right
boolean flags.
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from hana_ai.tools.hana_ml_tools import ts_check_tools


def _make_df(*, endog="VAL", key="TS", key_is_int=True, zero_fraction=0.0, row_count=100):
    """Build a MagicMock that mimics the ``hana_ml`` DataFrame surface used by
    ``ts_char``: get_table_structure, count, filter+count, add_id, select.
    """
    df = MagicMock()
    key_type = "INT" if key_is_int else "TIMESTAMP"
    df.get_table_structure.return_value = {key: key_type, endog: "DOUBLE"}
    df.count.return_value = row_count
    # __getitem__(key).min()/.max()
    col = MagicMock()
    col.min.return_value = 0
    col.max.return_value = row_count - 1
    df.__getitem__.return_value = col
    # filter(...).count() -> zero_values
    filt = MagicMock()
    filt.count.return_value = int(zero_fraction * row_count)
    df.filter.return_value = filt
    # add_id(...) is a no-op returning the same shape
    df.add_id.return_value = df
    return df


def _stub_pal(*, seasonal=False, trend=0):
    """Patch the three PAL calls ``ts_char`` performs. Returns the patch objects
    so the caller can start()/stop() them itself."""
    stationarity_ret = MagicMock()
    stationarity_ret.collect.return_value = pd.DataFrame(
        [{"STATS_NAME": "stationarity", "STATS_VALUE": 1}]
    )

    trend_frame = MagicMock()
    trend_frame.collect.return_value = pd.DataFrame(
        [{"STAT_NAME": "TREND", "STAT_VALUE": trend}]
    )

    seasonal_frame = MagicMock()
    seasonal_val = 1 if seasonal else 0
    seasonal_frame.collect.return_value = pd.DataFrame(
        [
            {"STAT_NAME": "SEASONAL", "STAT_VALUE": seasonal_val},
            {"STAT_NAME": "PERIOD", "STAT_VALUE": 14 if seasonal else 0},
        ]
    )

    return [
        patch.object(ts_check_tools, "stationarity_test", return_value=stationarity_ret),
        patch.object(ts_check_tools, "trend_test", return_value=(trend_frame,)),
        patch.object(ts_check_tools, "seasonal_decompose", return_value=(seasonal_frame,)),
    ]


class TestTsCharNextStep(unittest.TestCase):
    def _run_ts_char(self, **stub_kwargs):
        df = _make_df(**{k: v for k, v in stub_kwargs.items() if k in ("zero_fraction", "row_count")})
        patches = _stub_pal(
            seasonal=stub_kwargs.get("seasonal", False),
            trend=stub_kwargs.get("trend", 0),
        )
        for p in patches:
            p.start()
        try:
            return ts_check_tools.ts_char(df, "TS", "VAL")
        finally:
            for p in patches:
                p.stop()

    def test_next_step_points_at_pal_config_tools(self):
        report = self._run_ts_char(seasonal=True)
        self.assertIn("get_pal_pipeline_info", report)
        self.assertIn("modify_automl_config_dict", report)
        self.assertIn("pipeline_type='timeseries'", report)
        self.assertIn("DO NOT hand-author", report)
        self.assertIn("Seasonality", report)  # the anti-pattern example

    def test_flags_reflect_signals(self):
        report = self._run_ts_char(zero_fraction=0.2, seasonal=True, trend=1)
        self.assertIn("has_intermittency=True", report)
        self.assertIn("has_seasonality=True", report)
        self.assertIn("has_trend=True", report)

    def test_flags_default_false(self):
        report = self._run_ts_char(zero_fraction=0.0, seasonal=False, trend=0)
        self.assertIn("has_intermittency=False", report)
        self.assertIn("has_seasonality=False", report)
        self.assertIn("has_trend=False", report)

    def test_low_zero_fraction_below_threshold(self):
        # 5% is the threshold; 0.04 -> False, 0.06 -> True.
        low = self._run_ts_char(zero_fraction=0.04)
        self.assertIn("has_intermittency=False", low)
        high = self._run_ts_char(zero_fraction=0.06)
        self.assertIn("has_intermittency=True", high)

    def test_downward_trend_also_sets_flag(self):
        report = self._run_ts_char(trend=-1)
        self.assertIn("has_trend=True", report)


if __name__ == "__main__":
    unittest.main()
