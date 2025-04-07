#unittest for Automatic timeseries tools
import json
import os
from pathlib import Path
import tempfile
from hana_ai.tools.hana_ml_tools.automatic_timeseries_tools import AutomaticTimeSeriesFitAndSave, AutomaticTimeseriesLoadModelAndPredict
from hana_ai.tools.hana_ml_tools.ts_visualizer_tools import ForecastLinePlot
from testML_BaseTestClass import TestML_BaseTestClass
from hana_ml.model_storage import ModelStorage

class TestForecastLinePlotTools(TestML_BaseTestClass):
    tableDef = {
        '#HANAI_DATA_TBL_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_RAW ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)',
        '#HANAI_DATA_TBL_SCORE_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_SCORE_RAW ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)',
    }
    def setUp(self):
        super(TestForecastLinePlotTools, self).setUp()
        self._createTable('#HANAI_DATA_TBL_RAW')
        self._createTable('#HANAI_DATA_TBL_SCORE_RAW')
        data_list_raw = [
            ('1900-01-01 12:00:00', 998.23063348829),
            ('1900-01-01 13:00:00', 997.984413594973),
            ('1900-01-01 14:00:00', 998.076511123945),
            ('1900-01-01 15:00:00', 997.9165407258),
            ('1900-01-01 16:00:00', 997.438758925335),
            ]
        data_list_predict_raw = [
            ('1900-01-01 17:00:00', 0),
            ('1900-01-01 18:00:00', 0),
            ]
        self._insertData('#HANAI_DATA_TBL_RAW', data_list_raw)
        self._insertData('#HANAI_DATA_TBL_SCORE_RAW', data_list_predict_raw)
        self.conn.table("#HANAI_DATA_TBL_SCORE_RAW").drop("VALUE").save("#HANAI_DATA_TBL_PREDICT_RAW")

    def tearDown(self):
        ms = ModelStorage(self.conn)
        ms.delete_model("AUTOML_MODEL", 1)
        self._dropTableIgnoreError('#HANAI_DATA_TBL_RAW')
        self._dropTableIgnoreError('#HANAI_DATA_TBL_PREDICT_RAW')
        self._dropTableIgnoreError('#HANAI_DATA_TBL_SCORE_RAW')
        super(TestForecastLinePlotTools, self).tearDown()

    def test_forecast_line_plot(self):
        tool = AutomaticTimeSeriesFitAndSave(connection_context=self.conn)
        tool.run({"fit_table": "#HANAI_DATA_TBL_RAW", "key": "TIMESTAMP", "endog": "VALUE", "name": "AUTOML_MODEL", "version": 1, 'fold_num': 2})
        tool = AutomaticTimeseriesLoadModelAndPredict(connection_context=self.conn)
        tool.run({"predict_table": "#HANAI_DATA_TBL_PREDICT_RAW", "key": "TIMESTAMP", "name": "AUTOML_MODEL", "version": 1})
        tool = ForecastLinePlot(connection_context=self.conn)
        result = json.loads(tool.run({"predict_table_name": "AUTOML_MODEL_1_PREDICTED_RESULTS", "actual_table_name": "#HANAI_DATA_TBL_RAW"}))
        expected_result = {'html_file': 'C:\\Users\\I308290\\AppData\\Local\\Temp\\hanaml_chart\\AUTOML_MODEL_1_PREDICTED_RESULTS_forecast_line_plot.html'}
        self.assertTrue(result['html_file']==expected_result['html_file'])
        self.assertTrue(Path(result['html_file']).exists())
