#unittest for AdditiveModelForecastTools
import json
from hana_ai.tools.hana_ml_tools.additive_model_forecast_tools import AdditiveModelForecastFitAndSave, AdditiveModelForecastLoadModelAndPredict
from hana_ai.tools.hana_ml_tools.ts_visualizer_tools import ForecastLinePlot
from testML_BaseTestClass import TestML_BaseTestClass
from hana_ml.model_storage import ModelStorage
class TestAdditiveModelForecastTools(TestML_BaseTestClass):
    tableDef = {
        '#HANAI_DATA_TBL_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_RAW ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)',
        '#HANAI_DATA_TBL_PREDICT_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_PREDICT_RAW ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)',
    }
    def setUp(self):
        super(TestAdditiveModelForecastTools, self).setUp()
        self._createTable('#HANAI_DATA_TBL_RAW')
        self._createTable('#HANAI_DATA_TBL_PREDICT_RAW')
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
        self._insertData('#HANAI_DATA_TBL_PREDICT_RAW', data_list_predict_raw)

    def tearDown(self):
        self._dropTableIgnoreError('#HANAI_DATA_TBL_RAW')
        self._dropTableIgnoreError('#HANAI_DATA_TBL_PREDICT_RAW')
        super(TestAdditiveModelForecastTools, self).tearDown()

    def test_fit_and_save(self):
        tool = AdditiveModelForecastFitAndSave(connection_context=self.conn)
        result = json.loads(tool.run({"fit_table": "#HANAI_DATA_TBL_RAW", "key": "TIMESTAMP", "endog": "VALUE", "name": "HANAI_MODEL", "version": 1}))
        self.assertTrue(result['trained_table']=="#HANAI_DATA_TBL_RAW")
        self.assertTrue(result['model_storage_name']=="HANAI_MODEL")
        self.assertTrue(int(result['model_storage_version'])==1)

    def test_load_model_and_predict(self):
        tool = AdditiveModelForecastLoadModelAndPredict(connection_context=self.conn)
        result = json.loads(tool.run({"predict_table": "#HANAI_DATA_TBL_PREDICT_RAW", "key": "TIMESTAMP", "name": "HANAI_MODEL", "version": 1}))
        print(result)
        self.assertTrue(result['predicted_results_table']=="HANAI_MODEL_1_PREDICTED_RESULT")

        tool = ForecastLinePlot(connection_context=self.conn)
        result = json.loads(tool.run({"predict_table_name": "HANAI_MODEL_1_PREDICTED_RESULT", "actual_table_name": "#HANAI_DATA_TBL_RAW"}))

        self.conn.drop_table('HANAI_MODEL_1_PREDICTED_RESULT')
