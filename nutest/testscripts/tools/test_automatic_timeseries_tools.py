#unittest for Automatic timeseries tools
import json
from hana_ai.tools.hana_ml_tools.automatic_timeseries_tools import AutomaticTimeSeriesFitAndSave, AutomaticTimeseriesLoadModelAndPredict, AutomaticTimeseriesLoadModelAndScore
from testML_BaseTestClass import TestML_BaseTestClass
from hana_ml.model_storage import ModelStorage

class TestAutomaticTimeSeriesTools(TestML_BaseTestClass):
    tableDef = {
        '#HANAI_DATA_TBL_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_RAW ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)',
        '#HANAI_DATA_TBL_SCORE_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_SCORE_RAW ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)',
    }
    def setUp(self):
        super(TestAutomaticTimeSeriesTools, self).setUp()
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
        self._dropTableIgnoreError('#HANAI_DATA_TBL_RAW')
        self._dropTableIgnoreError('#HANAI_DATA_TBL_PREDICT_RAW')
        self._dropTableIgnoreError('#HANAI_DATA_TBL_SCORE_RAW')
        super(TestAutomaticTimeSeriesTools, self).tearDown()

    def test_fit_and_save(self):
        tool = AutomaticTimeSeriesFitAndSave(connection_context=self.conn)
        result = json.loads(tool.run({"fit_table": "#HANAI_DATA_TBL_RAW", "key": "TIMESTAMP", "endog": "VALUE", "name": "AUTOML_MODEL", "version": 1, 'fold_num': 2}))
        self.assertTrue(result['trained_table']=="#HANAI_DATA_TBL_RAW")
        self.assertTrue(result['model_storage_name']=="AUTOML_MODEL")
        self.assertTrue(int(result['model_storage_version'])==1)

    def test_load_model_and_predict(self):
        tool = AutomaticTimeseriesLoadModelAndPredict(connection_context=self.conn)
        result = json.loads(tool.run({"predict_table": "#HANAI_DATA_TBL_PREDICT_RAW", "key": "TIMESTAMP", "name": "AUTOML_MODEL", "version": 1}))
        print(result)
        self.assertTrue(result['predicted_results_table']=="AUTOML_MODEL_1_PREDICTED_RESULTS")
        self.conn.drop_table('AUTOML_MODEL_1_PREDICTED_RESULTS')

    def test_load_model_and_score(self):
        tool = AutomaticTimeseriesLoadModelAndScore(connection_context=self.conn)
        result = json.loads(tool.run({"score_table": "#HANAI_DATA_TBL_SCORE_RAW", "key": "TIMESTAMP", "endog": "VALUE", "name": "AUTOML_MODEL", "version": 1}))
        print(result)
        self.assertTrue(result['scored_results_table']=="AUTOML_MODEL_1_SCORED_RESULTS")
        self.conn.drop_table('AUTOML_MODEL_1_SCORED_RESULTS')
