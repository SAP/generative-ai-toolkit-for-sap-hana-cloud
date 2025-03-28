#unittest for accuracy measure tools
import json
from hana_ai.tools.hana_ml_tools.ts_accuracy_measure_tools import AccuracyMeasure
from testML_BaseTestClass import TestML_BaseTestClass

class TestTSAccuracyMeasure(TestML_BaseTestClass):
    tableDef = {
        '#HANAI_DATA_TBL_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_RAW ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)',
        '#HANAI_DATA_TBL_PREDICT_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_PREDICT_RAW ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)',
    }
    def setUp(self):
        super(TestTSAccuracyMeasure, self).setUp()
        self._createTable('#HANAI_DATA_TBL_RAW')
        self._createTable('#HANAI_DATA_TBL_PREDICT_RAW')
        data_list_actual = [
            ('1900-01-01 12:00:00', 998.23063348829),
            ('1900-01-01 13:00:00', 997.984413594973),
            ('1900-01-01 14:00:00', 998.076511123945),
            ('1900-01-01 15:00:00', 997.9165407258),
            ('1900-01-01 16:00:00', 997.438758925335),
            ]
        data_list_predict = [
            ('1900-01-01 12:00:00', 998.23063348829),
            ('1900-01-01 13:00:00', 997.984413594973),
            ('1900-01-01 14:00:00', 998.076511123945),
            ('1900-01-01 15:00:00', 997.9165407258),
            ('1900-01-01 16:00:00', 997.438758925335),
            ]
        self._insertData('#HANAI_DATA_TBL_RAW', data_list_actual)
        self._insertData('#HANAI_DATA_TBL_PREDICT_RAW', data_list_predict)

    def tearDown(self):
        self._dropTableIgnoreError('#HANAI_DATA_TBL_RAW')
        self._dropTableIgnoreError('#HANAI_DATA_TBL_PREDICT_RAW')
        super(TestTSAccuracyMeasure, self).tearDown()

    def test_accuracy_measure(self):
        tool = AccuracyMeasure(connection_context=self.conn)
        result = json.loads(tool.run({"predict_table": "#HANAI_DATA_TBL_PREDICT_RAW", 
                                      "actual_table": "#HANAI_DATA_TBL_RAW",
                                      "predict_key": "TIMESTAMP",
                                      "actual_key": "TIMESTAMP",
                                      "predict_target": "VALUE",
                                      "actual_target": "VALUE",
                                      "evaluation_metric": ["MSE", "MAD", "MAPE", "RMSE", "SPEC"],
                                      }))
        expected_result = {
            'MSE': 0.0,
            'MAD': 0.0,
            'MAPE': 0.0,
            'RMSE': 0.0,
            "SPEC": 0.0
        }
        self.assertTrue(result['MSE']==expected_result['MSE'])
        self.assertTrue(result['MAD']==expected_result['MAD'])
        self.assertTrue(result['MAPE']==expected_result['MAPE'])
        self.assertTrue(result['RMSE']==expected_result['RMSE'])
        self.assertTrue(result['SPEC']==expected_result['SPEC'])
