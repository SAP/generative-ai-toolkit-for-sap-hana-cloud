#unittest for accuracy measure tools
import json
import datetime
import numpy as np
from hana_ai.tools.hana_ml_tools.ts_accuracy_measure_tools import AccuracyMeasure
from hana_ml.algorithms.pal.tsa.accuracy_measure import accuracy_measure
from testML_BaseTestClass import TestML_BaseTestClass

class TestTSAccuracyMeasure(TestML_BaseTestClass):
    tableDef = {
        '#HANAI_DATA_TBL_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_RAW ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)',
        '#HANAI_DATA_TBL_PREDICT_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_PREDICT_RAW ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)',
        '#ACC_MEASURE_REAL_TBL':
            'CREATE LOCAL TEMPORARY TABLE #ACC_MEASURE_REAL_TBL ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)',
        '#ACC_MEASURE_PREDICT_TBL':
            'CREATE LOCAL TEMPORARY TABLE #ACC_MEASURE_PREDICT_TBL ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)',
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
        self._createTable('#ACC_MEASURE_REAL_TBL')
        self._createTable('#ACC_MEASURE_PREDICT_TBL')
        date_string = "2025-01-01"
        date_obj = datetime.datetime.strptime(date_string, "%Y-%m-%d")
        date_range = [str(date_obj + datetime.timedelta(days=31 - i)) for i in range(32)]
        np.random.seed(23)
        X = np.random.rand(32) * 100
        Y = X + np.random.normal(size=32)
        data_list_actual2 = [(date_range[i], X[i]) for i in range(32)]
        data_list_predict2 = [(date_range[i], Y[i]) for i in range(32)]
        self._insertData('#ACC_MEASURE_REAL_TBL', data_list_actual2)
        self._insertData('#ACC_MEASURE_PREDICT_TBL', data_list_predict2)

    def tearDown(self):
        self._dropTableIgnoreError('#HANAI_DATA_TBL_RAW')
        self._dropTableIgnoreError('#HANAI_DATA_TBL_PREDICT_RAW')
        self._dropTableIgnoreError('#ACC_MEASURE_REAL_TBL')
        self._dropTableIgnoreError('#ACC_MEASURE_PREDICT_TBL')
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
        result2 = tool.run({"predict_table": "#ACC_MEASURE_PREDICT_TBL", 
                            "actual_table": "#ACC_MEASURE_REAL_TBL",
                            "predict_key": "TIMESTAMP",
                            "actual_key": "TIMESTAMP",
                            "predict_target": "VALUE",
                            "actual_target": "VALUE",
                            "evaluation_metric": "spec"})
        spec_from_tool = json.loads(result2)['SPEC']
        acc_df = self.conn.table("#ACC_MEASURE_PREDICT_TBL")\
        .rename_columns({"TIMESTAMP": "TIMESTAMP_P", "VALUE": "VALUE_P"})\
        .join(self.conn.table("#ACC_MEASURE_REAL_TBL"), "TIMESTAMP_P=TIMESTAMP")\
        .add_id("ID", ref_col="TIMESTAMP")[["ID", "VALUE", "VALUE_P"]]
        acc_measures = accuracy_measure(acc_df, evaluation_metric="spec")
        spec_val = acc_measures.collect().iloc[0,1]
        self.assertTrue(spec_from_tool == spec_val)

if __name__ == '__main__':
    unittest.main()
