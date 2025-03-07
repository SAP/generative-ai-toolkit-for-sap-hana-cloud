
import json
from testML_BaseTestClass import TestML_BaseTestClass
from hana_ai.tools.hana_ml_tools.ts_outlier_detection_tools import TSOutlierDetection

class TestTSOutlierDetectionTools(TestML_BaseTestClass):
    tableDef = {
        '#HANAI_DATA_TBL_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_RAW ("ID" INT, "VALUE" DOUBLE)',
    }
    def setUp(self):
        super(TestTSOutlierDetectionTools, self).setUp()
        self._createTable('#HANAI_DATA_TBL_RAW')
        data_list_raw = [
            (1, 998.23063348829),
            (2, 20000000),
            (3, 998.076511123945),
            (4, 998),
            (5, 997.438758925335),
            ]
        self._insertData('#HANAI_DATA_TBL_RAW', data_list_raw)

    def tearDown(self):
        self._dropTableIgnoreError('#HANAI_DATA_TBL_RAW')
        super(TestTSOutlierDetectionTools, self).tearDown()

    def test_TSOutlierDetection(self):
        from hana_ml.algorithms.pal.tsa.outlier_detection import OutlierDetectionTS
        tool = TSOutlierDetection(connection_context=self.conn)
        result = json.loads(tool.run({"table_name": "#HANAI_DATA_TBL_RAW", "key": "ID", "endog": "VALUE", "outlier_method": "isolationforest"}))
        expected_result = {'outliers': [2]}
        self.assertTrue(result['outliers']==expected_result['outliers'])
