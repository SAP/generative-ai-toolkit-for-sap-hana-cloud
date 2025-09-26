
import json
from testML_BaseTestClass import TestML_BaseTestClass
from hana_ai.tools.hana_ml_tools.massive_ts_outlier_detection_tools import MassiveTSOutlierDetection

class TestMassiveTSOutlierDetectionTools(TestML_BaseTestClass):
    tableDef = {
        '#HANAI_DATA_TBL_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_RAW ("GROUP_ID" INT, "ID" INT, "VALUE" DOUBLE)',
    }
    def setUp(self):
        super(TestMassiveTSOutlierDetectionTools, self).setUp()
        self._createTable('#HANAI_DATA_TBL_RAW')
        data_list_raw = [
            (0, 1, 998.23063348829),
            (0, 2, 20000000),
            (0, 3, 998.076511123945),
            (0, 4, 998),
            (0, 5, 997.438758925335),
            (1, 1, 998.23063348829),
            (1, 2, 20000000),
            (1, 3, 998.076511123945),
            (1, 4, 998),
            (1, 5, 997.438758925335),
            ]
        self._insertData('#HANAI_DATA_TBL_RAW', data_list_raw)

    def tearDown(self):
        self._dropTableIgnoreError('#HANAI_DATA_TBL_RAW')
        super(TestMassiveTSOutlierDetectionTools, self).tearDown()

    def test_TSOutlierDetection(self):
        tool = MassiveTSOutlierDetection(connection_context=self.conn)
        result = json.loads(tool.run({"table_name": "#HANAI_DATA_TBL_RAW", "group_key": "GROUP_ID", "key": "ID", "endog": "VALUE", "outlier_method": "isolationforest"}))
        expected_result = {'outliers': [{"group_id": '0', "outlier_ids": [2]}, {"group_id": '1', "outlier_ids": [2]}]}
        # 按group_id排序
        expected_sorted = sorted(expected_result['outliers'], key=lambda x: x["group_id"])
        actual_sorted = sorted(result['outliers'], key=lambda x: x["group_id"])

        # 比较排序后的列表
        self.assertEqual(len(expected_sorted), len(actual_sorted))
        for i in range(len(expected_sorted)):
            self.assertEqual(expected_sorted[i]["group_id"], actual_sorted[i]["group_id"])
            self.assertCountEqual(expected_sorted[i]["outlier_ids"], actual_sorted[i]["outlier_ids"])
