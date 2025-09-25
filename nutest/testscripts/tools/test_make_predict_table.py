"""
test for hana_ai.tools.hana_ml_tools.ts_make_predict_table module.
"""

import json
from hana_ai.tools.hana_ml_tools.ts_make_predict_table import TSMakeFutureTableTool, TSMakeFutureTableForMassiveForecastTool
from testML_BaseTestClass import TestML_BaseTestClass

class TestTSMakePredictTableTools(TestML_BaseTestClass):
    tableDef = {
        '#HANAI_DATA_TBL_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_RAW ("TIMESTAMP" TIMESTAMP, "GROUP_ID" INT, "VALUE" DOUBLE)',
    }
    def setUp(self):
        super(TestTSMakePredictTableTools, self).setUp()
        self._createTable('#HANAI_DATA_TBL_RAW')
        data_list_raw = [
            ('1900-01-01 12:00:00', 0, 998.23063348829),
            ('1900-01-01 13:00:00', 0, 997.984413594973),
            ('1900-01-01 14:00:00', 0, 998.076511123945),
            ('1900-01-01 15:00:00', 0, 997.9165407258),
            ('1900-01-01 16:00:00', 0, 997.438758925335),
            ('1900-01-01 12:00:00', 1, 998.23063348829),
            ('1900-01-01 13:00:00', 1, 997.984413594973),
            ('1900-01-01 14:00:00', 1, 998.076511123945),
            ('1900-01-01 15:00:00', 1, 997.9165407258),
            ('1900-01-01 16:00:00', 1, 997.438758925335),
            ]
        self._insertData('#HANAI_DATA_TBL_RAW', data_list_raw)

    def tearDown(self):
        self._dropTableIgnoreError('#HANAI_DATA_TBL_RAW')
        super(TestTSMakePredictTableTools, self).tearDown()

    def test_tsmake_future_table_tool(self):
        self.conn.table("#HANAI_DATA_TBL_RAW").filter('"GROUP_ID"=0').save("#HANAI_DATA_TBL_PREDICT_RAW")
        tool = TSMakeFutureTableTool(connection_context=self.conn)
        result = tool.run({"train_table": "#HANAI_DATA_TBL_PREDICT_RAW", "key": "TIMESTAMP"})
        self.assertTrue(result.startswith('Successfully created the forecast input table'))
        self.conn.drop_table("#HANAI_DATA_TBL_PREDICT_RAW")

    def test_tsmake_future_table_for_massive_forecast_tool(self):
        tool = TSMakeFutureTableForMassiveForecastTool(connection_context=self.conn)
        result = tool.run({"train_table": "#HANAI_DATA_TBL_RAW", "key": "TIMESTAMP", "group_key": "GROUP_ID"})
        self.assertTrue(result.startswith('Successfully created the forecast input table'))
        print(result)
