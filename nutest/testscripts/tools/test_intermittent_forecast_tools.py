import json
from hana_ai.tools.hana_ml_tools.intermittent_forecast_tools import IntermittentForecast
from testML_BaseTestClass import TestML_BaseTestClass

class TestIntermittentForecastTools(TestML_BaseTestClass):
    tableDef = {
        '#HANAI_DATA_TBL_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_RAW ("ID" INT, "VALUE" DOUBLE)',
    }
    def setUp(self):
        super(TestIntermittentForecastTools, self).setUp()
        self._createTable('#HANAI_DATA_TBL_RAW')
        data_list_raw = [
            (1, 998.23063348829),
            (2, 0),
            (3, 998.076511123945),
            (4, 0),
            (5, 997.438758925335),
            ]
        self._insertData('#HANAI_DATA_TBL_RAW', data_list_raw)

    def tearDown(self):
        self._dropTableIgnoreError('#HANAI_DATA_TBL_RAW')
        super(TestIntermittentForecastTools, self).tearDown()

    def test_IntermittentForecast(self):
        tool = IntermittentForecast(connection_context=self.conn)
        result = json.loads(tool.run({"table_name": "#HANAI_DATA_TBL_RAW", "key": "ID", "endog": "VALUE"}))
        print(result)
        expected_result = {'predicted_result_table': 'HANAI_DATA_TBL_RAW_INTERMITTENT_FORECAST_RESULT', 'LAST_DEMAND': 998.1278852453934, 'LAST_INTERVAL': 1.3333333333333333, 'OPT_P': 2.0, 'OPT_Q': 2.0}
        self.assertTrue(result['predicted_result_table']==expected_result['predicted_result_table'])
        self.assertTrue(result['LAST_DEMAND']==expected_result['LAST_DEMAND'])
        self.assertTrue(result['LAST_INTERVAL']==expected_result['LAST_INTERVAL'])
        self.assertTrue(result['OPT_P']==expected_result['OPT_P'])
        self.assertTrue(result['OPT_Q']==expected_result['OPT_Q'])
