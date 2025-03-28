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
        result = json.loads(tool.run({"table_name": "#HANAI_DATA_TBL_RAW", "key": "ID", "endog": "VALUE", "accuracy_measure": "rmse"}))
        expected_result = {'predicted_result_table': 'HANAI_DATA_TBL_RAW_INTERMITTENT_FORECAST_RESULT', 'DEMAND_FORECAST': 997.9076977658694, 'RMSE': 0.2684688160121085, 'PROBABILITY_FORECAST': 0.5418550000000001}
        self.assertTrue(result['predicted_result_table']==expected_result['predicted_result_table'])
        self.assertAlmostEqual(result['DEMAND_FORECAST'], expected_result['DEMAND_FORECAST'], places=5)
        self.assertAlmostEqual(result['RMSE'], expected_result['RMSE'], places=5)
        self.assertAlmostEqual(result['PROBABILITY_FORECAST'], expected_result['PROBABILITY_FORECAST'], places=5)
