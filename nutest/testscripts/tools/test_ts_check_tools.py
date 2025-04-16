
import json
from testML_BaseTestClass import TestML_BaseTestClass
from hana_ai.tools.hana_ml_tools.ts_check_tools import TimeSeriesCheck, TrendTest, SeasonalityTest, StationarityTest, WhiteNoiseTest

class TestTSCheckTools(TestML_BaseTestClass):
    tableDef = {
        '#HANAI_DATA_TBL_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_RAW ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)'
    }
    def setUp(self):
        super(TestTSCheckTools, self).setUp()
        self._createTable('#HANAI_DATA_TBL_RAW')
        data_list_raw = [
            ('1900-01-01 12:00:00', 998.23063348829),
            ('1900-01-01 13:00:00', 997.984413594973),
            ('1900-01-01 14:00:00', 998.076511123945),
            ('1900-01-01 15:00:00', 997.9165407258),
            ('1900-01-01 16:00:00', 997.438758925335),
            ]
        self._insertData('#HANAI_DATA_TBL_RAW', data_list_raw)

    def tearDown(self):
        self._dropTableIgnoreError('#HANAI_DATA_TBL_RAW')
        super(TestTSCheckTools, self).tearDown()

    def test_TimeSeriesCheck(self):
        tool = TimeSeriesCheck(connection_context=self.conn)
        result = tool.run({"table_name": "#HANAI_DATA_TBL_RAW", "key": "TIMESTAMP", "endog": "VALUE"})
        expected_result = "Index: starts from 1900-01-01 12:00:00 to 1900-01-01 16:00:00. Time series length is 5\nIntermittent Test: proportion of zero values is 0.0\nStationarity Test: The stationary is 0.The kpss_stat is 0.499999.The p-value is 0.041666666666844876.The lags is 4.The number of observations is 5.The critical values is {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739}.\nTrend Test: Downward trend.\nSeasonality Test: The type is non-seasonal.The period is 0.The acf is -0.292111."
        self.assertTrue(result==expected_result)

    def test_TrendTest(self):
        tool = TrendTest(connection_context=self.conn)
        result = json.loads(tool.run({"table_name": "#HANAI_DATA_TBL_RAW", "key": "TIMESTAMP", "endog": "VALUE"}))
        expected_result = {'Trend': 'Downward trend.'}
        self.assertTrue(result['Trend']==expected_result['Trend'])

    def test_SeasonalityTest(self):
        tool = SeasonalityTest(connection_context=self.conn)
        result = json.loads(tool.run({"table_name": "#HANAI_DATA_TBL_RAW", "key": "TIMESTAMP", "endog": "VALUE"}))
        expected_result = {'type': 'non-seasonal', 'period': '0', 'acf': '-0.292111'}
        self.assertTrue(result['type']==expected_result['type'])
        self.assertTrue(result['period']==expected_result['period'])
        self.assertTrue(result['acf']==expected_result['acf'])

    def test_StationarityTest(self):
        tool = StationarityTest(connection_context=self.conn)
        result = json.loads(tool.run({"table_name": "#HANAI_DATA_TBL_RAW", "key": "TIMESTAMP", "endog": "VALUE"}))
        expected_result = {'stationary': '0', 'kpss_stat': '0.499999', 'p-value': '0.041666666666844876', 'lags': '4', 'number of observations': '5', 'critical values': "{'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739}"}
        self.assertTrue(result['stationary']==expected_result['stationary'])
        self.assertTrue(result['kpss_stat']==expected_result['kpss_stat'])
        self.assertTrue(result['p-value']==expected_result['p-value'])
        self.assertTrue(result['lags']==expected_result['lags'])
        self.assertTrue(result['number of observations']==expected_result['number of observations'])
        self.assertTrue(result['critical values']==expected_result['critical values'])

    def test_WhiteNoiseTest(self):
        tool = WhiteNoiseTest(connection_context=self.conn)
        result = json.loads(tool.run({"table_name": "#HANAI_DATA_TBL_RAW", "key": "TIMESTAMP", "endog": "VALUE"}))
        expected_result = {'WN': 1.0, 'Q': 0.05831685173473957, 'chi^2': 2.705543454095414}
        self.assertTrue(result['WN']==expected_result['WN'])
        self.assertTrue(result['Q']==expected_result['Q'])
        self.assertTrue(result['chi^2']==expected_result['chi^2'])
