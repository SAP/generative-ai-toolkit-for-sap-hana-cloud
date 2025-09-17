import json
from hana_ai.tools.df_tools.ts_visualizer_tools import TimeSeriesDatasetReport
from testML_BaseTestClass import TestML_BaseTestClass
import unittest

class TestTimeSeriesDatasetReport(TestML_BaseTestClass):
    tableDef = {
        '#HANAI_DATA_TBL_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_RAW ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)'
    }
    def setUp(self):
        super(TestTimeSeriesDatasetReport, self).setUp()
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
        super(TestTimeSeriesDatasetReport, self).tearDown()

    def test_forecast_line_plot(self):
        tool = TimeSeriesDatasetReport(connection_context=self.conn)
        result = json.loads(tool.run({"select_statement": "select * from #HANAI_DATA_TBL_RAW", "key": "TIMESTAMP", "endog": "VALUE"}))
        #print(result)
        self.assertTrue("html_file" in result)
        self.assertTrue("hanaml_report/ts_report.html" in result['html_file'])

if __name__ == '__main__':
    unittest.main()