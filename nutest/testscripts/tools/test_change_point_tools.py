import unittest
import json
import numpy as np
from hana_ml.dataframe import create_dataframe_from_pandas
from hana_ai.tools.hana_ml_tools.change_point_tools import BayesianChangePoint
from testML_BaseTestClass import TestML_BaseTestClass

class TestBayesianChangePointTools(TestML_BaseTestClass):
    tableDef = {
        '#CHANGEPOINT_SIM_DATA_TBL':
            'CREATE LOCAL TEMPORARY COLUMN TABLE #CHANGEPOINT_SIM_DATA_TBL("ID" INTEGER, "TS_VAL" DOUBLE)'
    }

    def setUp(self):
        super(TestBayesianChangePointTools, self).setUp()
        self._createTable("#CHANGEPOINT_SIM_DATA_TBL")
        np.random.seed(23)
        x = np.arange(256) / 16
        y1 = np.array(list(np.sin(x[:128] * np.pi)) + list(np.sin(x[128:] * 2 * np.pi)))
        y2 = np.array([1] * 72 + [2] * 184)
        y = y1 + y2
        data_list = [(int(i), y[i]) for i in range(256)]
        self._insertData('#CHANGEPOINT_SIM_DATA_TBL', data_list)

    def tearDown(self):
        self._dropTableIgnoreError("#CHANGEPOINT_SIM_DATA_TBL")
        super(TestBayesianChangePointTools, self).tearDown()

    def test_accuracy_measure_tools(self):
        tool = BayesianChangePoint(connection_context=self.conn)
        tool_input = dict(input_table="#CHANGEPOINT_SIM_DATA_TBL",
                          key="ID",
                          endog="TS_VAL",
                          max_tcp=1,
                          max_scp=1,
                          random_seed=23,
                          max_harmonic_order=2,
                          max_iter=2000)
        result = json.loads(tool.run(tool_input=tool_input))
        self.assertTrue(abs(72 - int(result["trend_change_points"])) < 10)
        self.assertTrue(abs(128 - int(result["seasonal_change_points"])) < 10)
        self.assertTrue("32" in result["periods"])
        self.assertTrue(result["bcpd_decomposed_table"] == "CHANGEPOINT_SIM_DATA_TBL_BCPD_DECOMPOSED")

if __name__ == '__main__':
    unittest.main()