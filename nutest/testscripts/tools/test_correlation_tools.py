import unittest
import json
import numpy as np
from hana_ml.dataframe import create_dataframe_from_pandas
from hana_ai.tools.hana_ml_tools.correlation_tools import Correlation
from testML_BaseTestClass import TestML_BaseTestClass

class TestCorrelationTools(TestML_BaseTestClass):
    tableDef = {
        '#CORRELATION_SIM_DATA_TBL':
            'CREATE LOCAL TEMPORARY COLUMN TABLE #CORRELATION_SIM_DATA_TBL("ID" INTEGER, "XVAL" DOUBLE, "YVAL" DOUBLE)'
    }

    def setUp(self):
        super(TestCorrelationTools, self).setUp()
        self._createTable("#CORRELATION_SIM_DATA_TBL")
        np.random.seed(3)
        val = np.random.rand(32, 2)
        data_list = [(int(i), val[i,0], val[i,1]) for i in range(32)]
        self._insertData('#CORRELATION_SIM_DATA_TBL', data_list)

    def tearDown(self):
        self._dropTableIgnoreError("#CORRELATION_SIM_DATA_TBL")
        super(TestCorrelationTools, self).tearDown()

    def test_fft_tools(self):
        tool = Correlation(connection_context=self.conn)
        tool_input = dict(input_table="#CORRELATION_SIM_DATA_TBL",
                          key="ID",
                          x='XVAL',
                          #y='YVAL',
                          method='fft',
                          max_lag=4,
                          calculate_pacf=True,
                          calculate_confint=True,
                          alpha=0.05,
                          bartlett=True)
        result = json.loads(tool.run(tool_input=tool_input))
        self.assertTrue(result["correlation_result_table"] == "CORRELATION_SIM_DATA_TBL_CORRELATION_RESULT")

if __name__ == '__main__':
    unittest.main()