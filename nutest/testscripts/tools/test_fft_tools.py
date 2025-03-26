import unittest
import json
import numpy as np
from hana_ml.dataframe import create_dataframe_from_pandas
from hana_ai.tools.hana_ml_tools.fft_tools import FFT
from testML_BaseTestClass import TestML_BaseTestClass

class TestFFTTools(TestML_BaseTestClass):
    tableDef = {
        '#FFT_SIM_DATA_TBL':
            'CREATE LOCAL TEMPORARY TABLE #FFT_SIM_DATA_TBL ("ID" INTEGER, "REAL_VAL" DOUBLE, "IMAG_VAL" DOUBLE)'
    }

    def setUp(self):
        super(TestFFTTools, self).setUp()
        self._createTable("#FFT_SIM_DATA_TBL")
        np.random.seed(3)
        val = np.random.rand(32, 2)
        data_list = [(i, val[i,0], val[i,1]) for i in range(32)]
        self._insertData('#FFT_SIM_DATA_TBL', data_list)

    def tearDown(self):
        self._dropTableIgnoreError("#FFT_SIM_DATA_TBL")
        super(TestFFTTools, self).tearDown()

    def test_fft_tools(self):
        tool = FFT(connection_context=self.conn)
        tool_input = dict(input_table="#FFT_SIM_DATA_TBL",
                          key="ID", num_type="real")
        result = json.loads(tool.run(tool_input=tool_input))
        self.assertTrue(result['fft_result_table'] == "FFT_SIM_DATA_TBL_FFT_RESULT")

if __name__ == '__main__':
    unittest.main()