import unittest
import json
import numpy as np
from hana_ml.dataframe import create_dataframe_from_pandas
from hana_ai.tools.hana_ml_tools.accuracy_measure_tools import AccuracyMeasure
from testML_BaseTestClass import TestML_BaseTestClass

class TestAccuracyMeasureTools(TestML_BaseTestClass):
    tableDef = {
        '#ACC_MEASURE_SIM_DATA_TRI_COL':
            'CREATE LOCAL TEMPORARY COLUMN TABLE #ACC_MEASURE_SIM_DATA_TRI_COL("ID" INTEGER, "TRUE_VAL" DOUBLE, "FCST_VAL" DOUBLE)',
        '#ACC_MEASURE_SIM_DATA_TWO_COL':
            'CREATE LOCAL TEMPORARY COLUMN TABLE #ACC_MEASURE_SIM_DATA_TWO_COL("TRUE_VAL" DOUBLE, "FCST_VAL" DOUBLE)'
    }

    def setUp(self):
        super(TestAccuracyMeasureTools, self).setUp()
        self._createTable("#ACC_MEASURE_SIM_DATA_TRI_COL")
        self._createTable("#ACC_MEASURE_SIM_DATA_TWO_COL")
        np.random.seed(23)
        val = np.random.rand(32, 2)
        data_list = [(int(i), val[i,0], val[i,1]) for i in range(32)]
        self._insertData('#ACC_MEASURE_SIM_DATA_TRI_COL', data_list)
        data_list = [(d[1], d[2]) for d in data_list]
        self._insertData('#ACC_MEASURE_SIM_DATA_TWO_COL', data_list)


    def tearDown(self):
        self._dropTableIgnoreError("#ACC_MEASURE_SIM_DATA_TRI_COL")
        self._dropTableIgnoreError("#ACC_MEASURE_SIM_DATA_TWO_COL")
        super(TestAccuracyMeasureTools, self).tearDown()

    def test_accuracy_measure_tools(self):
        tool = AccuracyMeasure(connection_context=self.conn)
        tool_input = dict(input_table="#ACC_MEASURE_SIM_DATA_TRI_COL",
                          evaluation_metric=['mad', 'mape', 'spec'],
                          ignore_zero=True,
                          alpha1=0.1,
                          alph2=0.2)
        result = json.loads(tool.run(tool_input=tool_input))
        self.assertTrue(all(key in list(result.keys()) for key in ["MAD", "MAPE", "SPEC"]))
        tool_input2 = dict(input_table="#ACC_MEASURE_SIM_DATA_TWO_COL",
                           evaluation_metric=['mad', 'mape'],
                           ignore_zero=True,
                           alpha1=0.1,
                           alph2=0.2)
        result2 = json.loads(tool.run(tool_input=tool_input2))
        self.assertTrue(result["MAD"] == result2["MAD"] and result["MAPE"] == result2["MAPE"])

if __name__ == '__main__':
    unittest.main()