import unittest
import json
import numpy as np
from hana_ml.dataframe import create_dataframe_from_pandas
from hana_ai.tools.hana_ml_tools.accuracy_measure_tools import AccuracyMeasure
from testML_BaseTestClass import TestML_BaseTestClass

class TestAccuracyMeasureTools(TestML_BaseTestClass):
    tableDef = {
        '#ACC_MEASURE_SIM_DATA_TBL':
            'CREATE LOCAL TEMPORARY COLUMN TABLE #ACC_MEASURE_SIM_DATA_TBL("ID" INTEGER, "TRUE_VAL" DOUBLE, "FCST_VAL" DOUBLE)'
    }

    def setUp(self):
        super(TestAccuracyMeasureTools, self).setUp()
        self._createTable("#ACC_MEASURE_SIM_DATA_TBL")
        np.random.seed(23)
        val = np.random.rand(32, 2)
        data_list = [(int(i), val[i,0], val[i,1]) for i in range(32)]
        self._insertData('#ACC_MEASURE_SIM_DATA_TBL', data_list)

    def tearDown(self):
        self._dropTableIgnoreError("#ACC_MEASURE_SIM_DATA_TBL")
        super(TestAccuracyMeasureTools, self).tearDown()

    def test_accuracy_measure_tools(self):
        tool = AccuracyMeasure(connection_context=self.conn)
        tool_input = dict(input_table="#ACC_MEASURE_SIM_DATA_TBL",
                          evaluation_metric=['mad', 'mape', 'spec'],
                          ignore_zero=True,
                          alpha1=0.1,
                          alph2=0.2)
        result = json.loads(tool.run(tool_input=tool_input))
        self.assertTrue(all(key in list(result.keys()) for key in ["MAD", "MAPE", "SPEC"]))

if __name__ == '__main__':
    unittest.main()