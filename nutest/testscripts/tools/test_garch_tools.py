import unittest
import json
import numpy as np
from hana_ml.dataframe import create_dataframe_from_pandas
from hana_ai.tools.hana_ml_tools.garch_tools import GARCHFitAndSave, GARCHLoadModelAndPredict
from testML_BaseTestClass import TestML_BaseTestClass

class TestGARCHTools(TestML_BaseTestClass):
    """
    Test class for GARCH tools.
    """
    tableDef = {
        '#GARCH_SIM_DATA_TBL':
            'CREATE LOCAL TEMPORARY TABLE #GARCH_SIM_DATA_TBL ("ID" INTEGER, "VAL" DOUBLE)'
    }

    def setUp(self):
        super(TestGARCHTools, self).setUp()
        self._createTable("#GARCH_SIM_DATA_TBL")
        np.random.seed(3)
        val = np.random.rand(16)
        data_list = [(i, val[i]) for i in range(16)]
        self._insertData('#GARCH_SIM_DATA_TBL', data_list)
 
    def tearDown(self):
        self._dropTableIgnoreError("#GARCH_SIM_DATA_TBL")
        super(TestGARCHTools, self).tearDown()

    def test_garch_fit_and_save(self):
        tool = GARCHFitAndSave(connection_context=self.conn)
        tool_input = dict(fit_table="#GARCH_SIM_DATA_TBL",
                          key="ID", endog="VAL",
                          name="GARCH_TEST_MODEL",
                          version=1)
        result = json.loads(tool.run(tool_input=tool_input))
        self.assertTrue(result['train_table/view']=="#GARCH_SIM_DATA_TBL")
        self.assertTrue(result['model_storage_name']=="GARCH_TEST_MODEL")
        self.assertTrue(int(result['model_storage_version'])==1)

    def test_garch_load_model_and_predict(self):
        tool = GARCHLoadModelAndPredict(connection_context=self.conn)
        tool_input = dict(name="GARCH_TEST_MODEL", horizon=5)
        result = json.loads(tool.run(tool_input=tool_input))
        self.assertTrue(result["predict_result_table"] == "GARCH_TEST_MODEL_1_PREDICT_RESULT")

if __name__ == '__main__':
    unittest.main()
