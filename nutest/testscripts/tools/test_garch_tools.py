import unittest
import json
import numpy as np
import pandas as pd
from hana_ai.tools.hana_ml_tools.garch_tools import GARCHFitPredict
from testML_BaseTestClass import TestML_BaseTestClass

class TestGARCHTools(TestML_BaseTestClass):
    """
    Test class for GARCH tools.
    """
    tableDef = {
        '#GARCH_SIM_DATA_TBL':
            'CREATE LOCAL TEMPORARY TABLE #GARCH_SIM_DATA_TBL ("ID" INTEGER, "DATE" TIMESTAMP, "VAL" DOUBLE)'
    }

    def setUp(self):
        super(TestGARCHTools, self).setUp()
        self._createTable("#GARCH_SIM_DATA_TBL")
        np.random.seed(3)
        val = np.random.normal(size=16)
        dates = pd.date_range(start='2025-01-01', periods=16)
        data_list = [(i, str(dates[i]),val[i]) for i in range(16)]
        self._insertData('#GARCH_SIM_DATA_TBL', data_list)
 
    def tearDown(self):
        self._dropTableIgnoreError("#GARCH_SIM_DATA_TBL")
        self._dropTableIgnoreError("GARCH_SIM_DATA_TBL_PREDICT_RESULT")
        super(TestGARCHTools, self).tearDown()

    def test_garch_fit_predict(self):
        tool = GARCHFitPredict(connection_context=self.conn)
        tool_input = dict(table_name="#GARCH_SIM_DATA_TBL",
                          key="ID", endog="VAL",
                          forecast_length=5)
        result = json.loads(tool.run(tool_input=tool_input))
        self.assertTrue(result["garch_predict_result_table"] == "GARCH_SIM_DATA_TBL_PREDICT_RESULT")
        res1 = self.conn.table("GARCH_SIM_DATA_TBL_PREDICT_RESULT").collect().iloc[:,1]
        tool_input2 = dict(table_name="#GARCH_SIM_DATA_TBL",
                           key="DATE", endog="VAL",
                           forecast_length=5)
        tool.run(tool_input=tool_input2)
        res2 = self.conn.table("GARCH_SIM_DATA_TBL_PREDICT_RESULT").collect().iloc[:,1]
        self.assertTrue(all(x == y for (x,y) in zip(res1, res2)))
        #test of error message
        tool_input_err = dict(table_name="#GARCH_SIM_DATA_TBL",
                              key="TIME_STAMP",
                              endog="VAL",
                              forecast_length=5)
        err_res1 = tool.run(tool_input=tool_input_err)
        self.assertTrue(all(str_v in err_res1 for str_v in ['TIME_STAMP', 'ValueError']))
        tool_input_err = dict(table_name="#GARCH_SIM_DATA_TBL",
                              key="TIMESTAMP",
                              endog="VAL",
                              model_type='zgarch',
                              forecast_length=4)
        err_res2 = tool.run(tool_input=tool_input_err)
        self.assertTrue(all(str_v in err_res2 for str_v in ['ValueError', 'zgarch']))

if __name__ == '__main__':
    unittest.main()
