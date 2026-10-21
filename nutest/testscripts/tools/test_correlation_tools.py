import unittest
import json
import numpy as np
import pandas as pd
from hana_ml.dataframe import create_dataframe_from_pandas
from hana_ai.tools.hana_ml_tools.correlation_tools import Correlation
from testML_BaseTestClass import TestML_BaseTestClass

class TestCorrelationTools(TestML_BaseTestClass):
    tableDef = {
        '#CORRELATION_SIM_DATA_TBL':
            'CREATE LOCAL TEMPORARY COLUMN TABLE #CORRELATION_SIM_DATA_TBL("ID" INTEGER, ' +\
            '"DATE" TIMESTAMP, "XVAL" DOUBLE, "YVAL" DOUBLE)'
    }

    def setUp(self):
        super(TestCorrelationTools, self).setUp()
        self._createTable("#CORRELATION_SIM_DATA_TBL")
        np.random.seed(3)
        dates = pd.date_range(start='2020-02-02', periods=32)
        val = np.random.rand(32, 2)
        data_list = [(int(i), dates[i], val[i,0], val[i,1]) for i in range(32)]
        self._insertData('#CORRELATION_SIM_DATA_TBL', data_list)

    def tearDown(self):
        self._dropTableIgnoreError("#CORRELATION_SIM_DATA_TBL")
        super(TestCorrelationTools, self).tearDown()

    def test_correlation_tools(self):
        tool = Correlation(connection_context=self.conn)
        cc = self.conn
        tool_input1 = dict(table_name="#CORRELATION_SIM_DATA_TBL",
                           key="ID",
                           x='XVAL',
                           #y='YVAL',
                           method='fft',
                           max_lag=4,
                           calculate_pacf=True,
                           calculate_confint=True,
                           alpha=0.05,
                           bartlett=True)
        result1 = json.loads(tool.run(tool_input=tool_input1))
        result_tab1 = cc.table("CORRELATION_SIM_DATA_TBL_CORRELATION_RESULT")
        corr_sq1 = result_tab1.sort(result_tab1.columns[0]).collect().iloc[:,1]
        tool_input2 = tool_input1
        tool_input2['key'] = 'DATE'
        result2 = json.loads(tool.run(tool_input=tool_input2))
        result_tab2 = cc.table("CORRELATION_SIM_DATA_TBL_CORRELATION_RESULT")
        corr_sq2 = result_tab2.sort(result_tab2.columns[0]).collect().iloc[:,1]
        self.assertTrue(all(corr_sq1 == corr_sq2))
        tool_input_err = dict(table_name="#CORRELATION_SIM_DATA_TBL",
                              key="ID", x='XVAL',
                              y='YVAL',
                              calculate_confint=True)
        err_result = tool.run(tool_input=tool_input_err)
        msg = "confidence intervals are only applicable to the autocorrelation of one time-series"
        self.assertTrue(msg in err_result)
        tool_input_err2 = dict(table_name="#CORRELATION_SIM_DATA_TBL",
                               key="ID", x='XVAL',
                               y='ZVAL')
        err_result2 = tool.run(tool_input=tool_input_err2)
        self.assertTrue(all(xx in err_result2 for xx in ['ValueError', 'ZVAL']))

if __name__ == '__main__':
    unittest.main()