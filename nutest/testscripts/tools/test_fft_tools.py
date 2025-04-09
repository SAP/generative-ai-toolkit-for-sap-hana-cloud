import unittest
import json
import datetime
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from hana_ml.dataframe import create_dataframe_from_pandas
from hana_ai.tools.hana_ml_tools.fft_tools import FFT
from testML_BaseTestClass import TestML_BaseTestClass

class TestFFTTools(TestML_BaseTestClass):
    tableDef = {
        '#FFT_SIM_DATA_TBL':
            'CREATE LOCAL TEMPORARY TABLE #FFT_SIM_DATA_TBL ("ID" INTEGER, "TIMESTAMP" TIMESTAMP, "REAL_VAL" DOUBLE, "IMAG_VAL" DOUBLE)'
    }

    def setUp(self):
        super(TestFFTTools, self).setUp()
        self._createTable("#FFT_SIM_DATA_TBL")
        np.random.seed(23)
        val = np.random.rand(32, 2)
        tp_range = pd.date_range(start='2025-01-01', periods=32)
        data_list = [(i, tp_range[i] , val[i,0], val[i,1]) for i in range(32)]
        self._insertData('#FFT_SIM_DATA_TBL', data_list)

    def tearDown(self):
        self._dropTableIgnoreError("#FFT_SIM_DATA_TBL")
        self._dropTableIgnoreError("FFT_SIM_DATA_TBL_FFT_RESULT")
        self._dropTableIgnoreError("FFT_SIM_DATA_TBL_FFT_RESULT_FFT_RESULT")
        super(TestFFTTools, self).tearDown()

    def test_fft_tools(self):
        tool = FFT(connection_context=self.conn)
        #verify that fft followed by fft-inverse results in identity transform
        tool_input = dict(table_name="#FFT_SIM_DATA_TBL",
                          key="ID", data_cols={'real':'REAL_VAL', 'imag':'IMAG_VAL'})
        result = json.loads(tool.run(tool_input=tool_input))
        self.assertTrue(result['fft_result_table'] == "FFT_SIM_DATA_TBL_FFT_RESULT")
        result_df0 = self.conn.table("FFT_SIM_DATA_TBL_FFT_RESULT").sort("ID").collect()
        tool_input_inv = dict(table_name="FFT_SIM_DATA_TBL_FFT_RESULT",
                              key=self.conn.table("FFT_SIM_DATA_TBL_FFT_RESULT").columns[0],
                              data_cols={'real': 'REAL', 'imag':'IMAG'},
                              inverse=True)
        result_inv = json.loads(tool.run(tool_input=tool_input_inv))
        result_df_inv = self.conn.table(result_inv["fft_result_table"])\
            .rename_columns({"REAL": "REAL_VAL", "IMAG": "IMAG_VAL"}).collect()
        assert_frame_equal(result_df_inv,
                           self.conn.table("#FFT_SIM_DATA_TBL").sort("ID").deselect("TIMESTAMP").collect())
        #verify the timestamp case
        tool_input_tp = dict(table_name="#FFT_SIM_DATA_TBL",
                             key="TIMESTAMP",
                             data_cols={'real':'REAL_VAL', 'imag':'IMAG_VAL'})
        result_tp = json.loads(tool.run(tool_input=tool_input_tp))
        result_df_tp = self.conn.table(result_tp["fft_result_table"]).sort('TIMESTAMP_int').collect()
        assert_frame_equal(result_df0.iloc[:,1:], result_df_tp.iloc[:,1:])
        #verify that fft is linear in essense
        tool_input1 = dict(table_name="#FFT_SIM_DATA_TBL",
                           key="ID", data_cols={'real':'REAL_VAL'})
        tool.run(tool_input=tool_input1)
        result_df1 = self.conn.table("FFT_SIM_DATA_TBL_FFT_RESULT").sort("ID").collect()
        tool_input2 = dict(table_name="#FFT_SIM_DATA_TBL",
                           key="ID", data_cols={'imag':'IMAG_VAL'})
        tool.run(tool_input=tool_input2)
        result_df2 = self.conn.table("FFT_SIM_DATA_TBL_FFT_RESULT").sort("ID").collect()
        result_df1["REAL"] = result_df1["REAL"] + result_df2["REAL"]
        result_df1["IMAG"] = result_df1["IMAG"] + result_df2["IMAG"]
        assert_frame_equal(result_df0, result_df1)

if __name__ == '__main__':
    unittest.main()