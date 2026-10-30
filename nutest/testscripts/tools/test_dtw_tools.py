import unittest
import json
import numpy as np
from hana_ml.dataframe import create_dataframe_from_pandas
from hana_ai.tools.hana_ml_tools.dtw_tools import DTW
from testML_BaseTestClass import TestML_BaseTestClass

class TestDTWTools(TestML_BaseTestClass):
    tableDef = {
        '#DTW_SIM_QUERY_DATA_TBL':
            'CREATE LOCAL TEMPORARY COLUMN TABLE #DTW_SIM_QUERY_DATA_TBL("QID" VARCHAR(20), "TS_ORDER" INTEGER, "TS_VAL" DOUBLE)',
        '#DTW_SIM_REF_DATA_TBL':
            'CREATE LOCAL TEMPORARY COLUMN TABLE #DTW_SIM_REF_DATA_TBL("RID" VARCHAR(20), "TS_ORDER" INTEGER, "TS_VAL" DOUBLE)'
    }

    def setUp(self):
        super(TestDTWTools, self).setUp()
        self._createTable("#DTW_SIM_QUERY_DATA_TBL")
        data_list = [('A', int(i) , float(i) / 10) for i in range(11)]
        self._insertData('#DTW_SIM_QUERY_DATA_TBL', data_list)
        self._createTable("#DTW_SIM_REF_DATA_TBL")
        data_list = [('B', int(i) , np.log2(1 + float(i) / 16)) for i in range(17)]
        self._insertData('#DTW_SIM_REF_DATA_TBL', data_list)

    def tearDown(self):
        self._dropTableIgnoreError("#DTW_SIM_QUERY_DATA_TBL")
        self._dropTableIgnoreError("#DTW_SIM_REF_DATA_TBL")
        super(TestDTWTools, self).tearDown()

    def test_dtw_tools(self):
        tool = DTW(connection_context=self.conn)
        tool_input = dict(query_table="#DTW_SIM_QUERY_DATA_TBL",
                          ref_table="#DTW_SIM_REF_DATA_TBL",
                          query_ts_id="QID",
                          query_ts_order="TS_ORDER",
                          ref_ts_id="RID",
                          ref_ts_order="TS_ORDER",
                          radius=8,
                          distance_method="manhattan",
                          alignment_method="closed",
                          step_pattern=1,
                          save_alignment=True)
        result = json.loads(tool.run(tool_input=tool_input))
        dtw_res = result["DTW_results_in_tuple(QUERY_QID, REF_RID, DISTANCE, WEIGHT, AVG_DISTANCE)"]
        self.assertTrue(dtw_res == "[('A', 'B', 0.39079479746211876, 17.0, 0.022987929262477575)]")
        self.assertTrue(result["dtw_alignment_table"] == "DTW_SIM_QUERY_DATA_TBL_DTW_SIM_REF_DATA_TBL_DTW_ALIGNMENT")
        tool_input2 = dict(query_table="#DTW_SIM_QUERY_DATA_TBL",
                           ref_table="#DTW_SIM_REF_DATA_TBL",
                           query_ts_id="QID",
                           query_ts_order="TS_ORDER",
                           ref_ts_id="RID",
                           ref_ts_order="TS_ORDER",
                           radius=8,
                           distance_method="manhattan",
                           alignment_method="closed",
                           step_pattern=5,
                           save_alignment=True)
        result2 = json.loads(tool.run(tool_input=tool_input2))
        dtw_res2 = result2["DTW_results_in_tuple(QUERY_QID, REF_RID, DISTANCE, WEIGHT, AVG_DISTANCE)"]
        tool_input3 = dict(query_table="#DTW_SIM_QUERY_DATA_TBL",
                           ref_table="#DTW_SIM_REF_DATA_TBL",
                           query_ts_id="QID",
                           query_ts_order="TS_ORDER",
                           ref_ts_id="RID",
                           ref_ts_order="TS_ORDER",
                           radius=8,
                           distance_method="manhattan",
                           alignment_method="closed",
                           step_pattern=[((1,1,1),(1,0,1)), (1,1,1), ((1,1,0.5), (0,1,0.5))],
                           save_alignment=True)
        result3 = json.loads(tool.run(tool_input=tool_input3))
        dtw_res3 = result3["DTW_results_in_tuple(QUERY_QID, REF_RID, DISTANCE, WEIGHT, AVG_DISTANCE)"]
        self.assertTrue(dtw_res2 == dtw_res3)

if __name__ == '__main__':
    unittest.main()