import json
from testML_BaseTestClass import TestML_BaseTestClass
from hana_ai.tools.hana_ml_tools.ts_check_tools import MassiveTimeSeriesCheck

class TestMassiveTSCheckTools(TestML_BaseTestClass):
    tableDef = {
        '#HANAI_DATA_TBL_MASSIVE':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_MASSIVE ("GROUP_ID" VARCHAR(10), "TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)'
    }
    
    def setUp(self):
        super(TestMassiveTSCheckTools, self).setUp()
        self._createTable('#HANAI_DATA_TBL_MASSIVE')
        
        # 创建包含两个分组的数据
        data_list = [
            # Group A
            ('A', '1900-01-01 12:00:00', 998.23),
            ('A', '1900-01-01 13:00:00', 997.98),
            ('A', '1900-01-01 14:00:00', 998.08),
            ('A', '1900-01-01 15:00:00', 997.92),
            ('A', '1900-01-01 16:00:00', 997.44),
            
            # Group B
            ('B', '1900-01-01 12:00:00', 100.0),
            ('B', '1900-01-01 13:00:00', 101.5),
            ('B', '1900-01-01 14:00:00', 102.2),
            ('B', '1900-01-01 15:00:00', 103.1),
            ('B', '1900-01-01 16:00:00', 104.0),
        ]
        self._insertData('#HANAI_DATA_TBL_MASSIVE', data_list)

    def tearDown(self):
        self._dropTableIgnoreError('#HANAI_DATA_TBL_MASSIVE')
        super(TestMassiveTSCheckTools, self).tearDown()

    def test_MassiveTimeSeriesCheck_Basic(self):
        """测试基本功能 - 两个分组"""
        tool = MassiveTimeSeriesCheck(connection_context=self.conn)
        
        # 使用字典作为tool_input
        result = tool.run({
            "table_name": "#HANAI_DATA_TBL_MASSIVE",
            "group_key": "GROUP_ID",
            "key": "TIMESTAMP",
            "endog": "VALUE"
        })
        
        # 验证基本结构
        self.assertIn("Time Series Analysis Report (2 groups)", result)
        self.assertIn("Group 1/2: GROUP_ID = ", result)
        self.assertIn("Group 2/2: GROUP_ID = ", result)
        
        # 验证分组A的内容
        self.assertIn("Key: TIMESTAMP", result)
        self.assertIn("Endog: VALUE", result)
        self.assertIn("Index: starts from 1900-01-01 12:00:00 to 1900-01-01 16:00:00", result)
        self.assertIn("Intermittent Test: proportion of zero values is 0.0", result)
        
        # 验证分组B的内容
        self.assertIn("Trend Test:", result)
        self.assertIn("Seasonality Test:", result)
        self.assertIn("Available algorithms:", result)

    def test_MassiveTimeSeriesCheck_MissingParams(self):
        """测试缺失参数的情况"""
        tool = MassiveTimeSeriesCheck(connection_context=self.conn)
        
        # 缺少group_key
        try:
            result = tool.run({
            "table_name": "#HANAI_DATA_TBL_MASSIVE",
            "key": "TIMESTAMP",
            "endog": "VALUE"
            })
        except Exception as e:
            self.assertIn("group_key", str(e))
            self.assertIn("1 validation error for MassiveTSCheckInput", str(e))
        
        # 缺少key
        try:
            result = tool.run({
                "table_name": "#HANAI_DATA_TBL_MASSIVE",
                "group_key": "GROUP_ID",
                "endog": "VALUE"
            })
        except Exception as e:
            self.assertIn("key", str(e))
            self.assertIn("1 validation error for MassiveTSCheckInput", str(e))
        
        # 缺少endog
        try:
            result = tool.run({
                "table_name": "#HANAI_DATA_TBL_MASSIVE",
                "group_key": "GROUP_ID",
                "key": "TIMESTAMP"
            })
        except Exception as e:
            self.assertIn("endog", str(e))
            self.assertIn("1 validation error for MassiveTSCheckInput", str(e))

    def test_MassiveTimeSeriesCheck_TooManyGroups(self):
        """测试分组过多的情况"""
        # 创建包含101个分组的测试数据
        data_list = []
        for i in range(101):
            group_id = f"G{i:03d}"
            for j in range(5):
                timestamp = f"1900-01-01 {12+j}:00:00"
                value = 100.0 + i + j
                data_list.append((group_id, timestamp, value))
        
        # 清空并重新插入数据
        self.conn.sql(f"TRUNCATE TABLE #HANAI_DATA_TBL_MASSIVE")
        self._insertData('#HANAI_DATA_TBL_MASSIVE', data_list)
        
        tool = MassiveTimeSeriesCheck(connection_context=self.conn)
        result = tool.run({
            "table_name": "#HANAI_DATA_TBL_MASSIVE",
            "group_key": "GROUP_ID",
            "key": "TIMESTAMP",
            "endog": "VALUE"
        })
        
        self.assertIn("Too many groups (103) for analysis", result)
