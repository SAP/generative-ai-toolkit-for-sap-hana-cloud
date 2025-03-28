#unittest for AdditiveModelForecastTools
import json
from hana_ai.tools.hana_ml_tools.additive_model_forecast_tools import AdditiveModelForecastFitAndSave
from hana_ai.tools.hana_ml_tools.model_storage_tools import ListModels
from testML_BaseTestClass import TestML_BaseTestClass
class TestModelStorage(TestML_BaseTestClass):
    tableDef = {
        '#HANAI_DATA_TBL_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_RAW ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)',
    }
    def setUp(self):
        super(TestModelStorage, self).setUp()
        self._createTable('#HANAI_DATA_TBL_RAW')
        data_list_raw = [
            ('1900-01-01 12:00:00', 998.23063348829),
            ('1900-01-01 13:00:00', 997.984413594973),
            ('1900-01-01 14:00:00', 998.076511123945),
            ('1900-01-01 15:00:00', 997.9165407258),
            ('1900-01-01 16:00:00', 997.438758925335),
            ]
        self._insertData('#HANAI_DATA_TBL_RAW', data_list_raw)

    def tearDown(self):
        self._dropTableIgnoreError('#HANAI_DATA_TBL_RAW')
        super(TestModelStorage, self).tearDown()

    def test_list_models(self):
        tool = AdditiveModelForecastFitAndSave(connection_context=self.conn)
        result = json.loads(tool.run({"fit_table": "#HANAI_DATA_TBL_RAW", "key": "TIMESTAMP", "endog": "VALUE", "name": "HANAI_MODEL", "version": 1}))
        self.assertTrue(result['trained_table']=="#HANAI_DATA_TBL_RAW")
        self.assertTrue(result['model_storage_name']=="HANAI_MODEL")
        self.assertTrue(int(result['model_storage_version'])==1)

        ms_tool = ListModels(connection_context=self.conn)
        ms_result = ms_tool.run({"name": "HANAI_MODEL", "version": 1})
        #ms_result is a pandas dataframe
        self.assertTrue(ms_result.shape[0] == 1)
        self.assertTrue(ms_result["NAME"][0] == "HANAI_MODEL")
        self.assertTrue(ms_result["VERSION"][0] == 1)
