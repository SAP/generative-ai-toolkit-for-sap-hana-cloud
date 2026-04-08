import json

from hana_ai.tools.hana_ml_tools.additive_model_forecast_tools import MassiveAdditiveModelForecastFitAndSave
from hana_ai.tools.hana_ml_tools.additive_model_forecast_tools import MassiveAdditiveModelForecastLoadModelAndPredict
from testML_BaseTestClass import TestML_BaseTestClass


class TestMassiveAdditiveModelForecastToolsLive(TestML_BaseTestClass):
    tableDef = {
        '#HANAI_MASSIVE_AMF_FIT_RAW': (
            'CREATE LOCAL TEMPORARY TABLE #HANAI_MASSIVE_AMF_FIT_RAW '
            '("TIMESTAMP" TIMESTAMP, "GROUP_ID" INT, "VALUE" DOUBLE)'
        ),
        '#HANAI_MASSIVE_AMF_SCORE_RAW': (
            'CREATE LOCAL TEMPORARY TABLE #HANAI_MASSIVE_AMF_SCORE_RAW '
            '("TIMESTAMP" TIMESTAMP, "GROUP_ID" INT, "VALUE" DOUBLE)'
        ),
    }

    def setUp(self):
        super(TestMassiveAdditiveModelForecastToolsLive, self).setUp()
        self._createTable('#HANAI_MASSIVE_AMF_FIT_RAW')
        self._createTable('#HANAI_MASSIVE_AMF_SCORE_RAW')
        data_list_raw = [
            ('1900-01-01 12:00:00', 0, 998.23063348829),
            ('1900-01-01 13:00:00', 0, 997.984413594973),
            ('1900-01-01 14:00:00', 0, 998.076511123945),
            ('1900-01-01 15:00:00', 0, 997.9165407258),
            ('1900-01-01 16:00:00', 0, 997.438758925335),
            ('1900-01-01 12:00:00', 1, 1098.23063348829),
            ('1900-01-01 13:00:00', 1, 1097.984413594973),
            ('1900-01-01 14:00:00', 1, 1098.076511123945),
            ('1900-01-01 15:00:00', 1, 1097.9165407258),
            ('1900-01-01 16:00:00', 1, 1097.438758925335),
        ]
        data_list_score_raw = [
            ('1900-01-01 17:00:00', 0, 0),
            ('1900-01-01 18:00:00', 0, 0),
            ('1900-01-01 17:00:00', 1, 0),
            ('1900-01-01 18:00:00', 1, 0),
        ]
        self._insertData('#HANAI_MASSIVE_AMF_FIT_RAW', data_list_raw)
        self._insertData('#HANAI_MASSIVE_AMF_SCORE_RAW', data_list_score_raw)
        self.conn.table('#HANAI_MASSIVE_AMF_SCORE_RAW').drop('VALUE').save('#HANAI_MASSIVE_AMF_PREDICT_RAW')

    def tearDown(self):
        self._dropTableIgnoreError('#HANAI_MASSIVE_AMF_FIT_RAW')
        self._dropTableIgnoreError('#HANAI_MASSIVE_AMF_PREDICT_RAW')
        self._dropTableIgnoreError('#HANAI_MASSIVE_AMF_SCORE_RAW')
        self._dropTableIgnoreError('PREDICT_RESULT_#HANAI_MASSIVE_AMF_PREDICT_RAW_MASSIVE_AMF_MODEL_1')
        self._dropTableIgnoreError('PREDICT_ERROR_#HANAI_MASSIVE_AMF_PREDICT_RAW_MASSIVE_AMF_MODEL_1')
        self._dropTableIgnoreError('REASON_CODE_#HANAI_MASSIVE_AMF_PREDICT_RAW_MASSIVE_AMF_MODEL_1')
        super(TestMassiveAdditiveModelForecastToolsLive, self).tearDown()

    def test_fit_and_save(self):
        tool = MassiveAdditiveModelForecastFitAndSave(connection_context=self.conn)
        result = json.loads(tool.run({
            'fit_table': '#HANAI_MASSIVE_AMF_FIT_RAW',
            'key': 'TIMESTAMP',
            'group_key': 'GROUP_ID',
            'endog': 'VALUE',
            'name': 'MASSIVE_AMF_MODEL',
            'version': 1,
        }))
        self.assertTrue(result['trained_table'] == '#HANAI_MASSIVE_AMF_FIT_RAW')
        self.assertTrue(result['model_storage_name'] == 'MASSIVE_AMF_MODEL')
        self.assertTrue(int(result['model_storage_version']) == 1)

    def test_load_model_and_predict(self):
        tool = MassiveAdditiveModelForecastLoadModelAndPredict(connection_context=self.conn)
        result = json.loads(tool.run({
            'predict_table': '#HANAI_MASSIVE_AMF_PREDICT_RAW',
            'key': 'TIMESTAMP',
            'group_key': 'GROUP_ID',
            'name': 'MASSIVE_AMF_MODEL',
            'version': 1,
        }))
        self.assertTrue(
            result['predicted_results_table'] == 'PREDICT_RESULT_#HANAI_MASSIVE_AMF_PREDICT_RAW_MASSIVE_AMF_MODEL_1'
        )
        self.assertTrue(
            result['prediction_error_table'] == 'PREDICT_ERROR_#HANAI_MASSIVE_AMF_PREDICT_RAW_MASSIVE_AMF_MODEL_1'
        )