#unittest for CAP Artifacts tools
import os
import shutil
from hana_ai.tools.hana_ml_tools.automatic_timeseries_tools import AutomaticTimeSeriesFitAndSave, AutomaticTimeseriesLoadModelAndPredict, AutomaticTimeseriesLoadModelAndScore
from hana_ai.tools.hana_ml_tools.cap_artifacts_tools import CAPArtifactsTool
from testML_BaseTestClass import TestML_BaseTestClass
from hana_ml.model_storage import ModelStorage

class TestCAPArtifactsTool(TestML_BaseTestClass):
    tableDef = {
        '#HANAI_DATA_TBL_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_RAW ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)',
        '#HANAI_DATA_TBL_SCORE_RAW':
            'CREATE LOCAL TEMPORARY TABLE #HANAI_DATA_TBL_SCORE_RAW ("TIMESTAMP" TIMESTAMP, "VALUE" DOUBLE)',
    }
    def setUp(self):
        super(TestCAPArtifactsTool, self).setUp()
        self._createTable('#HANAI_DATA_TBL_RAW')
        self._createTable('#HANAI_DATA_TBL_SCORE_RAW')
        data_list_raw = [
            ('1900-01-01 12:00:00', 998.23063348829),
            ('1900-01-01 13:00:00', 997.984413594973),
            ('1900-01-01 14:00:00', 998.076511123945),
            ('1900-01-01 15:00:00', 997.9165407258),
            ('1900-01-01 16:00:00', 997.438758925335),
            ]
        data_list_predict_raw = [
            ('1900-01-01 17:00:00', 0),
            ('1900-01-01 18:00:00', 0),
            ]
        self._insertData('#HANAI_DATA_TBL_RAW', data_list_raw)
        self._insertData('#HANAI_DATA_TBL_SCORE_RAW', data_list_predict_raw)
        self.conn.table("#HANAI_DATA_TBL_SCORE_RAW").drop("VALUE").save("#HANAI_DATA_TBL_PREDICT_RAW")

    def tearDown(self):
        shutil.rmtree('./CAP_OUTPUT_DIR', ignore_errors=True)
        ms = ModelStorage(self.conn)
        ms.delete_model("AUTOML_MODEL", 1)
        self._dropTableIgnoreError('#HANAI_DATA_TBL_RAW')
        self._dropTableIgnoreError('#HANAI_DATA_TBL_PREDICT_RAW')
        self._dropTableIgnoreError('#HANAI_DATA_TBL_SCORE_RAW')
        super(TestCAPArtifactsTool, self).tearDown()

    def test_CAPArtifactsTool(self):
        tool = AutomaticTimeSeriesFitAndSave(connection_context=self.conn)
        tool.run({"fit_table": "#HANAI_DATA_TBL_RAW", "key": "TIMESTAMP", "endog": "VALUE", "name": "AUTOML_MODEL", "version": 1, 'fold_num': 2})
        tool = CAPArtifactsTool(connection_context=self.conn)
        result =tool.run({"name": "AUTOML_MODEL", "version": 1, "project_name": "CAP_PROJECT", "output_dir": "CAP_OUTPUT_DIR"})
        #check if the output directory is created
        self.assertTrue(os.path.exists(os.path.join('.', 'CAP_OUTPUT_DIR')))
        #check if the CDS files are created
        self.assertTrue(os.path.exists(os.path.join('.', 'CAP_OUTPUT_DIR', 'CAP_PROJECT', 'srv')))
        self.assertTrue(os.path.exists(os.path.join('.', 'CAP_OUTPUT_DIR', 'CAP_PROJECT', 'db', 'src', 'hana-ml-base-pal-automl-fit.hdbprocedure')))
        tool = AutomaticTimeseriesLoadModelAndPredict(connection_context=self.conn)
        tool.run({"predict_table": "#HANAI_DATA_TBL_PREDICT_RAW", "key": "TIMESTAMP", "name": "AUTOML_MODEL", "version": 1})
        tool = AutomaticTimeseriesLoadModelAndScore(connection_context=self.conn)
        tool.run({"score_table": "#HANAI_DATA_TBL_SCORE_RAW", "key": "TIMESTAMP", "endog": "VALUE", "name": "AUTOML_MODEL", "version": 1})
        tool = CAPArtifactsTool(connection_context=self.conn)
        result =tool.run({"name": "AUTOML_MODEL", "version": 1, "project_name": "CAP_PROJECT", "output_dir": "CAP_OUTPUT_DIR", "cds_gen": True, "tudf": True})
        #check if the output directory is created
        self.assertTrue(os.path.exists(os.path.join('.', 'CAP_OUTPUT_DIR', 'CAP_PROJECT', 'db', 'src', 'hana-ml-base-pal-pipeline-predict.hdbprocedure')))
        self.assertTrue(os.path.exists(os.path.join('.', 'CAP_OUTPUT_DIR', 'CAP_PROJECT', 'db', 'src', 'hana-ml-base-pal-pipeline-score.hdbprocedure')))
