import unittest

from hana_ml.dataframe import ConnectionContext
from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine
from hana_ai.vectorstore.code_templates import get_code_templates
from nutest.testscripts.testML_BaseTestClass import TestML_BaseTestClass
"create unittest for HANAMLinVectorEngine"
class TestHANAMLinVectorEngine(TestML_BaseTestClass):

    def setUp(self):
        super(TestHANAMLinVectorEngine, self).setUp()

    def tearDown(self):
        super(TestHANAMLinVectorEngine, self).tearDown()

    def test_HANAMLinVectorEngine(self):
        cc = self.conn
        result = HANAMLinVectorEngine(cc, "test_table")
        self.assertEqual(result.table_name, "test_table")
        result.upsert_knowledge(get_code_templates())
        # test query
        query_input = "AutomaticRegression"
        query_result = result.query(query_input)
        cc.drop_table("test_table")
        self.assertTrue('AutomaticRegression' in str(query_result))
