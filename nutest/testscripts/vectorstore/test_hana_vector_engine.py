import unittest

from hana_ml.dataframe import ConnectionContext
from generative_ai_toolkit_for_sap_hana_cloud.vectorstore.hana_vector_engine import HANAMLinVectorEngine
from generative_ai_toolkit_for_sap_hana_cloud.vectorstore.code_templates import get_code_templates
"create unittest for HANAMLinVectorEngine"
class TestHANAMLinVectorEngine(unittest.TestCase):
    def test_HANAMLinVectorEngine(self):
        url, port, user = "hana-ml-api.hana-ml.c.ap-cn-1.cloud.sap", 30115, "PAL_TEST"
        cc = ConnectionContext(url, port, user)
        cc.drop_table("test_table")
        result = HANAMLinVectorEngine(cc, "test_table")
        self.assertEqual(result.table_name, "test_table")
        result.upsert_knowledge(get_code_templates())
        # test query
        query_input = "AutomaticRegression"
        query_result = result.query(query_input)
        cc.drop_table("test_table")
        self.assertTrue('AutomaticRegression' in str(query_result))
