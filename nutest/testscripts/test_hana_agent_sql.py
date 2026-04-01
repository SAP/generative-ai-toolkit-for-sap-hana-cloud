import importlib.util
import os
import sys
import unittest


HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

UTILITY_PATH = os.path.join(SRC_DIR, "hana_ai", "agents", "hana_agent", "utility.py")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

UTILITY_SPEC = importlib.util.spec_from_file_location("hana_agent_utility", UTILITY_PATH)
UTILITY_MODULE = importlib.util.module_from_spec(UTILITY_SPEC)
assert UTILITY_SPEC.loader is not None
UTILITY_SPEC.loader.exec_module(UTILITY_MODULE)
_call_agent_sql = UTILITY_MODULE._call_agent_sql


class TestHanaAgentSql(unittest.TestCase):
    def test_ai_object_retrieval_sql_matches_expected_signature(self):
        sql = _call_agent_sql(
            remote_source_schema_name=None,
            remote_source_name="HANA_DISCOVERY_AGENT_CREDENTIALS",
            ai_metadata_schema_name="MY_SCHEMA",
            ai_metadata_object_prefix="MY_INDEX_NAME",
            model_and_version=None,
            query="In which position does Mia Nguyen work ?",
            options=None,
            schema_name="SYS",
            procedure_name="AI_OBJECT_RETRIEVAL",
        )

        expected_sql = (
            "DO\n"
            "BEGIN\n"
            "DECLARE output NCLOB;\n"
            "CALL SYS.AI_OBJECT_RETRIEVAL(NULL, 'HANA_DISCOVERY_AGENT_CREDENTIALS', "
            "'MY_SCHEMA', 'MY_INDEX_NAME', NULL, 'In which position does Mia Nguyen work ?', "
            "output, NULL);\n"
            "SELECT :output FROM DUMMY;\n"
            "END"
        )

        self.assertEqual(sql, expected_sql)

    def test_ai_data_retrieval_sql_serializes_options_and_escapes_query(self):
        sql = _call_agent_sql(
            remote_source_schema_name=None,
            remote_source_name="REMOTE_SOURCE",
            ai_metadata_schema_name="AI_SCHEMA",
            ai_metadata_object_prefix="OBJECT_PREFIX",
            model_and_version=None,
            query="what's the revenue?",
            options={"allow_sql": ["SELECT"], "limit": 10},
            schema_name="CUSTOM_SYS",
            procedure_name="AI_DATA_RETRIEVAL",
        )

        expected_sql = (
            "DO\n"
            "BEGIN\n"
            "DECLARE output NCLOB;\n"
            "CALL CUSTOM_SYS.AI_DATA_RETRIEVAL(NULL, 'REMOTE_SOURCE', 'AI_SCHEMA', "
            "'OBJECT_PREFIX', NULL, 'what''s the revenue?', output, "
            "'{\"allow_sql\": [\"SELECT\"], \"limit\": 10}');\n"
            "SELECT :output FROM DUMMY;\n"
            "END"
        )

        self.assertEqual(sql, expected_sql)

    def test_generated_sql_uses_new_procedure_names(self):
        data_sql = _call_agent_sql(
            remote_source_schema_name=None,
            remote_source_name="REMOTE_SOURCE",
            ai_metadata_schema_name="AI_SCHEMA",
            ai_metadata_object_prefix="OBJECT_PREFIX",
            model_and_version=None,
            query="show me data",
            options=None,
            schema_name="SYS",
            procedure_name="AI_DATA_RETRIEVAL",
        )
        object_sql = _call_agent_sql(
            remote_source_schema_name=None,
            remote_source_name="REMOTE_SOURCE",
            ai_metadata_schema_name="AI_SCHEMA",
            ai_metadata_object_prefix="OBJECT_PREFIX",
            model_and_version=None,
            query="describe object",
            options=None,
            schema_name="SYS",
            procedure_name="AI_OBJECT_RETRIEVAL",
        )

        self.assertIn("CALL SYS.AI_DATA_RETRIEVAL(", data_sql)
        self.assertIn("CALL SYS.AI_OBJECT_RETRIEVAL(", object_sql)


if __name__ == "__main__":
    unittest.main()