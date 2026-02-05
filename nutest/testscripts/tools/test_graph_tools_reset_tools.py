import unittest
from testML_BaseTestClass import TestML_BaseTestClass

from hana_ai.tools.toolkit import HANAMLToolkit
from hana_ai.tools.hana_ml_tools.graph_tools import DiscoveryAgentTool, DataAgentTool


class TestGraphToolsReset(TestML_BaseTestClass):
    def test_reset_to_graph_tools_instances(self):
        toolkit = HANAMLToolkit(connection_context=self.conn)
        # Reset to only DiscoveryAgentTool and DataAgentTool using instances
        discovery_tool = DiscoveryAgentTool(connection_context=self.conn)
        data_tool = DataAgentTool(connection_context=self.conn)
        toolkit.reset_tools([discovery_tool, data_tool])

        tools = toolkit.get_tools()
        self.assertEqual(len(tools), 2)
        names = sorted([t.name for t in tools])
        self.assertEqual(names, ["data_agent", "discovery_agent"])
        self.assertTrue(isinstance(tools[0], (DiscoveryAgentTool, DataAgentTool)))
        self.assertTrue(isinstance(tools[1], (DiscoveryAgentTool, DataAgentTool)))

    def test_reset_to_default_when_none(self):
        toolkit = HANAMLToolkit(connection_context=self.conn)
        # First set custom tools, then reset to default
        toolkit.reset_tools([DiscoveryAgentTool(connection_context=self.conn)])
        self.assertEqual(len(toolkit.get_tools()), 1)

        toolkit.reset_tools(None)
        default_tools = toolkit.get_tools()
        self.assertGreater(len(default_tools), 1)
        # Ensure default contains some known tool (e.g., ts_check)
        default_names = [t.name for t in default_tools]
        self.assertIn("ts_check", default_names)

    def test_configure_discovery_agent_custom_schema_and_procedure(self):
        discovery_tool = DiscoveryAgentTool(connection_context=self.conn)
        discovery_tool.configure(
            remote_source_name="RS_AICORE",
            rag_schema_name="MY_RAG_SCHEMA",
            rag_table_name="MY_RAG_TABLE",
            knowledge_graph_name="MY_KG",
            schema_name="MY_SCHEMA",
            procedure_name="MY_PROC"
        )

        self.assertEqual(discovery_tool.remote_source_name, "RS_AICORE")
        self.assertEqual(discovery_tool.rag_schema_name, "MY_RAG_SCHEMA")
        self.assertEqual(discovery_tool.rag_table_name, "MY_RAG_TABLE")
        self.assertEqual(discovery_tool.knowledge_graph_name, "MY_KG")
        self.assertEqual(discovery_tool.schema_name, "MY_SCHEMA")
        self.assertEqual(discovery_tool.procedure_name, "MY_PROC")

    def test_configure_data_agent_custom_schema_and_procedure(self):
        data_tool = DataAgentTool(connection_context=self.conn)
        data_tool.configure(
            remote_source_name="RS_AICORE",
            rag_schema_name="MY_RAG_SCHEMA",
            rag_table_name="MY_RAG_TABLE",
            knowledge_graph_name="MY_KG",
            schema_name="MY_SCHEMA",
            procedure_name="MY_PROC"
        )

        self.assertEqual(data_tool.remote_source_name, "RS_AICORE")
        self.assertEqual(data_tool.rag_schema_name, "MY_RAG_SCHEMA")
        self.assertEqual(data_tool.rag_table_name, "MY_RAG_TABLE")
        self.assertEqual(data_tool.knowledge_graph_name, "MY_KG")
        self.assertEqual(data_tool.schema_name, "MY_SCHEMA")
        self.assertEqual(data_tool.procedure_name, "MY_PROC")


if __name__ == "__main__":
    unittest.main()
