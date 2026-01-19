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


if __name__ == "__main__":
    unittest.main()
