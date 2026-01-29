#!/usr/bin/env python3
import unittest
from unittest.mock import patch

from testML_BaseTestClass import TestML_BaseTestClass


class TestHANAAgentTools(TestML_BaseTestClass):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_discovery_agent_tool_schema_and_direct_call(self):
        from hana_ai.tools.hana_ml_tools.graph_tools import DiscoveryAgentTool, HANAAgentToolInput

        tool = DiscoveryAgentTool(connection_context=self.conn)
        # Basic description presence
        self.assertTrue(isinstance(tool.description, str) and len(tool.description.strip()) > 0)

        # Input schema sanity checks
        try:
            schema = HANAAgentToolInput.model_json_schema(by_alias=True)
        except Exception:
            schema = HANAAgentToolInput.schema(by_alias=True)

        props = schema.get("properties", {})
        self.assertIn("query", props)
        self.assertIn("model_name", props)

        # Patch underlying agent run to avoid hitting external services
        with patch("hana_ai.agents.hana_agent.discovery_agent.DiscoveryAgent.run", return_value="DISCOVERY_OK") as mock_run:
            res = tool._run(kwargs={
                "query": "What tables exist?",
                "model_name": "gpt-4.1",
            })
            self.assertEqual(res, "DISCOVERY_OK")
            self.assertTrue(mock_run.called)

    def test_data_agent_tool_schema_and_direct_call(self):
        from hana_ai.tools.hana_ml_tools.graph_tools import DataAgentTool, HANAAgentToolInput

        tool = DataAgentTool(connection_context=self.conn)
        # Basic description presence
        self.assertTrue(isinstance(tool.description, str) and len(tool.description.strip()) > 0)

        # Input schema sanity checks
        try:
            schema = HANAAgentToolInput.model_json_schema(by_alias=True)
        except Exception:
            schema = HANAAgentToolInput.schema(by_alias=True)

        props = schema.get("properties", {})
        self.assertIn("query", props)
        self.assertIn("model_name", props)

        # Patch underlying agent run to avoid hitting external services
        with patch("hana_ai.agents.hana_agent.data_agent.DataAgent.run", return_value="DATA_OK") as mock_run:
            res = tool._run(kwargs={
                "query": "Select top 1 from DUMMY",
                "model_name": "gpt-4.1",
            })
            self.assertEqual(res, "DATA_OK")
            self.assertTrue(mock_run.called)

    def test_discovery_agent_tool_parameter_forwarding(self):
        from hana_ai.tools.hana_ml_tools.graph_tools import DiscoveryAgentTool

        captured = {}

        class FakeAgent:
            def __init__(self, connection_context, **kwargs):
                self.connection_context = connection_context
                self.remote_source_name = kwargs.get("remote_source_name")
                captured["init_kwargs"] = kwargs

            def run(self, query: str, additional_config: dict | None = None, show_progress: bool = True):
                captured["query"] = query
                captured["additional_config"] = additional_config
                captured["remote_source_name_at_run"] = self.remote_source_name
                return "OK"

        with patch("hana_ai.tools.hana_ml_tools.graph_tools.DiscoveryAgent", FakeAgent):
            tool = DiscoveryAgentTool(connection_context=self.conn)
            tool.configure(
                remote_source_name="MY_RS",
                rag_schema_name="MY_RAG_SCHEMA",
                rag_table_name="MY_RAG_TABLE",
                knowledge_graph_name="MY_GRAPH",
            )
            res = tool._run(kwargs={
                "query": "Q",
                "model_name": "my-model",
            })

        self.assertEqual(res, "OK")
        self.assertEqual(captured.get("query"), "Q")
        # additional_config should only carry model info now
        ac = captured.get("additional_config", {})
        self.assertEqual((ac.get("model") or {}).get("name"), "my-model")
        # Ensure constructor received forwarding params
        init_kwargs = captured.get("init_kwargs", {})
        self.assertEqual(init_kwargs.get("remote_source_name"), "MY_RS")
        self.assertEqual(init_kwargs.get("rag_schema_name"), "MY_RAG_SCHEMA")
        self.assertEqual(init_kwargs.get("rag_table_name"), "MY_RAG_TABLE")
        self.assertEqual(init_kwargs.get("knowledge_graph_name"), "MY_GRAPH")
        # Ensure agent holds remote_source_name at run time
        self.assertEqual(captured.get("remote_source_name_at_run"), "MY_RS")

    def test_data_agent_tool_parameter_forwarding(self):
        from hana_ai.tools.hana_ml_tools.graph_tools import DataAgentTool

        captured = {}

        class FakeAgent:
            def __init__(self, connection_context, **kwargs):
                self.connection_context = connection_context
                self.remote_source_name = kwargs.get("remote_source_name")
                captured["init_kwargs"] = kwargs

            def run(self, query: str, additional_config: dict | None = None, show_progress: bool = True):
                captured["query"] = query
                captured["additional_config"] = additional_config
                captured["remote_source_name_at_run"] = self.remote_source_name
                return "OK"

        with patch("hana_ai.tools.hana_ml_tools.graph_tools.DataAgent", FakeAgent):
            tool = DataAgentTool(connection_context=self.conn)
            tool.configure(
                remote_source_name="MY_RS2",
                rag_schema_name="MY_RAG_SCHEMA2",
                rag_table_name="MY_RAG_TABLE2",
                knowledge_graph_name="MY_GRAPH2",
            )
            res = tool._run(kwargs={
                "query": "Q2",
                "model_name": "my-model-2",
            })

        self.assertEqual(res, "OK")
        self.assertEqual(captured.get("query"), "Q2")
        # additional_config should only carry model info now
        ac = captured.get("additional_config", {})
        self.assertEqual((ac.get("model") or {}).get("name"), "my-model-2")
        # Ensure constructor received forwarding params
        init_kwargs = captured.get("init_kwargs", {})
        self.assertEqual(init_kwargs.get("remote_source_name"), "MY_RS2")
        self.assertEqual(init_kwargs.get("rag_schema_name"), "MY_RAG_SCHEMA2")
        self.assertEqual(init_kwargs.get("rag_table_name"), "MY_RAG_TABLE2")
        self.assertEqual(init_kwargs.get("knowledge_graph_name"), "MY_GRAPH2")
        # Ensure agent holds remote_source_name at run time
        self.assertEqual(captured.get("remote_source_name_at_run"), "MY_RS2")


if __name__ == "__main__":
    unittest.main()
