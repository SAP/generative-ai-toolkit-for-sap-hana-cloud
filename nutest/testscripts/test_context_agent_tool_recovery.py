import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import patch

from hana_ai.iagents.context_agent import AgentConfig, ContextAgent


class _ExecutorRaises:
    def __init__(self, err):
        self.err = err

    def invoke(self, _payload):
        raise Exception(self.err)


class _ExecutorReturns:
    def __init__(self, result):
        self.result = result

    def invoke(self, _payload):
        return self.result


class TestContextAgentToolRecovery(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def _make_agent(self):
        return ContextAgent(
            llm=lambda prompt: "ok",
            tools=[],
            storage_dir=self.temp_dir.name,
            config=AgentConfig(skills_use_llm_selector=True, max_active_skills=4, skills_cache_turns=0),
        )

    def test_executor_error_explains_predict_input_shape(self):
        agent = self._make_agent()
        agent.tools = [SimpleNamespace(name="automatic_timeseries_load_model_and_predict")]
        agent._executor = _ExecutorRaises(
            "(423, 'AFL error: exception 73001007: Invalid table:$TAB$.; feature number of predict table does not match the trained model.')"
        )

        response = agent.chat(
            "Predict the grouped holdout table with group key STORE_ID, key BOOKING_DATE and endog REFUNDS."
        )

        self.assertIn("predict input shape does not match the trained model", response)
        self.assertIn("Do not include the label/endog column", response)
        self.assertIn("group key", response.lower())

    def test_executor_error_mentions_backend_issue_for_massive_pal_failure(self):
        agent = self._make_agent()
        agent.tools = [SimpleNamespace(name="massive_automatic_timeseries_load_model_and_predict")]
        agent._executor = _ExecutorRaises(
            "AFL error: AFL DESCRIBE for nested call failed - invalid table(s) for ANY-procedure call"
        )

        response = agent.chat(
            "Forecast many series with group key STORE_ID and predict the grouped holdout table."
        )

        self.assertIn("backend HANA PAL / hana_ml runtime issue", response)
        self.assertIn("group_key + key", response)

    def test_chat_surfaces_auto_repair_diagnostic_from_tool_output(self):
        agent = self._make_agent()
        agent.tools = [SimpleNamespace(name="automatic_timeseries_load_model_and_predict")]
        action = SimpleNamespace(tool="automatic_timeseries_load_model_and_predict", tool_input={"predict_table": "SALES_REFUNDS_TEST"})
        observation = (
            '{"predicted_results_table": "PREDICT_RESULT_SALES_REFUNDS_TEST_MODEL_1", '
            '"auto_repaired_predict_input": true, '
            '"predict_table_columns_before_repair": ["BOOKING_DATE", "REFUNDS"], '
            '"predict_table_columns_used_for_prediction": ["BOOKING_DATE"]}'
        )
        agent._executor = _ExecutorReturns({
            "output": "Prediction completed successfully.",
            "intermediate_steps": [(action, observation)],
        })

        response = agent.chat("Predict using the test table that still contains REFUNDS.")

        self.assertIn("Prediction completed successfully.", response)
        self.assertIn("auto-corrected the prediction input", response)
        self.assertIn("Used for prediction", response)

    def test_agent_tracks_latest_predicted_results_table_in_working_set(self):
        agent = self._make_agent()
        agent.tools = [SimpleNamespace(name="automatic_timeseries_load_model_and_predict")]
        action = SimpleNamespace(tool="automatic_timeseries_load_model_and_predict", tool_input={"predict_table": "SALES_REFUNDS_PRED"})
        observation = '{"predicted_results_table": "PREDICT_RESULT_SALES_REFUNDS_TEST_MODEL_1"}'
        agent._executor = _ExecutorReturns({
            "output": "Prediction completed successfully.",
            "intermediate_steps": [(action, observation)],
        })

        agent.chat("Predict using the holdout table.")

        self.assertEqual(agent._last_predicted_results_table, "PREDICT_RESULT_SALES_REFUNDS_TEST_MODEL_1")
        pack = agent._build_context("Generate the line plot.")
        self.assertIn("Latest predicted_results_table", pack.working_set)
        self.assertIn("PREDICT_RESULT_SALES_REFUNDS_TEST_MODEL_1", pack.working_set)

    def test_chat_recognizes_html_artifact_and_attempts_render(self):
        agent = self._make_agent()
        agent.tools = [SimpleNamespace(name="forecast_line_plot")]
        html_path = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        self.addCleanup(lambda: os.unlink(html_path.name) if os.path.exists(html_path.name) else None)
        with open(html_path.name, "w", encoding="utf-8") as handle:
            handle.write("<html><body><h1>plot</h1></body></html>")

        action = SimpleNamespace(tool="forecast_line_plot", tool_input={"predict_result": "PREDICT_RESULT"})
        observation = '{"html_file": "%s"}' % html_path.name.replace('\\', '\\\\')
        agent._executor = _ExecutorReturns({
            "output": "Plot created.",
            "intermediate_steps": [(action, observation)],
        })

        with patch.object(agent, "_display_html_artifact", return_value=True) as display_mock:
            response = agent.chat("Plot the forecast result.")

        display_mock.assert_called_once()
        self.assertIn("Rendered HTML artifact", response)
        self.assertIn("Plot created.", response)

    def test_display_html_artifact_uses_ipython_display_when_available(self):
        agent = self._make_agent()
        html_file = os.path.join(self.temp_dir.name, "plot.html")
        with open(html_file, "w", encoding="utf-8") as handle:
            handle.write("<html><body><p>render me</p></body></html>")

        rendered = []
        fake_ipython = ModuleType("IPython")
        fake_display_module = ModuleType("IPython.display")

        class FakeHTML:
            def __init__(self, data):
                self.data = data

        def fake_display(obj):
            rendered.append(obj)

        fake_display_module.HTML = FakeHTML
        fake_display_module.display = fake_display
        fake_ipython.display = fake_display_module

        with patch.dict(sys.modules, {"IPython": fake_ipython, "IPython.display": fake_display_module}):
            did_render = agent._display_html_artifact(Path(html_file))

        self.assertTrue(did_render)
        self.assertEqual(len(rendered), 1)
        self.assertIn("render me", rendered[0].data)


if __name__ == "__main__":
    unittest.main()