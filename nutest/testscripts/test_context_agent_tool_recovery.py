import tempfile
import unittest
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()