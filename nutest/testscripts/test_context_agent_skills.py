import tempfile
import unittest
from pathlib import Path
import json

from hana_ai.iagents.context_agent import AgentConfig, ContextAgent, _parse_skills_markdown
from hana_ai.tools.hana_ml_tools.dataset_prep_tools import _validate_split_request
from hana_ai.tools.toolkit import HANAMLToolkit


class TestContextAgentSkills(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def _make_agent(self, **config_kwargs):
        config = AgentConfig(skills_use_llm_selector=True, max_active_skills=4, skills_cache_turns=0, **config_kwargs)
        return ContextAgent(
            llm=self._mock_llm_selector,
            tools=[],
            storage_dir=self.temp_dir.name,
            config=config,
        )

    @staticmethod
    def _mock_llm_selector(prompt):
        prompt_text = str(prompt).lower()

        if "user request:" in prompt_text:
            request_text = prompt_text.split("user request:", maxsplit=1)[-1]
        else:
            request_text = prompt_text

        skill_names = []
        if any(token in request_text for token in ("sql comparison", "sql", "comparison table")):
            skill_names.append("hana_dataframe_fallback")
            return json.dumps(skill_names)

        if any(token in request_text for token in ("dataset report", "take a look at", "profile", "report")):
            skill_names.append("timeseries_data_profiling")
        if any(
            token in request_text
            for token in (
                "csv",
                "training set",
                "test set",
                "validation set",
                "training, test, and validation",
                "split",
            )
        ):
            skill_names.append("data_ingestion_and_dataset_preparation")
        if any(token in request_text for token in ("predicted results", "forecast results", "insights from the forecast", "actual values")):
            skill_names.append("prediction_result_analysis")
        if any(token in request_text for token in ("cap artifacts", "artifact", "hdi")):
            skill_names.append("model_lifecycle_and_artifacts")
        if any(token in request_text for token in ("many series", "grouped by", "group comparison")):
            skill_names.append("massive_forecasting")
        if any(token in request_text for token in ("forecasting model", "train the table", "predict model", "forecast")) and not skill_names:
            skill_names.append("timeseries_forecasting")

        return json.dumps(skill_names)

    def test_parse_skills_markdown_extracts_named_blocks(self):
        text = """
## example_skill

Status: test

Goal:
- Do something useful.

When to activate:
- On demand.

## another_skill

Goal:
- Do another thing.
"""
        skills = _parse_skills_markdown(text)

        self.assertEqual(sorted(skills.keys()), ["another_skill", "example_skill"])
        self.assertEqual(skills["example_skill"].title, "Example Skill")
        self.assertEqual(skills["example_skill"].description, "Do something useful.")
        self.assertIn("Do something useful", skills["example_skill"].content)

    def test_parse_skills_markdown_ignores_blocks_without_goal_section(self):
        text = """
## invalid_skill

Status: test only

When to activate:
- Never.

## valid_skill

Goal:
- Do a valid thing.
"""
        skills = _parse_skills_markdown(text)

        self.assertNotIn("invalid_skill", skills)
        self.assertIn("valid_skill", skills)

    def test_default_runtime_loads_skills_markdown(self):
        agent = self._make_agent()

        self.assertIn("data_ingestion_and_dataset_preparation", agent._skills)
        self.assertIn("prediction_result_analysis", agent._skills)
        self.assertIn("timeseries_data_profiling", agent._skills)
        self.assertIn("model_lifecycle_and_artifacts", agent._skills)

    def test_custom_skills_markdown_is_merged_with_defaults(self):
        custom_path = Path(self.temp_dir.name) / "custom_skills.md"
        custom_path.write_text(
            """
## custom_debug_skill

Goal:
- Debug context issues.
""".strip()
            + "\n",
            encoding="utf-8",
        )

        agent = self._make_agent(skills_markdown_path=str(custom_path))

        self.assertIn("custom_debug_skill", agent._skills)
        self.assertIn("prediction_result_analysis", agent._skills)

    def test_invalid_custom_skill_block_does_not_pollute_runtime_catalog(self):
        custom_path = Path(self.temp_dir.name) / "custom_skills.md"
        custom_path.write_text(
            """
## invalid_custom_skill

When to activate:
- This block is invalid because it has no goal.
""".strip()
            + "\n",
            encoding="utf-8",
        )

        agent = self._make_agent(skills_markdown_path=str(custom_path))

        self.assertNotIn("invalid_custom_skill", agent._skills)
        self.assertIn("prediction_result_analysis", agent._skills)

    def test_fallback_selector_routes_profile_prompt(self):
        agent = self._make_agent()

        selected = agent._active_skill_names("Then create a dataset report for me on SALES_REFUNDS table")

        self.assertEqual(selected, ["timeseries_data_profiling"])

    def test_fallback_selector_routes_data_preparation_prompt(self):
        agent = self._make_agent()

        selected = agent._active_skill_names(
            "Import this csv and split it into training set, test set, and validation set for me"
        )

        self.assertEqual(selected, ["data_ingestion_and_dataset_preparation"])

    def test_fallback_selector_routes_prediction_insight_prompt(self):
        agent = self._make_agent()

        selected = agent._active_skill_names("Give some insights from the predicted results.")

        self.assertEqual(selected, ["prediction_result_analysis"])

    def test_fallback_selector_routes_artifact_prompt(self):
        agent = self._make_agent()

        selected = agent._active_skill_names(
            "I want to generate CAP artifacts, the project name is my_project and output path is cap"
        )

        self.assertEqual(selected, ["model_lifecycle_and_artifacts"])

    def test_fallback_selector_routes_massive_forecast_prompt_without_single_series_skill(self):
        agent = self._make_agent()

        selected = agent._active_skill_names("Forecast many series grouped by store_id for the next 30 periods")

        self.assertEqual(selected, ["massive_forecasting"])

    def test_fallback_selector_routes_sql_comparison_prompt(self):
        agent = self._make_agent()

        selected = agent._active_skill_names("Create a SQL comparison table for predicted versus actual values")

        self.assertEqual(selected, ["hana_dataframe_fallback"])

    def test_llm_selector_falls_back_when_model_returns_no_skills(self):
        agent = ContextAgent(
            llm=lambda prompt: "[]",
            tools=[],
            storage_dir=self.temp_dir.name,
            config=AgentConfig(skills_use_llm_selector=True, max_active_skills=4, skills_cache_turns=0),
        )

        selected = agent._active_skill_names(
            "I want to do forecasting on the SALES_REFUNDS table. Please split the data respecting time order."
        )

        self.assertEqual(selected, ["data_ingestion_and_dataset_preparation"])

    def test_llm_selector_merges_high_confidence_fallback_skills(self):
        agent = ContextAgent(
            llm=lambda prompt: json.dumps(["timeseries_forecasting"]),
            tools=[],
            storage_dir=self.temp_dir.name,
            config=AgentConfig(skills_use_llm_selector=True, max_active_skills=4, skills_cache_turns=0),
        )

        selected = agent._active_skill_names(
            "I want to do forecasting on the SALES_REFUNDS table. Please split the data respecting time order."
        )

        self.assertIn("timeseries_forecasting", selected)
        self.assertIn("data_ingestion_and_dataset_preparation", selected)

    def test_toolkit_registers_dataset_preparation_tools(self):
        toolkit_source = Path(HANAMLToolkit.__module__.replace(".", "/") + ".py")
        if not toolkit_source.exists():
            toolkit_source = Path("src/hana_ai/tools/toolkit.py")

        text = toolkit_source.read_text(encoding="utf-8")

        self.assertIn("ImportCSVToTableTool", text)
        self.assertIn("SplitTableForForecastingTool", text)

    def test_split_validation_rejects_non_time_ordered_split(self):
        error = _validate_split_request(
            split_mode="random",
            order_by=None,
        )

        self.assertIn("only split_mode=time_ordered is currently supported", error)

    def test_split_validation_requires_order_by_for_time_ordered(self):
        error = _validate_split_request(
            split_mode="time_ordered",
            order_by=None,
        )

        self.assertEqual(error, "Error: order_by is required for time_ordered splits")

    def test_tool_guidance_mentions_chronology_for_forecasting_splits(self):
        source_path = Path("src/hana_ai/iagents/context_agent.py")
        text = source_path.read_text(encoding="utf-8")

        self.assertIn("split_table_for_forecasting", text)
        self.assertIn("preserve chronology", text)


if __name__ == "__main__":
    unittest.main()