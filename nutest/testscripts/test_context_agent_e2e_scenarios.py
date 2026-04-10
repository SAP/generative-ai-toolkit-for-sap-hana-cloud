import os
import re
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


class TestContextAgentE2EScenarios(unittest.TestCase):
    """Live end-to-end scenarios for ContextAgent.

    These scenarios are derived from the conversation flow in
    nutest/testscripts/demo/e2e_scenarios/context_agent.ipynb and cover the
    main capabilities currently exercised there:

    - data preview and dataset report generation
    - time-series check, model training, and prediction
    - prediction visualization and insight generation
    - CAP artifact generation from model storage
    - outlier detection on a forecast dataset
        - massive forecasting on grouped time-series data
        - massive outlier detection on grouped time-series data

    This suite is intentionally opt-in because it requires a live HANA
    connection via userkey `RaysKey` and a working LLM setup for
    `gen_ai_hub.proxy.langchain.init_llm`.

        Note:
        - ContextAgent is the only recommended live e2e target here.
        - HANAMLRAGAgent and Mem0HANARAGAgent are deprecated for new e2e scenario
            expansion and are intentionally not extended in this suite.

        Scenario matrix:
        - preview_and_report:
            prompt flow = preview rows -> create dataset report
            assertions = textual preview or markdown table, html report exists
        - forecast_train_predict:
            prompt flow = ts_check -> suggested model training -> prediction
            assertions = model version captured, predicted table exists
        - prediction_analysis:
            prompt flow = train/predict -> line plot -> insight generation
            assertions = plot html exists, insight mentions error or quality signals
        - prediction_analysis_notebook_render:
            prompt flow = train/predict -> line plot in notebook-like frontend
            assertions = html plot exists, notebook display hook receives rendered HTML
        - cap_artifacts:
            prompt flow = train/predict -> generate CAP artifacts
            assertions = artifact root exists and contains package.json
        - outlier_detection:
            prompt flow = detect outliers on a single-series table
            assertions = output mentions outliers or result_select_statement
        - massive_forecasting:
            prompt flow = massive ts_check -> grouped training -> grouped prediction
            assertions = predicted table exists and contains group key + time key
        - massive_additive_forecasting:
            prompt flow = grouped additive forecast request -> grouped additive training -> grouped additive prediction
            assertions = saved model is AdditiveModelForecast in massive mode, predicted table exists, prediction error table exists
        - massive_outlier_detection:
            prompt flow = detect grouped outliers on the massive training table
            assertions = output mentions grouped outliers or result_select_statement
    """

    @classmethod
    def setUpClass(cls):
        if os.environ.get("RUN_CONTEXT_AGENT_E2E") != "1":
            raise unittest.SkipTest("Set RUN_CONTEXT_AGENT_E2E=1 to run live ContextAgent e2e scenarios.")

        import certifi
        import pandas as pd
        from gen_ai_hub.proxy.langchain import init_llm
        from hana_ml import dataframe
        from hana_ml.model_storage import ModelStorage

        from hana_ai.agents.context_agent import ContextAgent
        from hana_ai.iagents.context_agent import AgentConfig
        from hana_ai.tools.toolkit import HANAMLToolkit

        cls.certifi = certifi
        cls.pd = pd
        cls.dataframe = dataframe
        cls.ContextAgent = ContextAgent
        cls.AgentConfig = AgentConfig
        cls.HANAMLToolkit = HANAMLToolkit
        cls.ModelStorage = ModelStorage

        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.scenario_dir = cls.repo_root / "nutest" / "testscripts" / "demo" / "e2e_scenarios"
        cls.csv_path = cls.scenario_dir / "SALES_REFUNDS.csv"
        cls.output_root = Path(tempfile.mkdtemp(prefix="context_agent_e2e_"))
        cls.name_prefix = f"CTXE2E_{uuid.uuid4().hex[:8].upper()}"
        cls.created_models = []
        cls.created_tables = []

        sslstore = certifi.where()
        cls.cc = dataframe.ConnectionContext(
            userkey="RaysKey",
            sslValidateCertificate=False,
            encrypt=True,
            sslKeyStore=sslstore,
        )
        cls.llm = init_llm("gpt-4.1", temperature=0.0, max_tokens=1800)
        cls._prepare_sales_refunds_fixture()
        cls._prepare_massive_sales_refunds_fixture()

    @classmethod
    def tearDownClass(cls):
        for model_name in reversed(cls.created_models):
            try:
                cls.ModelStorage(cls.cc).delete_models(name=model_name)
            except Exception:
                pass

        for table_name in reversed(cls.created_tables):
            try:
                cls.cc.sql(f'DROP TABLE "{table_name}"').collect()
            except Exception:
                pass

        try:
            cls.cc.connection.close()
        except Exception:
            pass

        shutil.rmtree(cls.output_root, ignore_errors=True)

    @classmethod
    def _prepare_sales_refunds_fixture(cls):
        raw_table = f"{cls.name_prefix}_SALES_REFUNDS"
        train_table = f"{cls.name_prefix}_SALES_REFUNDS_TRAIN"
        test_table = f"{cls.name_prefix}_SALES_REFUNDS_TEST"
        predict_table = f"{cls.name_prefix}_SALES_REFUNDS_PREDICT"

        pdf = cls.pd.read_csv(cls.csv_path)
        hana_df = cls.dataframe.create_dataframe_from_pandas(
            cls.cc,
            pandas_df=pdf,
            table_name=raw_table,
            force=True,
        ).to_datetime({"BOOKING_DATE": "YYYY-MM-DD"}).sort_values(by="BOOKING_DATE")
        hana_df.smart_save(raw_table, force=True)
        hana_df.head(150).save(train_table, force=True)
        hana_df.tail(30).save(test_table, force=True)
        cls.cc.table(test_table).select("BOOKING_DATE").save(predict_table, force=True)

        cls.raw_table = raw_table
        cls.train_table = train_table
        cls.test_table = test_table
        cls.predict_table = predict_table
        cls.created_tables.extend([raw_table, train_table, test_table, predict_table])

    @classmethod
    def _prepare_massive_sales_refunds_fixture(cls):
        raw_table = f"{cls.name_prefix}_MASSIVE_SALES_REFUNDS"
        train_table = f"{cls.name_prefix}_MASSIVE_SALES_REFUNDS_TRAIN"
        score_table = f"{cls.name_prefix}_MASSIVE_SALES_REFUNDS_TEST"
        predict_table = f"{cls.name_prefix}_MASSIVE_SALES_REFUNDS_PREDICT"

        pdf = cls.pd.read_csv(cls.csv_path)
        group_frames = []
        group_configs = [
            ("STORE_A", 1.0, 0.0),
            ("STORE_B", 1.15, 3.0),
            ("STORE_C", 0.85, -2.0),
        ]
        for store_id, scale, offset in group_configs:
            group_pdf = pdf.copy()
            group_pdf["STORE_ID"] = store_id
            group_pdf["REFUNDS"] = (group_pdf["REFUNDS"] * scale + offset).round(2)
            group_frames.append(group_pdf)

        massive_pdf = cls.pd.concat(group_frames, ignore_index=True)
        massive_hana_df = cls.dataframe.create_dataframe_from_pandas(
            cls.cc,
            pandas_df=massive_pdf,
            table_name=raw_table,
            force=True,
        ).to_datetime({"BOOKING_DATE": "YYYY-MM-DD"}).sort_values(by=["STORE_ID", "BOOKING_DATE"])
        massive_hana_df.smart_save(raw_table, force=True)

        train_parts = []
        score_parts = []
        predict_parts = []
        for store_id, _, _ in group_configs:
            store_df = massive_hana_df.filter(f'"STORE_ID" = \'{store_id}\'').sort_values(by="BOOKING_DATE")
            train_parts.append(store_df.head(150))
            score_parts.append(store_df.tail(30))
            predict_parts.append(store_df.tail(30).select("STORE_ID", "BOOKING_DATE"))

        union_train_sql = " UNION ALL ".join(df.select_statement for df in train_parts)
        union_score_sql = " UNION ALL ".join(df.select_statement for df in score_parts)
        union_predict_sql = " UNION ALL ".join(df.select_statement for df in predict_parts)
        cls.cc.sql(union_train_sql).smart_save(train_table, force=True)
        cls.cc.sql(union_score_sql).smart_save(score_table, force=True)
        cls.cc.sql(union_predict_sql).smart_save(predict_table, force=True)

        cls.massive_raw_table = raw_table
        cls.massive_train_table = train_table
        cls.massive_score_table = score_table
        cls.massive_predict_table = predict_table
        cls.massive_group_key = "STORE_ID"
        cls.created_tables.extend([raw_table, train_table, score_table, predict_table])

    def _new_agent(self, scenario_name: str):
        storage_dir = self.output_root / scenario_name
        storage_dir.mkdir(parents=True, exist_ok=True)
        tools = self.HANAMLToolkit(self.cc, used_tools="all").get_tools()
        return self.ContextAgent(
            llm=self.llm,
            tools=tools,
            storage_dir=str(storage_dir),
            config=self.AgentConfig(skills_use_llm_selector=True, max_active_skills=4),
            progress_bar=False,
        )

    def _extract_json_field(self, text: str, field_name: str):
        match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"([^"]+)"', text)
        if match:
            return match.group(1)

        fallback_prefixes = {
            "predicted_results_table": "PREDICT_RESULT_",
            "prediction_error_table": "PREDICT_ERROR_",
            "decomposed_and_reason_code_table": "REASON_CODE_",
            "scored_results_table": "SCORE_RESULT_",
        }
        prefix = fallback_prefixes.get(field_name)
        if prefix:
            return self._extract_table_name(text, prefix)
        return None

    def _extract_html_path(self, text: str):
        html_path = self._extract_json_field(text, "html_file")
        if html_path:
            return Path(html_path)
        match = re.search(r'(/[^"]+\.html)', text)
        return Path(match.group(1)) if match else None

    def _extract_model_version(self, text: str):
        match = re.search(r'"model_storage_version"\s*:\s*(\d+)', text)
        if match:
            return int(match.group(1))

        prose_match = re.search(r'\(version\s+(\d+)\)', text, re.IGNORECASE)
        if prose_match:
            return int(prose_match.group(1))

        prose_match = re.search(r'\bversion\s+(\d+)\b', text, re.IGNORECASE)
        return int(prose_match.group(1)) if prose_match else None

    def _extract_artifact_root(self, text: str):
        match = re.search(r'Root directory:\s*([^\n]+)', text)
        if not match:
            return None
        return Path(match.group(1).strip())

    def _resolve_model_version(self, model_name: str, response_text: str):
        version = self._extract_model_version(response_text)
        if version is not None:
            return version
        try:
            loaded_model = self.ModelStorage(self.cc).load_model(name=model_name, version=None)
        except Exception:
            return None
        return getattr(loaded_model, "version", None)

    def _extract_table_name(self, text: str, prefix: str):
        match = re.search(rf'({re.escape(prefix)}[^\s,]+)', text)
        return match.group(1) if match else None

    def _run_training_pipeline(self, agent, model_name: str):
        check_response = agent.chat(
            f"I want to check the time series data {self.train_table} and suggest the predict model for me. key is BOOKING_DATE and endog is REFUNDS."
        )
        self.assertTrue(any(token in check_response.lower() for token in ("forecast", "model", "seasonality", "trend")))

        train_response = agent.chat(
            f"Then please train the table using the suggested model and save as {model_name}"
        )
        self.assertIn(model_name, train_response)

        version = self._resolve_model_version(model_name, train_response)
        self.assertIsNotNone(version)
        self.created_models.append(model_name)

        predict_response = agent.chat(
            f"I want to predict the {self.predict_table} using the trained model, key is BOOKING_DATE and endog is REFUNDS."
        )
        predicted_table = self._extract_json_field(predict_response, "predicted_results_table")
        self.assertIsNotNone(predicted_table)
        self.assertTrue(self.cc.has_table(predicted_table))

        return {
            "check_response": check_response,
            "train_response": train_response,
            "predict_response": predict_response,
            "model_name": model_name,
            "model_version": version,
            "predicted_table": predicted_table,
        }

    def _run_massive_training_pipeline(self, agent, model_name: str):
        check_response = agent.chat(
            " ".join(
                [
                    f"I want to check the multiple time series data {self.massive_train_table}",
                    f"with group key {self.massive_group_key}, key is BOOKING_DATE and endog is REFUNDS,",
                    "and suggest the predict model for me.",
                ]
            )
        )
        self.assertTrue(any(token in check_response.lower() for token in ("massive", "group", "forecast", "model")))

        train_response = agent.chat(
            " ".join(
                [
                    f"Then please train the multiple time series table {self.massive_train_table}",
                    f"with group key {self.massive_group_key}, key BOOKING_DATE and endog REFUNDS,",
                    f"and save as {model_name}.",
                ]
            )
        )
        self.assertIn(model_name, train_response)
        version = self._resolve_model_version(model_name, train_response)
        self.assertIsNotNone(version)
        self.created_models.append(model_name)

        predict_response = agent.chat(
            " ".join(
                [
                    f"I want to predict the {self.massive_predict_table}",
                    f"using the trained model, group key is {self.massive_group_key},",
                    "key is BOOKING_DATE and endog is REFUNDS.",
                ]
            )
        )
        predicted_table = self._extract_json_field(predict_response, "predicted_results_table")
        self.assertIsNotNone(predicted_table)
        self.assertTrue(self.cc.has_table(predicted_table))

        return {
            "check_response": check_response,
            "train_response": train_response,
            "predict_response": predict_response,
            "model_name": model_name,
            "model_version": version,
            "predicted_table": predicted_table,
        }

    def test_e2e_data_preview_and_dataset_report_scenario(self):
        agent = self._new_agent("preview_and_report")

        preview_response = agent.chat(f"Show me the last 10 records of data from {self.raw_table}")
        self.assertTrue(self.raw_table in preview_response or "|" in preview_response)

        report_response = agent.chat(
            f"Then create a dataset report for me on {self.raw_table} table, key is BOOKING_DATE and endog is REFUNDS"
        )
        html_path = self._extract_html_path(report_response)
        self.assertIsNotNone(html_path)
        self.assertTrue(html_path.exists())

    def test_e2e_forecast_train_predict_scenario(self):
        agent = self._new_agent("forecast_train_predict")
        model_name = f"{self.name_prefix}_MODEL_FORECAST"

        pipeline = self._run_training_pipeline(agent, model_name)

        self.assertEqual(model_name, pipeline["model_name"])
        self.assertTrue(self.cc.has_table(pipeline["predicted_table"]))

    def test_e2e_prediction_visualization_and_insight_scenario(self):
        agent = self._new_agent("prediction_analysis")
        model_name = f"{self.name_prefix}_MODEL_ANALYSIS"

        self._run_training_pipeline(agent, model_name)

        plot_response = agent.chat(
            f"Generate the line plot on the predicted results table and compared with the actual table {self.raw_table}"
        )
        plot_path = self._extract_html_path(plot_response)
        self.assertIsNotNone(plot_path)
        self.assertTrue(plot_path.exists())

        insights_response = agent.chat("Give some insights from the predicted results.")
        self.assertTrue(any(token in insights_response.lower() for token in ("rmse", "mape", "mad", "bias", "error")))

    def test_e2e_prediction_visualization_renders_in_notebook_scenario(self):
        agent = self._new_agent("prediction_analysis_notebook_render")
        model_name = f"{self.name_prefix}_MODEL_NOTEBOOK_RENDER"

        self._run_training_pipeline(agent, model_name)

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
            plot_response = agent.chat(
                f"Generate the line plot on the predicted results table and compared with the actual table {self.raw_table}"
            )

        plot_path = self._extract_html_path(plot_response)
        self.assertIsNotNone(plot_path)
        self.assertTrue(plot_path.exists())
        self.assertTrue(rendered)
        self.assertTrue(hasattr(rendered[0], "data"))
        self.assertIn("<html", rendered[0].data.lower())
        self.assertIn("Rendered HTML artifact", plot_response)

    def test_e2e_cap_artifact_generation_scenario(self):
        agent = self._new_agent("cap_artifacts")
        model_name = f"{self.name_prefix}_MODEL_CAP"

        self._run_training_pipeline(agent, model_name)

        cap_output_dir = self.output_root / "cap_output"
        cap_response = agent.chat(
            f"I want to generate CAP artifacts, the project name is my_project and output path is {cap_output_dir}"
        )
        artifact_root = self._extract_artifact_root(cap_response)
        self.assertIsNotNone(artifact_root)
        self.assertTrue((artifact_root / "package.json").exists())

    def test_e2e_dataset_import_split_and_profile_scenario(self):
        agent = self._new_agent("dataset_import_split_profile")
        imported_table = f"{self.name_prefix}_IMPORTED_SALES_REFUNDS"
        imported_train = f"{imported_table}_TRAIN"
        imported_test = f"{imported_table}_TEST"
        imported_validation = f"{imported_table}_VALIDATION"

        import_response = agent.chat(
            " ".join(
                [
                    f"Please import the CSV file {self.csv_path} into HANA table {imported_table}.",
                    "The CSV has a header row, parse BOOKING_DATE as a date column, and overwrite the table if it already exists.",
                ]
            )
        )
        self.assertTrue(self.cc.has_table(imported_table))
        self.assertTrue(any(token in import_response.lower() for token in ("rows_imported", imported_table.lower(), "preview")))

        split_response = agent.chat(
            " ".join(
                [
                    f"Now split {imported_table} into {imported_train}, {imported_test}, and {imported_validation}",
                    "for forecasting using time order on BOOKING_DATE.",
                    "Use 0.75, 0.15, and 0.10 as the ratios and overwrite existing tables if needed.",
                ]
            )
        )
        self.assertTrue(self.cc.has_table(imported_train))
        self.assertTrue(self.cc.has_table(imported_test))
        self.assertTrue(self.cc.has_table(imported_validation))
        self.assertTrue(any(token in split_response for token in (imported_train, imported_test, imported_validation)))

        report_response = agent.chat(
            f"Then create a dataset report for me on {imported_train} table, key is BOOKING_DATE and endog is REFUNDS"
        )
        html_path = self._extract_html_path(report_response)
        self.assertIsNotNone(html_path)
        self.assertTrue(html_path.exists())

    def test_e2e_prediction_auto_repairs_labelled_holdout_scenario(self):
        agent = self._new_agent("prediction_auto_repair")
        model_name = f"{self.name_prefix}_MODEL_REPAIR"

        self._run_training_pipeline(agent, model_name)

        repair_response = agent.chat(
            " ".join(
                [
                    f"I accidentally used the labeled holdout table {self.test_table} for prediction.",
                    "Please use the same trained model anyway.",
                    "The key is BOOKING_DATE and the endog is REFUNDS.",
                ]
            )
        )
        predicted_table = self._extract_json_field(repair_response, "predicted_results_table")
        self.assertIsNotNone(predicted_table)
        self.assertTrue(self.cc.has_table(predicted_table))
        self.assertTrue(
            any(
                token in repair_response.lower()
                for token in (
                    "auto_repaired_predict_input",
                    "auto-corrected the prediction input",
                    "predict_table_columns_used_for_prediction",
                )
            )
        )

    def test_e2e_outlier_detection_scenario(self):
        agent = self._new_agent("outlier_detection")

        response = agent.chat(
            f"Detect outliers in table {self.train_table}. The key is BOOKING_DATE and the endog is REFUNDS."
        )
        self.assertTrue("outlier" in response.lower())
        self.assertTrue("result_select_statement" in response or "outliers" in response.lower())

    def test_e2e_massive_forecasting_scenario(self):
        agent = self._new_agent("massive_forecasting")
        model_name = f"{self.name_prefix}_MODEL_MASSIVE"

        pipeline = self._run_massive_training_pipeline(agent, model_name)

        self.assertEqual(model_name, pipeline["model_name"])
        self.assertTrue(self.cc.has_table(pipeline["predicted_table"]))
        predicted_df = self.cc.table(pipeline["predicted_table"]).head(5).collect()
        self.assertIn(self.massive_group_key, predicted_df.columns)
        self.assertTrue(any(column in predicted_df.columns for column in ("BOOKING_DATE", "ID")))

    def test_e2e_massive_additive_forecasting_scenario(self):
        agent = self._new_agent("massive_additive_forecasting")
        model_name = f"{self.name_prefix}_MODEL_MASSIVE_ADDITIVE"

        check_response = agent.chat(
            " ".join(
                [
                    f"I want to forecast the grouped table {self.massive_train_table}",
                    f"with group key {self.massive_group_key}, key BOOKING_DATE and endog REFUNDS.",
                    "Please use a grouped additive forecast model so I can inspect trend and seasonality behavior.",
                ]
            )
        )
        self.assertTrue(any(token in check_response.lower() for token in ("additive", "seasonality", "trend", "group")))

        train_response = agent.chat(
            " ".join(
                [
                    f"Then train that grouped additive forecast model on {self.massive_train_table}",
                    f"with group key {self.massive_group_key}, key BOOKING_DATE and endog REFUNDS,",
                    f"and save it as {model_name}.",
                ]
            )
        )
        self.assertIn(model_name, train_response)
        self.created_models.append(model_name)

        saved_model = self.ModelStorage(self.cc).load_model(name=model_name, version=None)
        self.assertEqual(type(saved_model).__name__, "AdditiveModelForecast")
        self.assertTrue(getattr(saved_model, "massive", False))

        predict_response = agent.chat(
            " ".join(
                [
                    f"Now predict {self.massive_predict_table} with the grouped additive model {model_name}.",
                    f"The group key is {self.massive_group_key} and the key is BOOKING_DATE.",
                    "If the model can provide decomposition or extra grouped diagnostics, include them.",
                ]
            )
        )
        predicted_table = self._extract_json_field(predict_response, "predicted_results_table")
        if predicted_table is None:
            predicted_table = self._extract_table_name(predict_response, "PREDICT_RESULT_")
        error_table = self._extract_json_field(predict_response, "prediction_error_table")
        if error_table is None:
            error_table = self._extract_table_name(predict_response, "PREDICT_ERROR_")
        self.assertIsNotNone(predicted_table)
        self.assertIsNotNone(error_table)
        self.assertTrue(self.cc.has_table(predicted_table))
        self.assertTrue(self.cc.has_table(error_table))

        predicted_df = self.cc.table(predicted_table).head(5).collect()
        self.assertTrue(any(column in predicted_df.columns for column in (self.massive_group_key, "GROUP_ID")))
        self.assertTrue(any(column in predicted_df.columns for column in ("BOOKING_DATE", "ID")))

    def test_e2e_massive_outlier_detection_scenario(self):
        agent = self._new_agent("massive_outlier_detection")

        response = agent.chat(
            " ".join(
                [
                    f"Detect outliers in multiple time series table {self.massive_train_table}.",
                    f"The group key is {self.massive_group_key}, key is BOOKING_DATE and endog is REFUNDS.",
                ]
            )
        )
        self.assertTrue("outlier" in response.lower())
        self.assertTrue("result_select_statement" in response or "outliers" in response.lower())


if __name__ == "__main__":
    unittest.main()