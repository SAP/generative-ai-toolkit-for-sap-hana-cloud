import os
import json
import re
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler


class _ToolEvidence(BaseCallbackHandler):
    """Observe actual tool calls; never replace their implementations/results."""

    def __init__(self):
        self.records = []
        self.pending = {}

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        self.pending[run_id] = {"name": serialized["name"], "inputs": inputs or input_str}

    def on_tool_end(self, output, *, run_id, **kwargs):
        record = self.pending.pop(run_id)
        content = getattr(output, "content", output)
        try:
            content = json.loads(content) if isinstance(content, str) else content
        except (ValueError, TypeError):
            pass
        record["output"] = content
        self.records.append(record)

    def on_tool_error(self, error, *, run_id, **kwargs):
        record = self.pending.pop(run_id, {})
        record["output"] = {"error": str(error)}
        self.records.append(record)


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
    connection via `HANA_USERKEY` (default `XinKey2`) and a working LLM setup for
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
            assertions = html plot exists, raw HTML display stays disabled by default
        - cap_artifacts:
            prompt flow = train/predict -> generate CAP artifacts
            assertions = artifact root exists and contains package.json
        - outlier_detection:
            prompt flow = detect outliers on a single-series table
            assertions = actual outlier tool result matches the HANA result rows
        - massive_forecasting:
            prompt flow = massive ts_check -> grouped training -> grouped prediction
            assertions = predicted table exists and contains group key + time key
        - massive_additive_forecasting:
            prompt flow = grouped additive forecast request -> grouped additive training -> grouped additive prediction
            assertions = saved model is AdditiveModelForecast in massive mode, predicted table exists, prediction error table exists
        - massive_outlier_detection:
            prompt flow = detect grouped outliers on the massive training table
            assertions = original groups/dates survive and result rows cover all inputs
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

        cls._env_patch = patch.dict(os.environ, {
            "SSL_CERT_FILE": certifi.where(), "REQUESTS_CA_BUNDLE": certifi.where(),
            "HANA_ALLOW_RAW_HTML_RENDER": "false",
        })
        cls._env_patch.start()
        cls.addClassCleanup(cls._env_patch.stop)

        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.scenario_dir = cls.repo_root / "nutest" / "testscripts" / "demo" / "e2e_scenarios"
        cls.csv_path = cls.scenario_dir / "SALES_REFUNDS.csv"
        cls.output_root = Path(tempfile.mkdtemp(prefix="context_agent_e2e_"))
        cls.name_prefix = f"CTXE2E_{uuid.uuid4().hex[:8].upper()}"
        cls.created_models = []
        cls.created_tables = []
        cls.cc = None
        cls.addClassCleanup(cls._cleanup_fixtures)

        # Use the HANA client's native certificate trust configuration. A certifi
        # CA bundle is not a Windows sslKeyStore (client identity store).
        cls.cc = dataframe.ConnectionContext(
            userkey=os.environ.get("HANA_USERKEY", "XinKey2"),
            sslValidateCertificate=True,
            encrypt=True,
        )
        cls.llm = init_llm(os.environ.get("CONTEXT_AGENT_E2E_LLM_MODEL", "gpt-4.1"), temperature=0.0, max_tokens=1800)
        cls._prepare_sales_refunds_fixture()
        cls._prepare_massive_sales_refunds_fixture()

    @classmethod
    def _cleanup_fixtures(cls):
        """Clean only this run's uniquely named synthetic DB and filesystem objects."""
        try:
            if cls.cc is not None:
                assert re.fullmatch(r"CTXE2E_[0-9A-F]{8}", cls.name_prefix)
                for model_name in reversed(cls.created_models):
                    assert model_name.startswith(cls.name_prefix + "_")
                    # A failed training call may never have created the registered model.
                    storage = cls.ModelStorage(cls.cc)
                    if not storage.list_models(name=model_name).empty:
                        storage.delete_models(name=model_name)
                cursor = cls.cc.connection.cursor()
                try:
                    pattern = "%" + cls.name_prefix.replace("_", "!_") + "!_%"
                    cursor.execute("SELECT TABLE_NAME FROM SYS.TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME LIKE ? ESCAPE '!'", (pattern,))
                    tables = [row[0] for row in cursor.fetchall()]
                    from hana_ml.dataframe import quotename
                    for table_name in tables:
                        assert cls.name_prefix + "_" in table_name
                        cursor.execute("DROP TABLE " + quotename(table_name))
                    cursor.execute("SELECT COUNT(*) FROM SYS.TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME LIKE ? ESCAPE '!'", (pattern,))
                    assert cursor.fetchone()[0] == 0
                    print(f"Cleaned synthetic prefix {cls.name_prefix}: {len(tables)} tables; remaining=0")
                finally:
                    cursor.close()
        finally:
            if cls.cc is not None:
                cls.cc.close()
            root = cls.output_root.resolve()
            assert root.parent == Path(tempfile.gettempdir()).resolve()
            assert root.name.startswith("context_agent_e2e_")
            shutil.rmtree(root)

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
        self._evidence = _ToolEvidence()
        for tool in tools:
            tool.callbacks = [self._evidence]
        return self.ContextAgent(
            llm=self.llm,
            tools=tools,
            storage_dir=str(storage_dir),
            config=self.AgentConfig(
                skills_use_llm_selector=True, max_active_skills=4,
                enable_structured_notes=False, enable_compaction=False, enable_session_summary=False,
            ),
            progress_bar=False,
        )

    def _chat(self, agent, prompt):
        # ContextPack truncates the task layer at this character budget. Never
        # silently test a partial user request (for example an incomplete JSON).
        self.assertLessEqual(len("User question: " + prompt.strip()), agent.config.budgets.budget_task)
        self._evidence.records.clear()
        response = agent.chat(prompt)
        print(f"[ContextE2E] {self._testMethodName}: tools="
              f"{[record['name'] for record in self._evidence.records]}")
        return response

    def _tool_result(self, *names):
        for record in reversed(self._evidence.records):
            if record["name"] in names:
                result = record["output"]
                self.assertFalse(isinstance(result, dict) and "error" in result, str(result))
                return result
        self.fail(f"Expected a real call to {names}; observed {self._evidence.records}")

    def _assert_html_artifact(self, tool_name):
        result = self._tool_result(tool_name)
        self.assertIsInstance(result, dict)
        path = Path(result["html_file"])
        self.assertTrue(path.is_file())
        self.assertIn("<html", path.read_text(encoding="utf-8").lower())
        return path

    def _assert_prediction(self, result, *, massive=False):
        self.assertIsInstance(result, dict)
        table = result["predicted_results_table"]
        self.assertTrue(self.cc.has_table(table))
        frame = self.cc.table(table).collect()
        self.assertEqual(len(frame), 90 if massive else 30)
        self.assertTrue(any(column in frame.columns for column in ("BOOKING_DATE", "ID")))
        if massive:
            group_column = self.massive_group_key if self.massive_group_key in frame.columns else "GROUP_ID"
            self.assertEqual(set(frame[group_column]), {"STORE_A", "STORE_B", "STORE_C"})
            self.assertEqual(set(frame.groupby(group_column).size()), {30})
        return table

    def _assert_outliers(self, result, *, massive=False):
        self.assertIsInstance(result, dict)
        frame = self.cc.sql(result["result_select_statement"]).collect()
        self.assertEqual(len(frame), 450 if massive else 150)
        detected = frame.loc[frame["IS_OUTLIER"] == 1]
        self.assertGreater(len(detected), 0)
        if massive:
            self.assertEqual(set(frame[self.massive_group_key]), {"STORE_A", "STORE_B", "STORE_C"})
            self.assertEqual(set(frame.groupby(self.massive_group_key).size()), {150})
            actual = {item["group_id"]: [self.pd.Timestamp(value) for value in item["outlier_ids"]]
                      for item in result["outliers"]}
            expected = {group: [self.pd.Timestamp(value) for value in rows["BOOKING_DATE"]]
                        for group, rows in detected.groupby(self.massive_group_key)}
            self.assertEqual({key: sorted(values) for key, values in actual.items()},
                             {key: sorted(values) for key, values in expected.items()})
        else:
            self.assertCountEqual([self.pd.Timestamp(value) for value in result["outliers"]],
                                  [self.pd.Timestamp(value) for value in detected.iloc[:, 0]])

    def _extract_json_field(self, text: str, field_name: str):
        for record in reversed(self._evidence.records):
            result = record["output"]
            if isinstance(result, dict) and field_name in result and "error" not in result:
                return result[field_name]
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
        for record in reversed(self._evidence.records):
            if record["name"] == "cap_artifacts":
                text = str(record["output"])
                break
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
        self.created_models.append(model_name)
        self._chat(agent,
            f"Check time series {self.train_table}, key BOOKING_DATE, endog REFUNDS. "
            "Inspect the data and report trend/seasonality; do not train yet."
        )
        self._tool_result("ts_check")
        # Pick a deterministic algorithm for this fixture; massive AutoML is
        # exercised separately with an explicitly selected PAL configuration.
        self._chat(agent,
            f"Train an additive model with additive_model_forecast_fit_and_save on {self.train_table}. "
            f"Save it under the exact name {model_name}, version 1. "
            "key=BOOKING_DATE, endog=REFUNDS, exog=null, growth=linear. "
            "I confirm these choices; execute the training now."
        )
        trained = self._tool_result("additive_model_forecast_fit_and_save")
        self.assertEqual(trained["model_storage_name"], model_name)
        version = trained["model_storage_version"]
        saved = self.ModelStorage(self.cc).load_model(name=model_name, version=version)
        self.assertEqual(type(saved).__name__, "AdditiveModelForecast")
        predict_response = self._chat(agent,
            f"Predict {self.predict_table} using additive_model_forecast_load_model_and_predict "
            f"with name={model_name}, version={version}, key=BOOKING_DATE, exog=null. "
            "The input has only BOOKING_DATE; no label or exogenous columns. Execute now."
        )
        result = self._tool_result("additive_model_forecast_load_model_and_predict")
        predicted_table = self._assert_prediction(result)
        return {"model_name": model_name, "model_version": version,
                "predicted_table": predicted_table, "predict_response": predict_response}

    def _run_massive_training_pipeline(self, agent, model_name: str):
        # PAL AutoML requires equally spaced timestamps. The demo CSV has a
        # missing calendar day. Use explicitly synthetic daily dates for this
        # scenario only; do not alter the CSV or the other grouped fixtures.
        fixture = self.cc.table(self.massive_raw_table).collect().sort_values(["STORE_ID", "BOOKING_DATE"])
        day_number = fixture.groupby("STORE_ID").cumcount()
        fixture["BOOKING_DATE"] = self.pd.Timestamp("2024-01-01") + self.pd.to_timedelta(day_number, unit="D")
        fit_table = f"{self.name_prefix}_AUTOML_TRAIN"
        predict_table = f"{self.name_prefix}_AUTOML_PREDICT"
        self.dataframe.create_dataframe_from_pandas(
            self.cc, pandas_df=fixture.loc[day_number < 150], table_name=fit_table, force=False,
        )
        self.dataframe.create_dataframe_from_pandas(
            self.cc, pandas_df=fixture.loc[day_number >= 150, ["STORE_ID", "BOOKING_DATE"]],
            table_name=predict_table, force=False,
        )
        self.created_models.append(model_name)
        self._chat(agent,
            f"Check grouped time series {fit_table}, group_key=STORE_ID, "
            "key=BOOKING_DATE, endog=REFUNDS. Do not train or select a model name yet."
        )
        self._tool_result("massive_ts_check")
        self._chat(agent,
            "I choose PAL's default AutoML search space. First call get_automl_config_dict "
            "with pipeline_type=timeseries, config_dict=default. Use its returned config_dict "
            "verbatim in the next training tool call, in this same turn. "
            f"Then train massive_automatic_timeseries_fit_and_save on {fit_table}, "
            f"name={model_name}, version=1, group_key=STORE_ID, key=BOOKING_DATE, "
            "endog=REFUNDS, exog=null. For this integration test, use generations=1, "
            "population_size=4, offspring_size=4, elite_number=1, fold_num=2, "
            "random_seed=42, max_eval_time_mins=1.0. "
            "I approve these settings. Execute both tools now without another confirmation."
        )
        config = self._tool_result("get_automl_config_dict")["config_dict"]
        trained = self._tool_result("massive_automatic_timeseries_fit_and_save")
        self.assertFalse(trained.get("group_errors"), str(trained))
        training_call = next(record for record in reversed(self._evidence.records)
                             if record["name"] == "massive_automatic_timeseries_fit_and_save")
        self.assertEqual(training_call["inputs"]["config_dict"], config)
        self.assertEqual(trained["model_storage_name"], model_name)
        version = trained["model_storage_version"]
        self.ModelStorage(self.cc).load_model(name=model_name, version=version)
        self._chat(agent,
            f"Predict {predict_table} using massive_automatic_timeseries_load_model_and_predict "
            f"with name={model_name}, version={version}, group_key=STORE_ID, key=BOOKING_DATE, exog=null. "
            "This input contains only STORE_ID and BOOKING_DATE. Execute the prediction now."
        )
        result = self._tool_result("massive_automatic_timeseries_load_model_and_predict")
        return {"model_name": model_name, "model_version": version,
                "predicted_table": self._assert_prediction(result, massive=True)}

    def test_e2e_data_preview_and_dataset_report_scenario(self):
        agent = self._new_agent("preview_and_report")

        preview_response = self._chat(agent, f"Show me the last 10 records of data from {self.raw_table}")
        self.assertTrue(self.raw_table in preview_response or "|" in preview_response)

        report_response = self._chat(agent,
            f"Then create a dataset report for me on {self.raw_table} table, key is BOOKING_DATE and endog is REFUNDS, output_dir={self.output_root / 'preview_report'}"
        )
        self._assert_html_artifact("ts_dataset_report")

    def test_e2e_forecast_train_predict_scenario(self):
        agent = self._new_agent("forecast_train_predict")
        model_name = f"{self.name_prefix}_MODEL_FORECAST"

        pipeline = self._run_training_pipeline(agent, model_name)

        self.assertEqual(model_name, pipeline["model_name"])
        self.assertTrue(self.cc.has_table(pipeline["predicted_table"]))

    def test_e2e_prediction_visualization_and_insight_scenario(self):
        agent = self._new_agent("prediction_analysis")
        model_name = f"{self.name_prefix}_MODEL_ANALYSIS"

        pipeline = self._run_training_pipeline(agent, model_name)

        plot_response = self._chat(agent,
            f"Generate forecast_line_plot with predict_result={pipeline['predicted_table']}, actual_table={self.raw_table}, output_dir={self.output_root / 'analysis_plot'}."
        )
        self._assert_html_artifact("forecast_line_plot")

        insights_response = self._chat(agent,
            f"Evaluate the predicted results with accuracy_measure: predict_table={pipeline['predicted_table']}, "
            f"actual_table={self.test_table}, predict_key=BOOKING_DATE, actual_key=BOOKING_DATE, "
            "predict_target=YHAT, actual_target=REFUNDS, evaluation_metric=[rmse, mad]. "
            "The actual table contains observed refunds for all 30 forecast dates. "
            "Compute these metrics now, then explain the forecast accuracy."
        )
        metrics = self._tool_result("accuracy_measure")
        compared = self.cc.table(pipeline["predicted_table"]).collect().merge(
            self.cc.table(self.test_table).collect(), on="BOOKING_DATE", validate="one_to_one",
        )
        self.assertEqual(len(compared), 30)
        errors = compared["YHAT"] - compared["REFUNDS"]
        self.assertAlmostEqual(metrics["RMSE"], float((errors ** 2).mean() ** 0.5), places=6)
        self.assertAlmostEqual(metrics["MAD"], float(errors.abs().mean()), places=6)
        self.assertIn("rmse", insights_response.lower())
        self.assertIn("mad", insights_response.lower())

    def test_e2e_prediction_visualization_renders_in_notebook_scenario(self):
        agent = self._new_agent("prediction_analysis_notebook_render")
        model_name = f"{self.name_prefix}_MODEL_NOTEBOOK_RENDER"

        pipeline = self._run_training_pipeline(agent, model_name)

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
            plot_response = self._chat(agent,
                f"Generate forecast_line_plot with predict_result={pipeline['predicted_table']}, actual_table={self.raw_table}, output_dir={self.output_root / 'notebook_plot'}."
            )

        plot_path = self._assert_html_artifact("forecast_line_plot")
        self.assertEqual(rendered, [], "Untrusted raw HTML must not be displayed by default")
        self.assertNotIn("Rendered HTML artifact", plot_response)
        self.assertIn(str(plot_path), plot_response.replace("/", os.sep))

    def test_e2e_cap_artifact_generation_scenario(self):
        agent = self._new_agent("cap_artifacts")
        model_name = f"{self.name_prefix}_MODEL_CAP"

        pipeline = self._run_training_pipeline(agent, model_name)

        cap_output_dir = self.output_root / "cap_output"
        cap_response = self._chat(agent,
            f"Generate CAP artifacts with cap_artifacts for model name={model_name}, version={pipeline['model_version']}, project_name=my_project, output_dir={cap_output_dir}, archive=false."
        )
        self._tool_result("cap_artifacts")
        artifact_root = self._extract_artifact_root(cap_response)
        self.assertIsNotNone(artifact_root)
        self.assertEqual(artifact_root.resolve(), (cap_output_dir / "my_project").resolve())
        self.assertTrue((artifact_root / "package.json").exists())
        self.assertIsInstance(json.loads((artifact_root / "package.json").read_text(encoding="utf-8")), dict)

    def test_e2e_dataset_import_split_and_profile_scenario(self):
        agent = self._new_agent("dataset_import_split_profile")
        imported_table = f"{self.name_prefix}_IMPORTED_SALES_REFUNDS"
        imported_train = f"{imported_table}_TRAIN"
        imported_test = f"{imported_table}_TEST"
        imported_validation = f"{imported_table}_VALIDATION"

        import_response = self._chat(agent,
            " ".join(
                [
                    f"Please import the CSV file {self.csv_path} into HANA table {imported_table}.",
                    "The CSV has a header row, parse BOOKING_DATE as a date column, and overwrite the table if it already exists.",
                ]
            )
        )
        self.assertTrue(self.cc.has_table(imported_table))
        self.assertEqual(self.cc.table(imported_table).count(), len(self.pd.read_csv(self.csv_path)))
        self.assertTrue(any(token in import_response.lower() for token in ("rows_imported", imported_table.lower(), "preview")))

        split_response = self._chat(agent,
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
        self.assertEqual(sum(self.cc.table(name).count() for name in
                             (imported_train, imported_test, imported_validation)), self.cc.table(imported_table).count())
        train_dates = self.cc.table(imported_train).collect()["BOOKING_DATE"]
        test_dates = self.cc.table(imported_test).collect()["BOOKING_DATE"]
        validation_dates = self.cc.table(imported_validation).collect()["BOOKING_DATE"]
        self.assertEqual((len(train_dates), len(test_dates), len(validation_dates)), (135, 27, 18))
        self.assertLess(max(train_dates), min(test_dates))
        self.assertLess(max(test_dates), min(validation_dates))
        self.assertTrue(any(token in split_response for token in (imported_train, imported_test, imported_validation)))

        report_response = self._chat(agent,
            f"Then create a dataset report for me on {imported_train} table, key is BOOKING_DATE and endog is REFUNDS, output_dir={self.output_root / 'import_report'}"
        )
        self._assert_html_artifact("ts_dataset_report")

    def test_e2e_prediction_auto_repairs_labelled_holdout_scenario(self):
        agent = self._new_agent("prediction_auto_repair")
        model_name = f"{self.name_prefix}_MODEL_REPAIR"

        pipeline = self._run_training_pipeline(agent, model_name)

        repair_response = self._chat(agent,
            " ".join(
                [
                    f"Use additive_model_forecast_load_model_and_predict with predict_table={self.test_table}.",
                    f"Use name={model_name}, version={pipeline['model_version']}, key=BOOKING_DATE, exog=null.",
                    "This labeled holdout also contains REFUNDS. The prediction tool supports automatic input repair.",
                    "Pass the original table unchanged so the tool can remove the label; do not create a replacement table.",
                ]
            )
        )
        result = self._tool_result("additive_model_forecast_load_model_and_predict")
        self._assert_prediction(result)
        self.assertEqual(result["input_predict_table"], self.test_table)
        self.assertIs(result["auto_repaired_predict_input"], True)
        self.assertIn("REFUNDS", result["predict_table_columns_before_repair"])
        self.assertEqual(result["predict_table_columns_used_for_prediction"], ["BOOKING_DATE"])
        self.assertIn("REFUNDS", self.cc.table(self.test_table).columns)

    def test_e2e_outlier_detection_scenario(self):
        agent = self._new_agent("outlier_detection")

        response = self._chat(agent,
            f"Detect outliers in table {self.train_table}. The key is BOOKING_DATE and the endog is REFUNDS."
        )
        self._assert_outliers(self._tool_result("ts_outlier_detection"))

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

        check_response = self._chat(agent,
            " ".join(
                [
                    f"I want to forecast the grouped table {self.massive_train_table}",
                    f"with group key {self.massive_group_key}, key BOOKING_DATE and endog REFUNDS.",
                    "Check trend and seasonality only. Do not train yet; I will give you the model name next.",
                ]
            )
        )
        self.assertTrue(any(token in check_response.lower() for token in ("additive", "seasonality", "trend", "group")))
        self.created_models.append(model_name)

        train_response = self._chat(agent,
            " ".join(
                [
                    f"Now execute massive_additive_model_forecast_fit_and_save on {self.massive_train_table}",
                    f"with group key {self.massive_group_key}, key BOOKING_DATE and endog REFUNDS,",
                    f"and save it under the exact name {model_name}, version=1, exog=null, growth=linear. I confirm these settings; train now.",
                ]
            )
        )
        trained = self._tool_result("massive_additive_model_forecast_fit_and_save")
        self.assertEqual(trained["model_storage_name"], model_name)

        saved_model = self.ModelStorage(self.cc).load_model(name=model_name, version=None)
        self.assertEqual(type(saved_model).__name__, "AdditiveModelForecast")
        self.assertTrue(getattr(saved_model, "massive", False))

        predict_response = self._chat(agent,
            " ".join(
                [
                    f"Now execute massive_additive_model_forecast_load_model_and_predict on {self.massive_predict_table} with name={model_name}, version={trained['model_storage_version']}.",
                    f"The group_key is {self.massive_group_key}, key=BOOKING_DATE and exog=null. I confirm the input contains only group and time keys.",
                    "If the model can provide decomposition or extra grouped diagnostics, include them.",
                ]
            )
        )
        result = self._tool_result("massive_additive_model_forecast_load_model_and_predict")
        self._assert_prediction(result, massive=True)
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

        response = self._chat(agent,
            " ".join(
                [
                    f"Detect outliers in multiple time series table {self.massive_train_table}.",
                    f"The group key is {self.massive_group_key}, key is BOOKING_DATE and endog is REFUNDS.",
                ]
            )
        )
        self._assert_outliers(self._tool_result("massive_ts_outlier_detection"), massive=True)


if __name__ == "__main__":
    unittest.main()
