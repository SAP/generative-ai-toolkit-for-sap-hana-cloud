# ContextAgent Skills Draft

This document extracts the current built-in skills from ContextAgent into a standalone draft and compares them against the real tool surface exposed by HANAMLToolkit.

It is intended as a design artifact first. It does not change runtime behavior by itself.

## Scope

- Source implementation: src/hana_ai/iagents/context_agent.py
- Tool surface: src/hana_ai/tools/toolkit.py and the default hana_ml_tools it registers

## Extracted Current Skills

## timeseries_forecasting

Status: extracted from current built-in skill

Goal:
- Analyze a time series table, suggest a model, train it, predict future values, and summarize the result.

When to activate:
- The user asks to forecast a single time series.
- The user asks for model suggestion and then training/prediction.
- The user asks to continue from dataset inspection into forecasting.

Required inputs:
- Training table
- Key or time column
- Target or endog column

Preferred tools:
- ts_dataset_report
- ts_check
- automatic_timeseries_fit_and_save
- additive_model_forecast_fit_and_save
- intermittent_forecast
- ts_make_future_table
- automatic_timeseries_load_model_and_predict
- additive_model_forecast_load_model_and_predict
- forecast_line_plot

Workflow:
1. Confirm the training table, key column, and target column. Ask if any are missing.
2. Use ts_dataset_report and ts_check to understand structure, seasonality, trend, and ADI/CV^2-based intermittent behavior.
3. Suggest a model:
   - Prefer automatic_timeseries_fit_and_save for the default single-series path.
   - Use additive_model_forecast_fit_and_save when trend and seasonality assumptions are explicit.
  - Use intermittent_forecast when ts_check reports ADI >= 1.32, which indicates sparse demand occurrence.
  - Prefer intermittent_forecast with method=constant when ts_check classifies the series as intermittent: ADI >= 1.32 and CV^2 < 0.49.
  - Prefer intermittent_forecast with method=sporadic when ts_check classifies the series as lumpy: ADI >= 1.32 and CV^2 >= 0.49.
4. Train and save the model. Record the exact model name and version.
5. If a future input table is needed, build it with ts_make_future_table.
6. Predict with the matching load_model_and_predict tool.
7. Summarize the returned result table names and next possible actions.

Rules:
- Do not guess missing table or column names.
- Do not fabricate model quality or output rows.
- Always return the exact model name, version, and prediction table names produced by the tool chain.

## prediction_result_analysis

Status: extracted from current built-in skill

Goal:
- Analyze a forecast output table and compare predictions against actuals when an actual table is available.

When to activate:
- The user asks for insights from predicted results.
- The user asks whether the forecast is good, accurate, biased, or usable.
- The user asks to compare predicted and actual data.

Required inputs:
- Prediction result table name
- Actual table name when comparison is requested
- Key column on both sides
- Target column on both sides

Preferred tools:
- accuracy_measure
- SelectStatement_to_table
- fetch_data
- forecast_line_plot

Workflow:
1. Identify the prediction output table from prior context or ask for it explicitly.
2. If actual data is available, compare predicted and actual on the key column.
3. Prefer accuracy_measure for MAD, RMSE, MAPE, sMAPE, WMAPE, or related supported metrics.
4. If a custom comparison table is needed, use SelectStatement_to_table to materialize a join result and fetch_data to inspect it.
5. Report error metrics, directional bias, obvious misses, interval availability, and data quality caveats.
6. If a visual comparison is requested, use forecast_line_plot.

Rules:
- Use only metric names supported by accuracy_measure.
- When joining predicted and actual data, state the exact tables and columns used.
- Do not claim statistical significance or production readiness without tool-backed evidence.

## massive_forecast_comparison

Status: extracted from current built-in skill

Goal:
- Validate and compare large-scale multi-series forecasts produced by MassiveAutomaticTimeSeries tools.

When to activate:
- The user asks about many series, grouped forecasting, or group_key-based prediction quality.

Required inputs:
- Training or prediction table
- group_key
- key column
- target or endog column

Preferred tools:
- massive_ts_check
- massive_automatic_timeseries_fit_and_save
- ts_make_future_table_for_massive_forecast
- massive_automatic_timeseries_load_model_and_predict
- massive_automatic_timeseries_load_model_and_score
- accuracy_measure
- SelectStatement_to_table
- fetch_data

Workflow:
1. Use massive_ts_check to understand multi-series characteristics.
2. Train and predict with the MassiveAutomaticTimeSeries tool chain.
3. Use scoring or accuracy_measure where possible.
4. Materialize comparison tables when per-group inspection is needed.
5. Report weak groups, error distribution, and next remediation steps.

Rules:
- Always include the group_key in comparison logic.
- Keep per-group results reproducible by naming the tables and join keys.

## data_ingestion_and_dataset_preparation

Status: recommended new standalone skill

Goal:
- Import user-provided CSV data into HANA and prepare train, test, and validation tables with minimal manual coding.

When to activate:
- The user asks to import a CSV file into HANA.
- The user asks to create training, test, or validation tables.
- The user asks to prepare a dataset before profiling or forecasting.

Required inputs:
- CSV path when importing a file
- Target table name for the imported dataset
- Split ratios and split mode when generating modeling datasets
- Order column for time-ordered splits

Preferred tools:
- import_csv_to_table
- split_table_for_forecasting
- fetch_data
- ts_dataset_report

Workflow:
1. Confirm the file path and target HANA table name before importing data.
2. Use import_csv_to_table to ingest the CSV into HANA, including datetime parsing when the user identifies date columns.
3. Preview the imported table with fetch_data when column names or data quality need confirmation.
4. If the user asks for train, test, and validation tables, choose the split mode explicitly:
  - Use time_ordered because the current toolkit scope is forecasting and chronology must be preserved.
5. Use split_table_for_forecasting to materialize train, test, and validation tables and return the generated table names and row counts.
6. Recommend the next step explicitly: data profiling, outlier detection, or forecasting.

Rules:
- Do not guess file paths, target table names, or split columns.
- Preserve chronology with time-ordered splitting.
- Always return the generated table names so later turns can reuse them precisely.

## timeseries_data_profiling

Status: recommended new standalone skill

Goal:
- Profile a time series dataset before model choice, with enough evidence to decide whether to proceed to forecasting, data cleaning, or a narrower statistical test.

When to activate:
- The user asks what is in a time series table.
- The user asks for a dataset report, data understanding, or model suggestion before training.
- The user asks whether the series is seasonal, trending, stationary, or noisy.

Required inputs:
- Table name
- Key or time column
- Target or endog column

Preferred tools:
- fetch_data
- ts_dataset_report
- ts_check
- stationarity_test
- trend_test
- seasonality_test
- white_noise_test

Workflow:
1. Confirm table, key, and target columns. Ask if any are missing.
2. Use fetch_data when a small preview helps verify assumptions.
3. Use ts_dataset_report for a broad report and ts_check for concise diagnostic output.
4. If the user asks for deeper evidence, branch to stationarity_test, trend_test, seasonality_test, or white_noise_test.
5. Summarize the characteristics that matter for later model selection, including whether the series is smooth, erratic, intermittent, or lumpy under ADI/CV^2, and whether it is seasonal, trending, or unstable.
6. Recommend the next workflow explicitly: forecasting, outlier detection, or additional data preparation.

Rules:
- Do not jump directly to training if the user only asked for inspection or model suggestion.
- Keep conclusions tied to tool outputs rather than generic forecasting advice.

## outlier_detection_and_repair_prep

Status: recommended new standalone skill

Goal:
- Detect and summarize outliers before forecasting, so the user can decide whether to clean, cap, or separately inspect problematic points or groups.

When to activate:
- The user asks about anomalies, spikes, abnormal periods, or suspicious forecast behavior.
- The user asks whether data should be cleaned before model training.
- The user asks why a forecast appears unstable or overreactive.

Required inputs:
- Table name
- Key or time column
- Target or endog column
- group_key when the scenario is multi-series

Preferred tools:
- ts_outlier_detection
- massive_ts_outlier_detection
- fetch_data

Workflow:
1. Determine whether the request is single-series or multi-series.
2. Use the matching outlier detection tool with conservative defaults unless the user specifies a method.
3. Report the returned outlier points, generated result table reference, and any useful statistics from the tool output.
4. If the user wants inspection, preview the affected rows or groups with fetch_data or an existing result table.
5. Recommend whether to continue with forecasting as-is, retrain after cleaning, or review a subset manually.

Rules:
- Do not silently modify source tables.
- Treat outlier detection as diagnostic unless the user explicitly asks for remediation steps.

## massive_forecasting

Status: recommended new standalone skill

Goal:
- Run an end-to-end grouped forecasting workflow for many related series using the massive time series tool chain.

When to activate:
- The user asks to forecast many product, store, customer, or location series.
- The request includes a group_key or refers to multiple series in one table.

Required inputs:
- Training table
- group_key
- Key or time column
- Target or endog column

Preferred tools:
- massive_ts_check
- massive_automatic_timeseries_fit_and_save
- massive_additive_model_forecast_fit_and_save
- ts_make_future_table_for_massive_forecast
- massive_automatic_timeseries_load_model_and_predict
- massive_additive_model_forecast_load_model_and_predict
- massive_automatic_timeseries_load_model_and_score
- accuracy_measure
- fetch_data

Workflow:
1. Confirm the group_key, key, and target columns.
2. Use massive_ts_check to understand differences across groups.
3. Train the grouped forecasting model with massive_automatic_timeseries_fit_and_save for AutoML-style search, or massive_additive_model_forecast_fit_and_save when the user wants additive decomposition and explicit seasonality control.
4. Create the future input table with ts_make_future_table_for_massive_forecast when needed.
5. Predict with the matching grouped forecasting tool: massive_automatic_timeseries_load_model_and_predict or massive_additive_model_forecast_load_model_and_predict.
6. If labeled holdout data exists, score with massive_automatic_timeseries_load_model_and_score or compute comparison metrics where appropriate.
7. Summarize both overall behavior and weak groups that likely need follow-up.

Rules:
- Always include group_key when referring to outputs, joins, or follow-up analysis.
- Do not collapse per-group issues into a single global conclusion when the tool output shows heterogeneity.

## model_lifecycle_and_artifacts

Status: recommended new standalone skill

Goal:
- Manage saved models and generate deployable artifacts from model storage.

When to activate:
- The user asks what models exist.
- The user asks to delete a model.
- The user asks to generate CAP or HDI artifacts for a saved model.

Required inputs:
- Model name
- Model version when a specific saved model is needed
- Project name and output directory for artifact generation

Preferred tools:
- list_models
- delete_models
- cap_artifacts
- hdi_artifacts

Workflow:
1. If the user does not know the saved model or version, use list_models first.
2. Confirm the exact model name and version before destructive or packaging actions.
3. Use delete_models only after the user clearly asks for deletion.
4. Use cap_artifacts or hdi_artifacts with explicit project and output paths.
5. Return the generated root directory and any model identifiers needed for follow-up deployment steps.

Rules:
- Never guess a model version.
- Treat deletion as destructive and only do it on explicit request.

## hana_dataframe_fallback

Status: recommended new standalone skill

Goal:
- Handle data manipulations or intermediate analyses that are awkward with the named toolkit tools but still safe within the hana-ml workflow.

When to activate:
- The user asks for a custom comparison table, transformation, or metric not directly covered by existing forecasting tools.
- Existing dedicated tools fail to express the needed transformation.

Required inputs:
- Enough detail to write the SQL or restricted Python snippet safely

Preferred tools:
- python_hanaml_exec
- SelectStatement_to_table
- fetch_data

Workflow:
1. Prefer SelectStatement_to_table when the task is naturally expressible in SQL and should materialize a table.
2. Use python_hanaml_exec for restricted hana-ml DataFrame logic that is cumbersome in pure SQL.
3. Preview outputs with fetch_data when the result needs inspection.
4. Summarize what was materialized or computed, including table names and key expressions used.

Rules:
- Prefer dedicated forecasting tools first when they already solve the task.
- Keep fallback steps reproducible by stating the exact generated table names or result variables.

## Recommended Standalone Skill Set

The current built-in set is directionally correct, but it is narrower than the real toolkit surface. A standalone skills file should cover the following skills.

| Skill | Status | Why it should exist | Primary tools |
|---|---|---|---|
| data_ingestion_and_dataset_preparation | new | The current notebook still uses hand-written Python for CSV import and dataset splitting, so a dedicated conversational preparation workflow would remove the last manual setup step. | import_csv_to_table, split_table_for_forecasting, fetch_data, ts_dataset_report |
| timeseries_data_profiling | new | The notebook starts with report and statistical checks before model choice. This is a stable workflow of its own. | ts_dataset_report, ts_check, stationarity_test, trend_test, seasonality_test, white_noise_test, fetch_data |
| timeseries_forecasting | extracted | Core single-series forecast workflow already exists and should be preserved. | ts_dataset_report, ts_check, automatic_timeseries_fit_and_save, additive_model_forecast_fit_and_save, intermittent_forecast, ts_make_future_table, automatic_timeseries_load_model_and_predict |
| prediction_result_analysis | extracted | Already implemented and aligned with the notebook insight step. | accuracy_measure, SelectStatement_to_table, fetch_data, forecast_line_plot |
| outlier_detection_and_repair_prep | new | The toolkit has strong outlier tooling, but current skills do not tell the agent when to use it before modeling. | ts_outlier_detection, massive_ts_outlier_detection, fetch_data |
| massive_forecasting | new | Current built-in massive skill is only comparison-oriented, not fully end-to-end. | massive_ts_check, massive_automatic_timeseries_fit_and_save, ts_make_future_table_for_massive_forecast, massive_automatic_timeseries_load_model_and_predict, massive_automatic_timeseries_load_model_and_score |
| model_lifecycle_and_artifacts | new | The toolkit supports listing, deleting, and packaging models into CAP and HDI artifacts. | list_models, delete_models, cap_artifacts, hdi_artifacts |
| hana_dataframe_fallback | new | The toolkit exposes Python and SQL materialization fallbacks for transformations not covered by a dedicated tool. | python_hanaml_exec, SelectStatement_to_table, fetch_data |

## Gaps Between Current Implementation and Real Toolkit Capability

## Missing skill coverage

- No dedicated data_ingestion_and_dataset_preparation skill.
  The notebook still imports CSV data and creates train and prediction tables through hand-written Python before the agent workflow begins.

- No dedicated timeseries_data_profiling skill.
  The notebook explicitly performs ts_dataset_report and ts_check before training, but current built-ins fold this only implicitly into timeseries_forecasting.

- No dedicated outlier_detection_and_repair_prep skill.
  The toolkit exposes ts_outlier_detection and massive_ts_outlier_detection, but the current skills never direct the agent to use them before model training.

- No end-to-end massive_forecasting skill.
  The current built-in massive_forecast_comparison focuses on evaluation after the fact. It does not fully capture the train and predict path using group_key and ts_make_future_table_for_massive_forecast.

- No model_lifecycle_and_artifacts skill.
  The notebook includes CAP artifact generation, and the toolkit also supports HDI generation plus model listing and deletion, but no current skill covers this lifecycle.

- No hana_dataframe_fallback skill.
  python_hanaml_exec and SelectStatement_to_table are important escape hatches when existing tools do not cover a specific transformation or comparison need.

## Weak spots in the current prediction_result_analysis skill

- It does not explicitly mention forecast_line_plot as an optional follow-up even though the notebook uses plotting before asking for insights.
- It does not mention score-based analysis using automatic_timeseries_load_model_and_score or massive_automatic_timeseries_load_model_and_score.
- It does not call out that accuracy_measure requires exact column alignment and supported metric names.
- It does not define how to recover when the user asks for insights but the prediction table name is only present in chat history.
- It does not instruct the agent to preserve and reuse exact returned prediction table names across later turns.

## Weak spots in the current timeseries_forecasting skill

- It does not separate profiling from forecasting, which makes selector behavior less precise.
- It does not explicitly route ADI/CV^2-classified intermittent or lumpy demand cases to intermittent_forecast strongly enough.
- It does not include automatic_timeseries_load_model_and_score as the preferred scoring path when a labeled test table exists.
- It does not mention outlier detection as a pre-modeling branch.

## Weak spots in the current massive_forecast_comparison skill

- The name is too narrow for the available toolkit surface. The toolkit supports full multi-series train, predict, score, and outlier workflows, not just comparison.
- It omits ts_make_future_table_for_massive_forecast even though that is a common prerequisite.
- It does not provide a clear branch for per-group scoring versus global comparison.

## Notebook Alignment Notes

The scenario in nutest/testscripts/demo/e2e_scenarios/context_agent.ipynb follows this order:

1. Import the CSV file and create the working HANA table.
2. Split the dataset into training and prediction-oriented tables.
3. Fetch and inspect data.
4. Create a dataset report.
5. Run ts_check and suggest a model.
6. Train and save the model.
7. Predict with the trained model.
8. Plot predicted versus actual.
9. Ask for insights from the predicted result.
10. Generate CAP artifacts.

This confirms that a standalone skills file should not start only after tables already exist. It should cover ingestion and dataset preparation, then continue through forecasting, result interpretation, and artifact generation as first-class workflows.

## Suggested Migration Path

1. Keep the current three built-in skills for backward compatibility.
2. Add a parser or loader that can read this standalone markdown draft into Skill objects.
3. Split selector labels so ingestion, profiling, forecasting, result analysis, massive forecasting, and artifact generation can be activated independently.
4. Add dedicated tools for CSV import and modeling-dataset splitting so the agent does not need to fall back to hand-written Python for the first mile of the workflow.
5. Expand prediction_result_analysis to include score-based analysis, plotting follow-up, and explicit handling of prior-turn table references.