[![REUSE status](https://api.reuse.software/badge/github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud)](https://api.reuse.software/info/github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud)

# Generative AI Toolkit for SAP HANA Cloud

## About this project

Generative AI Client for SAP HANA Cloud is an extension of the existing HANA ML Python client library, mainly focusing on GenAI and related use cases. It includes many leading-edge GenAI related open source libraries and provides seamless integration with HANA ML, HANA vector engine, and other SAP GenAI Hub SDK, see our [Introduction](https://github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud/blob/main/INTRODUCTION.md), [Notebook](https://github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud/blob/main/nutest/testscripts/demo/e2e_scenarios/context_agent.ipynb) and [Documentation](https://sap.github.io/generative-ai-toolkit-for-sap-hana-cloud/).

## Requirements and Setup

The prerequisites for using the Generative AI Toolkit for SAP HANA Cloud are listed at [Prerequisites](https://sap.github.io/generative-ai-toolkit-for-sap-hana-cloud/hana_ai.html#prerequisites).

The Generative AI Toolkit for SAP HANA Cloud is available as a Python package. You can install it via `pip`:

```bash
pip install hana-ai
```

## ContextAgent

The toolkit includes a file-based ContextAgent for conversational forecasting and data-preparation workflows with Markdown-backed memory, tool calling, and runtime-selectable skills.

Core ContextAgent skills include:

- `data_ingestion_and_dataset_preparation` for CSV import and time-ordered train, test, and validation table creation
- `timeseries_data_profiling` for dataset reports and statistical checks before model selection
- `timeseries_forecasting` for single-series train, predict, score, and plot workflows
- `prediction_result_analysis` for predicted-versus-actual comparison and quality analysis
- `outlier_detection_and_repair_prep` for anomaly inspection before model training
- `massive_forecasting` for grouped forecasting across many related series
- `model_lifecycle_and_artifacts` for listing, deleting, and packaging saved models
- `hana_dataframe_fallback` for SQL and restricted Python fallback transformations

## Tools

The toolkit exposes HANAML-oriented tools for data preparation, profiling, forecasting, evaluation, artifact generation, and grouped forecasting workflows.

Core forecasting and analysis tools include:

- `import_csv_to_table`, `split_table_for_forecasting`, `fetch_data`
- `ts_dataset_report`, `ts_check`, `stationarity_test`, `trend_test`, `seasonality_test`, `white_noise_test`
- `automatic_timeseries_fit_and_save`, `automatic_timeseries_load_model_and_predict`, `automatic_timeseries_load_model_and_score`
- `additive_model_forecast_fit_and_save`, `additive_model_forecast_load_model_and_predict`, `intermittent_forecast`
- `ts_outlier_detection`, `ts_make_future_table`, `forecast_line_plot`, `accuracy_measure`
- `list_models`, `delete_models`, `cap_artifacts`, `hdi_artifacts`
- `SelectStatement_to_table`, `python_hanaml_exec`

Grouped and massive forecasting tools include:

- `massive_ts_check`, `massive_ts_outlier_detection`
- `massive_automatic_timeseries_fit_and_save`, `massive_automatic_timeseries_load_model_and_predict`, `massive_automatic_timeseries_load_model_and_score`
- `massive_additive_model_forecast_fit_and_save`, `massive_additive_model_forecast_load_model_and_predict`
- `ts_make_future_table_for_massive_forecast`

See the ContextAgent notebook for an end-to-end example at [nutest/testscripts/demo/e2e_scenarios/context_agent.ipynb](https://github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud/blob/main/nutest/testscripts/demo/e2e_scenarios/context_agent.ipynb).

## Support, Feedback, Contributing

This project is open to feature requests/suggestions, bug reports etc. via [GitHub issues](https://github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud/issues). Contribution and feedback are encouraged and always welcome. For more information about how to contribute, the project structure, as well as additional contribution information, see our [Contribution Guidelines](https://github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud/blob/main/CONTRIBUTING.md).

## Security / Disclosure
If you find any bug that may be a security problem, please follow our instructions at [in our security policy](https://github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud/security/policy) on how to report it. Please do not create GitHub issues for security-related doubts or problems.

## Code of Conduct

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone. By participating in this project, you agree to abide by its [Code of Conduct](https://github.com/SAP/.github/blob/main/CODE_OF_CONDUCT.md) at all times.

## Licensing

Copyright 2026 SAP SE or an SAP affiliate company and generative-ai-toolkit-for-sap-hana-cloud contributors. Please see our [LICENSE](https://github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud/blob/main/LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is available [via the REUSE tool](https://api.reuse.software/info/github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud).
