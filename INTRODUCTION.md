# Project description

## Introduction

Welcome to the __generative AI toolkit for SAP HANA Cloud (hana_ai)__

This project provides a toolkit of hana-ml-oriented tools and a file-based ContextAgent for conversational SAP HANA Cloud forecasting and machine learning workflows. It focuses on reusable HANAMLToolkit integrations, workflow skills for data preparation and forecasting tasks, and tool-driven execution against SAP HANA Cloud machine learning functions.

## Overview
The generative AI toolkit for SAP HANA Cloud provides the following key capabilities:
* a file-based ContextAgent with Markdown memory, runtime skill routing, and command-style memory/skill controls for iterative forecasting workflows
* a library of prepared tools to streamline use of SAP HANA machine learning functions and aid in e.g. forecast algorithm selection with the given data
* tools for leveraging the SAP HANA Cloud vectorstore and embedding services

# Capabilities introduction

## ContextAgent with Markdown memory and skills
The file-based ContextAgent (agents.context_agent.ContextAgent class) is designed for tool-enabled, multi-turn workflows without requiring HANA-backed memory services. It persists conversation state to Markdown files, retrieves compact notes into context, and activates focused workflow skills depending on the request.

```python
from hana_ai.tools.toolkit import HANAMLToolkit
from hana_ai.agents.context_agent import ContextAgent

tools = HANAMLToolkit(cc, used_tools='all').get_tools()
agent = ContextAgent(llm=llm, tools=tools, storage_dir=".context_agent")

response = agent.chat("Import this csv, split it into train, test, and validation tables, then recommend a forecasting model.")
```

The ContextAgent currently supports the following workflow skills:

| Skill Name | Purpose |
|-----------|---------|
| data_ingestion_and_dataset_preparation | Import CSV data into HANA and create time-ordered train, test, and validation tables. |
| timeseries_data_profiling | Inspect a time series dataset with reports and statistical checks before modeling. |
| timeseries_forecasting | Train, predict, score, and summarize a single-series forecasting workflow. |
| prediction_result_analysis | Compare predicted results with actuals and summarize forecast quality. |
| outlier_detection_and_repair_prep | Detect anomalous points or groups before modeling. |
| massive_forecasting | Run grouped forecasting across many related series with group_key-aware tools. |
| model_lifecycle_and_artifacts | List, delete, and package saved models as CAP or HDI artifacts. |
| hana_dataframe_fallback | Materialize SQL or restricted Python fallback logic when dedicated tools are insufficient. |

The ContextAgent also exposes command-style controls during `chat()`:

- Memory commands: `!clear_notes`, `!clear_session`, `!reset_memory`, `!clear_notes_file`, `!clear_todo`, `!clear_decisions`, `!clear_context`, `!clear_chat`, `!clear_summary`
- Skill commands: `!list_skills`, `!active_skills`, `!skills_on`, `!skills_off`, `!enable_skill <skill_name>`, `!disable_skill <skill_name>`

## Library of tools for HANA-ML
Provided AI-tools for streamlining usage of HANA ML functions in context of the conversational agent.
| Tool Name | Description | Comment |
|-----------|-------------|---------|
| additive_model_forecast_fit_and_save | To fit an AutomaticTimeseries model and save it in the model storage. |
| additive_model_forecast_load_model_and_predict | To load an AutomaticTimeseries model and predict the future values. |
| automatic_timeseries_fit_and_save | To fit an AutomaticTimeseries model and save it in the model storage. |
| automatic_timeseries_load_model_and_predict | To load an AutomaticTimeseries model and predict the future values. |
| automatic_timeseries_load_model_and_score | To load an AutomaticTimeseries model and score the model. |
| accuracy_measure | To compute the accuracy measure using true and predict tables. |
| cap_artifacts | To generate CAP artifacts from the model in the model storage. |
| delete_models | To delete the model from the model storage. |
| fetch_data | To fetch the data from the HANA database.|
| forecast_line_plot | To generate line plot for the forecasted result. |
| hdi_artifacts | To generate HDI artifacts for a given model from model storage. |
| import_csv_to_table | To import a local CSV file into a HANA table with optional datetime parsing. | since 1.1.26040800 |
| intermittent_forecast | To forecast the intermittent time series data. |
| list_models | To list the models in the model storage. |
| python_hanaml_exec | To run restricted hana-ml Python logic for transformations or fallback analyses. |
| seasonality_test | To check the seasonality of the time series data. |
| SelectStatement_to_table | To execute a SELECT SQL statement and store the result in a new table. | since 1.0.250909 |
| split_table_for_forecasting | To create train, test, and validation tables from an existing HANA table using time-ordered splitting for forecasting workflows. | since 1.1.26040800 |
| stationarity_test | To check the stationarity of the time series data. |
| trend_test | To check the trend of the time series data. |
| ts_check | To check the time series data for stationarity, intermittent, trend and seasonality. |
| ts_dataset_report | To generate a report for the time series data. |
| ts_outlier_detection | To detect the outliers in the time series data. |
| ts_make_future_table | To generate a future table for time series forecasting. | since 1.0.250909 |
| white_noise_test | To check the white noise of the time series data. |

## Newly added tools for massive time series processing and forecasting, since 1.0.250930
| Tool Name | Description | Comment |
|-----------|-------------|---------|
| massive_automatic_timeseries_fit_and_save | To fit multiple AutomaticTimeseries models and save them in the model storage. |
| massive_automatic_timeseries_load_model_and_predict | To load multiple AutomaticTimeseries models and predict the future values. |
| massive_automatic_timeseries_load_model_and_score | To load multiple AutomaticTimeseries models and score the models. |
| massive_additive_model_forecast_fit_and_save | To fit grouped additive forecasting models and save them in model storage. |
| massive_additive_model_forecast_load_model_and_predict | To load grouped additive forecasting models and predict future values. |
| massive_ts_outlier_detection | To detect the outliers in multiple time series data. |
| ts_make_future_table_for_massive_forecast | To generate a future table for multiple time series forecasting. |
| massive_ts_check | To check multiple time series data for stationarity, intermittent, trend and seasonality. |

## Vector engine and Embedding generation tools
Different Embedding functions can be used ...
### Embedding some code examples
```python
from hana_ai.vectorstore.embedding_service import PALModelEmbeddings
model = PALModelEmbeddings(cc)
model(['hello', 'world'])

from hana_ai.vectorstore.embedding_service import HANAVectorEmbeddings

model = HANAVectorEmbeddings(cc)
model(['hello', 'world'])
```

```python
from hana_ai.vectorstore.embedding_service import GenAIHubEmbeddings
embedding_func = GenAIHubEmbeddings()
embedding_func('hello')
```

