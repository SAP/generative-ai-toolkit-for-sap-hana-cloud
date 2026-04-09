Changelog
=========

**Version 1.1.26040800**

``New Functions``
    - Added a standalone ContextAgent skills catalog at hana_ai.iagents.skills.md to make skill definitions easier to review, extend, and maintain.
    - Added markdown-based skill loading in ContextAgent with built-in fallback support for existing deployments.
    - Added dedicated dataset preparation tools for CSV import and time-ordered train, test, validation table generation in conversational forecasting workflows, including split_table_for_forecasting.
    - Added grouped additive forecasting tools backed by AdditiveModelForecast(massive=True) for save-and-predict workflows across many related series.

``Enhancements``
    - Added aggregate and file-level ContextAgent memory reset commands for clearing persisted notes, TODOs, decisions, context, chat history, and session summaries without deleting the storage directory.
    - Expanded ContextAgent skill coverage for dataset preparation, time-series profiling, prediction-result analysis, grouped forecasting, artifact generation, and dataframe-oriented fallback workflows.
    - Improved fallback skill selection so profiling, post-prediction analysis, grouped forecasting, artifact generation, and SQL comparison requests map more reliably to the intended skills.
    - Added validation for markdown-defined skills so malformed entries are ignored unless they provide a valid Goal section.
    - Clarified that dataset splitting currently targets forecasting scenarios and preserves chronology through time-ordered splitting.
    - Improved forecasting prediction robustness so automatic, additive, and grouped prediction tools proactively validate and repair predict inputs to inference-only columns when users accidentally pass labeled holdout tables.
    - Improved ContextAgent tool-failure handling with more actionable diagnostics for predict-versus-score input mismatches and backend PAL runtime failures during forecasting workflows.

``API Changes``
    - ContextAgent now prefers loading skills from the colocated markdown catalog and merges them with the built-in fallback skills.
    - ContextAgent is the recommended target for future live end-to-end scenario expansion; HANAMLRAGAgent and Mem0HANARAGAgent are no longer recommended for new scenario coverage.

**Version 1.0.260331**

``New Functions``
    - Added prediction result analysis skills and dataframe manipulation skills in the context agent to provide better insights and analysis of the predicted results.

**Version 1.0.260330**

``New Functions``
    - Added context agent with markdown type memory to support better context management and display in the conversation.

**Version 1.0.260319**

``Bug Fixes``
    - Fixed the hullucination issue of forecast tools in Mem0HANARAGAgent when the input table name is very similar to the prediction result.

**Version 1.0.260316**

``API Changes``
    - Removed the graph_tools due to the stored procedure missing.

**Version 1.0.260212**

``Bug Fixes``
    - Fixed the version incomatibility issue of langchain and sap-ai-sdk-gen.

**Version 1.0.260116**

``Enhancements``
    - Added reset_tools method to HANAMLToolkit to reset the toolkit's tools.

**Version 1.0.251223**

``Bug Fixes``
    - Fixed the import issues in hana_ai agents and tools.
    - Fixed the missing parameter in PALEmbeddings.

**Version 1.0.251217**

``Enhancements``
    - Added the customized procedure name in CAP generation tool.

**Version 1.0.251001**

``Bug Fixes``
    - Fixed the output parser issue in SmartDataFrame's transform function.

**Version 1.0.250930**

``Enhancements``
    - Added MassiveAutomaticTimeSeriesFitAndSave, MassiveAutomaticTimeSeriesLoadModelAndPredict, and MassiveAutomaticTimeSeriesLoadModelAndScore tools to support massive time series model training, prediction, and scoring with group_key parameter.
    - Added MassiveTSOutlierDetection tool to support massive time series outlier detection with group_key parameter.
    - Added TSMakeFutureTableForMassiveForecastTool to create future tables for massive time series forecasting with group_key parameter.
    - Added MassiveTimeSeriesCheck tool to perform time series analysis and generate reports for multiple time series with group_key parameter.
    - Updated HANAMLToolkit to include the new massive time series tools.

``API Changes``
    - Modified SelectStatementToTableTool to include a 'force' parameter that allows overwriting existing tables.
    - Changed "Timeseries" to "TimeSeries" in class names for consistency.

``Bug Fixes``
    - Fixed an issue for text with special characters in the HANAVectorEmbeddings class.

**Version 1.0.250923**

``Enhancements``
    - Enhanced the outputs of tools when select_statement is too large by creating temporary tables with unique names.
    - Added additive_model_forecast_tools and intermittent_forecast df tools to the default tools in SmartDataFrame.

**Version 1.0.250918**

``Enhancements``
    - Added HANA table schema support for tools.
    - Improved output information for outlier detections.
    - Removed the hana_connection_context parameter from the `HANAMLRAGAgent` class and infer it from the tools.
    - Refine the default value of `max_iterations` parameter of `HANAMLRAGAgent` class parameters from 10 to 20.
    - Change the default value of `vector_store_type` parameter of `HANAMLRAGAgent` class from "faiss" to "hanadb".
    - Change the default value of `long_term_db` parameter of `HANAMLRAGAgent` class from sqlite to HANA DB.
    - Added the `embedding_service` parameter to the `HANAMLRAGAgent` class to allow users to pass their own embedding service. The default embedding service has been changed from `GenAIHubEmbeddings` to `HANAVectorEmbeddings`.
    - Added `PAL CrossEncoder` as the default cross-encoder model for reranking in the `HANAMLRAGAgent` class. If it is not available, it will fall back to `sentence-transformers/all-MiniLM-L6-v2`.
    - Added `session_id` parameter to the `HANAMLRAGAgent` class to support multiple sessions in long-term memory. By default, it is set to "global_session".
    - Removed the restriction to save memory into long term memory when the result is pandas data or large data. Now, all the results will be saved into long term memory with chunking and embeddings.
    - Deprecated the code template tool and python REPL tool in `SmartDataFrame` class. Users can use the tools from `df_tools` as default tools instead.

**Version 1.0.250909**

``New Functions``
    - Added `TSMakeFutureTableTool` to create a future table for time series forecasting.
    - Added `SelectStatementToTableTool` to execute a SELECT SQL statement and store the result in a new table.

**Version 1.0.250904**

``Bug Fixes``
    - Fixed the issue of calling code template tool.

**Version 1.0.250707**

``Enhancements``
    - Added `vector_store_type` parameter to `HANAMLRAGAgent` class to support different vector store types, including "hanadb" and "faiss".
    - Improved the `HANAMLRAGAgent` class to handle vector store initialization and updates more efficiently.

``Bug Fixes``
    - Fixed the parameter issues in `HANAMLRAGAgent` class by adding rerank_candidates and rerank_k parameters.

**Version 1.0.250702**

``New Functions``
    - Added `HANAMLRagAgent` class to enable Retrieval-Augmented Generation (RAG) capabilities, leveraging a hybrid short-term and long-term memory architecture with CrossEncoder reranking techniques.

**Version 1.0.250630**

``New Functions``
    - Added hdi artifacts tool.

**Version 1.0.250617**

``New Functions``
    - Added model deletion tool and chat history deletion tool to hanaml Agent.

**Version 1.0.250530**

``Enhancements``
    - Added unsupported tools check (classfication, regression).
    - BAS integration enhancements.

**Version 1.0.250520**

``Enhancements``
    - Added set_return_direct function in hanaml Agent.

``Bug Fixes``
    - Fixed the prompt in hanaml Agent to enable tool cal.
    - Fixed CAP generation temporary location in MacOS.

**Version 1.0.250509**

``Enhancements``
    - Output the inspect code for BAS integration.

**Version 1.0.250506**

``Enhancements``
    - Enhanced the seasonality detection for additive_model_forecast_tools.
    - Provide table meta information and supported algorithms in ts_check tool.

**Version 1.0.250424**

``Enhancements``
    - Added input table check and columns check to avoid stopping the reasoning.
    - Added samples to ts_check tool.

``Bug Fixes``
    - Fixed wrong report filename issue. (hana-ml appends _report.html to the file.)

**Version 1.0.250411**

``Enhancements``
    - Save observations to chat history in HANA ML agent. Added max_observations parameter to control the number of observations saved in the chat history.
    - Adjust the default value of fetch_data tool to return pandas indirectly to avoid chain stopping due to the tool call of fetch_data in the intermediate step.

**Version 1.0.250410**

``Enhancements``
    - Enhanced the HANA SQL agent to support case-sensitive SQL queries.
    - Added create_hana_sql_toolkit function to create a toolkit for HANA SQL.
    - Optimized the chat history management in HANA ML agent.

``Bug Fixes``
    - Fixed the accuracy_measure tool issue when evaluation_metric="spec".

**Version 1.0.250407**

``Enhancements``
    - Improved `forecast_line_plot` tool to automatically detect the confidence if it is not provided.
    - Serialized the tool's return if it is pandas DataFrame when `return_direct` is set to `False`.

``Bug Fixes``
    - Fixed the json serialization issue when the tool's return contains Timestamp.

**Version 1.0.250403**

``New Functions``
    - Added `list_models` tool to list all trained models in the model storage.
    - Added `accuracy_measure` tool to measure the accuracy of a model on a test dataset for time series forecasting.

``Enhancements``
    - Improved the `intermittent_forecast` tool to use CrostonTSB instead.
    - Added parameter `return_direct` to all tools and toolkit.
    - Improved the `fetch_data` tool to return a pandas DataFrame instead of a list of dictionaries. By default, the tool parameter `return_direct` is set to `True`, which means the tool will return a pandas DataFrame.
