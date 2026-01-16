Changelog
=========

**Version 1.0.260116**

``New Functions``
    - Added DiscoveryAgentTool and DataAgentTool to hanaml tools for knowledge graph interactions.

``Enhancements``
    - Added reset_tools method to HANAMLToolkit to reset the toolkit's tools.
    - Updated launch_mcp_server function to support http protocol.

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
    - Added `launch_mcp_server` function to hanaml toolkit.
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
