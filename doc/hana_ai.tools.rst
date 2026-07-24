hana_ai.tools
=============

hana.ai tools is a set of tools that can be used to perform various tasks like forecasting, time series analysis, etc.

.. automodule:: hana_ai.tools
   :no-members:
   :no-inherited-members:

.. _agent_as_a_tool-label:

agent_as_a_tool
---------------
.. autosummary::
   :toctree: tools/
   :template: class.rst

   agent_as_a_tool.AgentAsATool

.. _code_template_tools-label:

code_template_tools
-------------------
.. autosummary::
   :toctree: tools/
   :template: class.rst

   code_template_tools.GetCodeTemplateFromVectorDB

.. _hana_ml_tools-label:

hana_ml_tools
-------------
.. autosummary::
   :toctree: tools/
   :template: class.rst

   hana_ml_tools.additive_model_forecast_tools.AdditiveModelForecastFitAndSave
   hana_ml_tools.additive_model_forecast_tools.AdditiveModelForecastLoadModelAndPredict
   hana_ml_tools.additive_model_forecast_tools.MassiveAdditiveModelForecastFitAndSave
   hana_ml_tools.additive_model_forecast_tools.MassiveAdditiveModelForecastLoadModelAndPredict
   hana_ml_tools.automatic_timeseries_tools.AutomaticTimeSeriesFitAndSave
   hana_ml_tools.automatic_timeseries_tools.AutomaticTimeSeriesLoadModelAndPredict
   hana_ml_tools.automatic_timeseries_tools.AutomaticTimeSeriesLoadModelAndScore
   hana_ml_tools.cap_artifacts_tools.CAPArtifactsTool
   hana_ml_tools.config_dict_validator_tools.GetPALPipelineInfo
   hana_ml_tools.config_dict_validator_tools.GetAutoMLConfigDict
   hana_ml_tools.config_dict_validator_tools.ModifyAutoMLConfigDict
   hana_ml_tools.dataset_prep_tools.ImportCSVToTableTool
   hana_ml_tools.dataset_prep_tools.SplitTableForForecastingTool
   hana_ml_tools.fetch_tools.FetchDataTool
   hana_ml_tools.hdi_artifacts_tools.HDIArtifactsTool
   hana_ml_tools.intermittent_forecast_tools.IntermittentForecast
   hana_ml_tools.model_storage_tools.ListModels
   hana_ml_tools.model_storage_tools.DeleteModels
   hana_ml_tools.model_storage_tools.DisplayConfigDict
   hana_ml_tools.select_statement_to_table_tools.SelectStatementToTableTool
   hana_ml_tools.ts_accuracy_measure_tools.AccuracyMeasure
   hana_ml_tools.ts_check_tools.TimeSeriesCheck
   hana_ml_tools.ts_check_tools.StationarityTest
   hana_ml_tools.ts_check_tools.TrendTest
   hana_ml_tools.ts_check_tools.SeasonalityTest
   hana_ml_tools.ts_check_tools.WhiteNoiseTest
   hana_ml_tools.ts_make_predict_table.TSMakeFutureTableTool
   hana_ml_tools.ts_outlier_detection_tools.TSOutlierDetection
   hana_ml_tools.ts_visualizer_tools.TimeSeriesDatasetReport
   hana_ml_tools.ts_visualizer_tools.ForecastLinePlot

.. _graph_tools-label:

graph_tools
-----------

Knowledge-graph-backed retrieval tools exposed on the MCP server. ``ObjectDiscoveryTool`` calls the HANA AI Core object-discovery procedure (default ``AI_OBJECT_RETRIEVAL``) to surface schemas, tables, columns, and their relationships as narrative context. ``DataRetrievalTool`` calls the paired data-retrieval procedure to fetch rows or aggregations for a natural-language question. Both tools share the ``hana_ai.retrieval`` clients and expect a HANA remote source connected to AI Core.

.. autosummary::
   :toctree: tools/
   :template: class.rst

   hana_ml_tools.graph_tools.ObjectDiscoveryTool
   hana_ml_tools.graph_tools.DataRetrievalTool

.. _df_tools-label:

df_tools
-------------
.. autosummary::
   :toctree: tools/
   :template: class.rst

   df_tools.automatic_timeseries_tools.AutomaticTimeSeriesFitAndSave
   df_tools.automatic_timeseries_tools.AutomaticTimeSeriesLoadModelAndPredict
   df_tools.automatic_timeseries_tools.AutomaticTimeSeriesLoadModelAndScore
   df_tools.fetch_tools.FetchDataTool
   df_tools.ts_outlier_detection_tools.TSOutlierDetection
   df_tools.ts_visualizer_tools.TimeSeriesDatasetReport

.. _hana_ml_toolkit-label:

hana_ml_toolkit
---------------
.. autosummary::
   :toctree: tools/
   :template: class.rst

   toolkit.HANAMLToolkit

.. _hana_ml_tools_utility-label:

hana_ml_tools.utility
---------------------

Module-level helpers for MCP server/client bootstrap, audit inspection, and
the HANA-side ``APPLICATIONSOURCE`` pack that carries MCP identity into
``M_SQL_PLAN_CACHE`` / ``AUDIT_LOG``. See :doc:`mcp_audit` for the audit
story end-to-end.

.. currentmodule:: hana_ai.tools.hana_ml_tools.utility

.. autofunction:: build_context_agent_mcp_tools
.. autofunction:: ensure_mcp_audit_log
.. autofunction:: fetch_mcp_audit_rows
.. autofunction:: fetch_hana_session_context
.. autofunction:: fetch_hana_appsource_pack
.. autofunction:: fetch_hana_mcp_audit_view
.. autofunction:: build_appsource_pack
.. autofunction:: parse_appsource_pack
.. autofunction:: find_free_port
.. autodata::     DEFAULT_MCP_SESSION_CONTEXT_KEYS
.. autodata::     APPLICATIONSOURCE_MAX_BYTES
.. autodata::     APPLICATIONSOURCE_FIELD_ORDER
.. autodata::     HANA_MCP_AUDIT_VIEW_COLUMNS
.. autodata::     MCP_BEACON_SQL_MARKER

.. toctree::
   :maxdepth: 1
   :caption: MCP auditing

   mcp_audit
