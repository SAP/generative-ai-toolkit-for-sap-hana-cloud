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
   hana_ml_tools.dataset_prep_tools.ImportCSVToTableTool
   hana_ml_tools.dataset_prep_tools.SplitTableForForecastingTool
   hana_ml_tools.fetch_tools.FetchDataTool
   hana_ml_tools.hdi_artifacts_tools.HDIArtifactsTool
   hana_ml_tools.intermittent_forecast_tools.IntermittentForecast
   hana_ml_tools.model_storage_tools.ListModels
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
