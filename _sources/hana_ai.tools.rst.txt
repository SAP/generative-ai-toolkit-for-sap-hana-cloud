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

.. _hana_ml_tools-label:

hana_ml_tools
-------------
.. autosummary::
   :toctree: tools/
   :template: class.rst

   hana_ml_tools.additive_model_forecast_tools.AdditiveModelForecastFitAndSave
   hana_ml_tools.additive_model_forecast_tools.AdditiveModelForecastLoadModelAndPredict
   hana_ml_tools.automatic_timeseries_tools.AutomaticTimeSeriesFitAndSave
   hana_ml_tools.automatic_timeseries_tools.AutomaticTimeSeriesLoadModelAndPredict
   hana_ml_tools.automatic_timeseries_tools.AutomaticTimeSeriesLoadModelAndScore
   hana_ml_tools.cap_artifacts_tools.CAPArtifactsTool
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
