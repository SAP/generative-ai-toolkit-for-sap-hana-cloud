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
   hana_ml_tools.automatic_timeseries_tools.AutomaticTimeSeriesFitAndSave
   hana_ml_tools.automatic_timeseries_tools.AutomaticTimeseriesLoadModelAndPredict
   hana_ml_tools.automatic_timeseries_tools.AutomaticTimeseriesLoadModelAndScore
   hana_ml_tools.cap_artifacts_tools.CAPArtifactsTool
   hana_ml_tools.intermittent_forecast_tools.IntermittentForecast
   hana_ml_tools.ts_check_tools.TimeSeriesCheck
   hana_ml_tools.ts_check_tools.StationarityTest
   hana_ml_tools.ts_check_tools.TrendTest
   hana_ml_tools.ts_check_tools.SeasonalityTest
   hana_ml_tools.ts_check_tools.WhiteNoiseTest
   hana_ml_tools.ts_outlier_detection_tools.TSOutlierDetection
   hana_ml_tools.ts_visualizer_tools.TimeSeriesDatasetReport
   hana_ml_tools.ts_visualizer_tools.ForecastLinePlot
