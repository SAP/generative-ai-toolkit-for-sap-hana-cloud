"""
This module is used to generate a report for the time series dataset.

The following classes are available:

    * :class `TimeSeriesDatasetReport`
    * :class `ForecastLinePlot`
"""

import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Optional, Type
from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.visualizers.visualizer_base import forecast_line_plot
from hana_ml.visualizers.unified_report import UnifiedReport

logger = logging.getLogger(__name__)

class TSDatasetInput(BaseModel):
    """
    The input schema for the TSDatasetTool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    output_dir: Optional[str] = Field(description="the output directory to save the report, it is optional", default=None)

class ForecastLinePlotInput(BaseModel):
    """
    The input schema for the ForecastLinePlot tool.
    """
    predict_table_name: str = Field(description="the name of the predicted result table. If not provided, ask the user. Do not guess.")
    actual_table_name: Optional[str] = Field(description="the name of the actual data table, it is optional", default=None)
    confidence: Optional[tuple] = Field(description="the column names of confidence bounds, it is optional", default=None)
    output_dir: Optional[str] = Field(description="the output directory to save the line plot, it is optional", default=None)

class TimeSeriesDatasetReport(BaseTool):
    """
    This tool generates a report for a time series dataset.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The path of the generated report.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - table_name
                  - the name of the table. If not provided, ask the user. Do not guess.
                * - key
                  - the key of the dataset. If not provided, ask the user. Do not guess.
                * - endog
                  - the endog of the dataset. If not provided, ask the user. Do not guess
    """
    name: str = "ts_dataset_report"
    """Name of the tool."""
    description: str = "To generate timeseries report for a given HANA table. "
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = TSDatasetInput
    return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context
        )

    def _run(
        self, table_name: str, key: str, endog: str, output_dir: Optional[str]=None, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        df = self.connection_context.table(table_name).select(key, endog)
        ur = UnifiedReport(df).build(key=key, endog=endog)
        if output_dir is None:
            destination_dir = os.path.join(tempfile.gettempdir(), "hanaml_report")
        else:
            destination_dir = output_dir
        if not os.path.exists(destination_dir):
            try:
                os.makedirs(destination_dir, exist_ok=True)
            except Exception as e:
                logger.error("Error creating directory %s: %s", destination_dir, e)
                raise

        output_file = os.path.join(
                    destination_dir,
                    f"{table_name}_ts_report.html",
                )
        ur.display(save_html=output_file)
        ur.display() #directly display in jupyter
        return json.dumps({"html_file": output_file})

    async def _arun(
        self, table_name: str, key: str, endog: str, output_dir, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(table_name, key, endog, output_dir, run_manager=run_manager)

class ForecastLinePlot(BaseTool):
    """
    This tool generates a line plot for the forecasted result.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The path of the generated line plot.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - predict_table_name
                  - the name of the predicted result table. If not provided, ask the user. Do not guess.
                * - actual_table_name
                  - the name of the actual data table, it is optional
                * - confidence
                  - the column names of confidence bounds, it is optional
    """
    name: str = "forecast_line_plot"
    """Name of the tool."""
    description: str = "To generate line plot for the forecasted result. "
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = ForecastLinePlotInput
    """Input schema of the tool."""
    return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context
        )

    def _run(
        self, predict_table_name: str, actual_table_name: Optional[str]=None, confidence: Optional[tuple]=None, output_dir: Optional[str]=None, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        if actual_table_name is None:
            fig = forecast_line_plot(self.connection_context.table(predict_table_name), confidence=confidence)
        else:
            fig = forecast_line_plot(self.connection_context.table(predict_table_name), self.connection_context.table(actual_table_name), confidence)
        if output_dir is None:
            destination_dir = os.path.join(tempfile.gettempdir(), "hanaml_chart")
        else:
            destination_dir = output_dir
        if not os.path.exists(destination_dir):
            try:
                os.makedirs(destination_dir, exist_ok=True)
            except Exception as e:
                logger.error("Error creating directory %s: %s", destination_dir, e)
                raise
        output_file = os.path.join(
                    destination_dir,
                    f"{predict_table_name}_forecast_line_plot.html",
                )
        with Path(output_file).open("w", encoding="utf-8") as f:
            f.write(fig.to_html(full_html=True))
        fig.show() #directly display in jupyter
        return json.dumps({"html_file": output_file})

    async def _arun(
        self, predict_table_name: str, actual_table_name: Optional[str]=None, confidence: Optional[tuple]=None, output_dir: Optional[str]=None, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(predict_table_name, actual_table_name, confidence, output_dir, run_manager=run_manager)
