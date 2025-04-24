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
    return_direct: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )

    def _run(
        self, table_name: str, key: str, endog: str, output_dir: Optional[str]=None, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        # check hana has the table
        if not self.connection_context.has_table(table_name):
            return json.dumps({"error": f"Table {table_name} does not exist."})
        # check key in the table
        if key not in self.connection_context.table(table_name).columns:
            return json.dumps({"error": f"Key {key} does not exist in table {table_name}."})
        # check endog in the table
        if endog not in self.connection_context.table(table_name).columns:
            return json.dumps({"error": f"Endog {endog} does not exist in table {table_name}."})
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
                    table_name)
        ur.display(save_html=output_file)
        ur.display() #directly display in jupyter
        return json.dumps({"html_file": output_file + "_report.html"})#hana-ml report will add _report.html to the file name

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
    return_direct: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )

    def _run(
        self, predict_table_name: str, actual_table_name: Optional[str]=None, confidence: Optional[tuple]=None, output_dir: Optional[str]=None, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        # check predict_table_name in the hana db
        if not self.connection_context.has_table(predict_table_name):
            return json.dumps({"error": f"Table {predict_table_name} does not exist."})
        # check actual_table_name in the hana db
        if actual_table_name is not None and not self.connection_context.has_table(actual_table_name):
            return json.dumps({"error": f"Table {actual_table_name} does not exist."})
        predict_df = self.connection_context.table(predict_table_name)
        if confidence is None:
            if "YHAT_LOWER" in predict_df.columns and "YHAT_UPPER" in predict_df.columns:
                # check if "YHAT_LOWER" column has values
                if not predict_df["YHAT_LOWER"].collect()["YHAT_LOWER"].isnull().all():
                    confidence = ("YHAT_LOWER", "YHAT_UPPER")
            elif "LO80" in predict_df.columns and "HI80" in predict_df.columns:
                if not predict_df["LO80"].collect()["LO80"].isnull().all():
                    confidence = ("LO80", "HI80")
            elif "LO95" in predict_df.columns and "HI95" in predict_df.columns:
                if not predict_df["LO95"].collect()["LO95"].isnull().all():
                    if confidence is None:
                        confidence = ("LO95", "HI95")
                    else:
                        confidence = confidence + ("LO95", "HI95")
            elif "PI1_LOWER" in predict_df.columns and "PI1_UPPER" in predict_df.columns:
                if not predict_df["PI1_LOWER"].collect()["PI1_LOWER"].isnull().all():
                    confidence = ("PI1_LOWER", "PI1_UPPER")
            elif "PI2_LOWER" in predict_df.columns and "PI2_UPPER" in predict_df.columns:
                if not predict_df["PI2_LOWER"].collect()["PI2_LOWER"].isnull().all():
                    if confidence is None:
                        confidence = ("PI2_LOWER", "PI2_UPPER")
                    else:
                        confidence = confidence + ("PI2_LOWER", "PI2_UPPER")

        if actual_table_name is None:
            fig = forecast_line_plot(predict_df, confidence=confidence)
        else:
            fig = forecast_line_plot(predict_df, self.connection_context.table(actual_table_name), confidence)
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
