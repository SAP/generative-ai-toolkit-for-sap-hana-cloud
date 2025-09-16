"""
The following classe is available:

    * :class `TimeSeriesDatasetReport`

"""

import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from hana_ml import ConnectionContext
from hana_ml.visualizers.unified_report import UnifiedReport
from hana_ai.tools.hana_ml_tools.utility import _CustomEncoder

logger = logging.getLogger(__name__)

class TSDatasetInput(BaseModel):
    """
    The input schema for the TSDatasetTool.
    """
    select_statement: str = Field(description="the select statement of dataframe. If not provided, ask the user. Do not guess.")
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    output_dir: Optional[str] = Field(description="the output directory to save the report, it is optional", default=None)

class TimeSeriesDatasetReport(BaseTool):
    """
    This tool generates a report for a time series.

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
                * - select_statement
                  - the select statement of the dataframe. If not provided, ask the user. Do not guess.
                * - key
                  - the key of the dataset. If not provided, ask the user. Do not guess.
                * - endog
                  - the endog of the dataset. If not provided, ask the user. Do not guess
    """
    name: str = "ts_dataset_report"
    """Name of the tool."""
    description: str = "To generate a timeseries report for a given HANA DataFrame. "
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
        self,
        **kwargs
    ) -> str:
        """Use the tool."""

        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        select_statement = kwargs.get("select_statement", None)
        if select_statement is None:
            return "Select statement is required"
        key = kwargs.get("key", None)
        if key is None:
            return "key is required"
        endog = kwargs.get("endog", None)
        if endog is None:
            return "endog is required"
        output_dir = kwargs.get("output_dir", None)

        df_check = self.connection_context.sql(select_statement)
        columns = df_check.columns
        if key not in columns:
            return json.dumps({
          "error": f"key '{key}' does not exist in the dataframe!"
            }, cls=_CustomEncoder)
        if endog not in columns:
            return json.dumps({
          "error": f"endog '{endog}' does not exist in the dataframe!"
            }, cls=_CustomEncoder)
        df = self.connection_context.sql(select_statement).select(key, endog)
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

        output_file = os.path.join(destination_dir, "ts_report")
        ur.display(save_html=output_file)
        return json.dumps({"html_file": str(Path(output_file + ".html").as_posix())}, ensure_ascii=False)

    async def _arun(
        self, **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs)
