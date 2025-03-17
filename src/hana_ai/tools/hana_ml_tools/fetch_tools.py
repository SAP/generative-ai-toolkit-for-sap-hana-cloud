"""
This module contains the functions to fetch data from HANA.
"""

import json
import logging
from typing import Optional, Type
from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext

logger = logging.getLogger(__name__)

class FetchDataInput(BaseModel):
    """
    The input schema for the FetchDataTool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    top_n: Optional[int] = Field(description="the number of rows to fetch, it is optional", default=None)

class FetchDataTool(BaseTool):
    """
    This tool fetches data from a given table.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The fetched data.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - table_name
                  - The name of the table. If not provided, ask the user. Do not guess.
                * - top_n
                  - The number of rows to fetch, it is optional
    """
    name: str = "fetch_data"
    """Name of the tool."""
    description: str = "Fetch data from a given table."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = FetchDataInput
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
        self, table_name: str, top_n: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        if top_n is None:
            results = self.connection_context.table(table_name).collect()
        else:
            results = self.connection_context.table(table_name).head(top_n).collect()
        # serialize the results
        return json.dumps({"fetched_data": results.to_json()})

    async def _arun(
        self, table_name: str, key: str, endog: str, auto: Optional[bool] = None,
        detect_intermittent_ts: Optional[bool] = None, smooth_method: Optional[str] = None,
        window_size: Optional[int] = None, loess_lag: Optional[int] = None,
        current_value_flag: Optional[bool] = None, outlier_method: Optional[str] = None,
        threshold: Optional[float] = None, detect_seasonality: Optional[bool] = None,
        alpha: Optional[float] = None, extrapolation: Optional[bool] = None,
        periods: Optional[int] = None, random_state: Optional[int] = None,
        n_estimators: Optional[int] = None, max_samples: Optional[int] = None,
        bootstrap: Optional[bool] = None, contamination: Optional[float] = None,
        minpts: Optional[int] = None, eps: Optional[float] = None,
        distiance_method: Optional[str] = None, dbscan_normalization: Optional[bool] = None,
        dbscan_outlier_from_cluster: Optional[bool] = None, thread_ratio: Optional[float] = None,
        residual_usage: Optional[str] = None, voting_config: Optional[dict] = None,
        voting_outlier_method_criterion: Optional[float] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(
            table_name, key, endog, auto=auto, detect_intermittent_ts=detect_intermittent_ts, smooth_method=smooth_method,
            window_size=window_size, loess_lag=loess_lag, current_value_flag=current_value_flag, outlier_method=outlier_method,
            threshold=threshold, detect_seasonality=detect_seasonality, alpha=alpha, extrapolation=extrapolation,
            periods=periods, random_state=random_state, n_estimators=n_estimators, max_samples=max_samples,
            bootstrap=bootstrap, contamination=contamination, minpts=minpts, eps=eps, distiance_method=distiance_method,
            dbscan_normalization=dbscan_normalization, dbscan_outlier_from_cluster=dbscan_outlier_from_cluster,
            thread_ratio=thread_ratio, residual_usage=residual_usage, voting_config=voting_config,
            voting_outlier_method_criterion=voting_outlier_method_criterion, run_manager=run_manager
        )
