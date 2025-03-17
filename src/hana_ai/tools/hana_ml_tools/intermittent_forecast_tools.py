"""
This module contains the tools for intermittent demand forecasting.
"""

import json
import logging
from typing import Optional, Type, Union
from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.algorithms.pal.tsa.intermittent_forecast import intermittent_forecast

from hana_ai.utility import remove_prefix_sharp

logger = logging.getLogger(__name__)

class IntermittentForecastInput(BaseModel):
    """
    The input schema for the IntermittentForecast tool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    p: Optional[int] = Field(description="the smoothing parameter for demand, it is optional", default=None)
    q: Optional[int] = Field(description="the smoothing parameter for the time-intervals between intermittent demands, it is optional", default=None)
    forecast_num: Optional[int] = Field(description="the number of forecast steps, it is optional", default=None)
    optimizer: Optional[str] = Field(description="the optimizer for the model chosen from {'lbfgsb', 'brute', 'sim_annealing'}, it is optional", default=None)
    method: Optional[str] = Field(description="the method for the output chosen from {'sporadic', 'constant'}, it is optional", default=None)
    grid_size: Optional[int] = Field(description="specifies the number of steps from the start point to the length of data for grid search, it is optional", default=None)
    optimize_step: Optional[int] = Field(description="specifies the minimum step for each iteration of LBFGSB method, it is optional", default=None)
    accuracy_measure: Union[Optional[str], Optional[list]] = Field(description="the metric to quantify how well a model fits input data chosen from 'mse', 'rmse', 'mae', 'mape', 'smape', 'mase', it is optional", default=None)
    ignore_zero: Optional[bool] = Field(description="whether to ignore zero values in the dataset to calculate mape, it is optional", default=None)
    expost_flag: Optional[bool] = Field(description="whether to output the expost forecast, it is optional", default=None)
    thread_ratio: Optional[float] = Field(description="the ratio of threads to use for parallel processing, it is optional", default=None)
    iter_count: Optional[int] = Field(description="a positive integer that controls the iteration of the simulated annealing, it is optional", default=None)
    random_state: Optional[int] = Field(description="specifies the seed for random number generator and valid for Simulated annealing method, it is optional", default=None)
    penalty: Optional[float] = Field(description="a penalty is applied to the cost function to avoid over-fitting, it is optional", default=None)

class IntermittentForecast(BaseTool):
    """
    This tool generates forecast for the intermittent demand dataset.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The name of the predicted result table and the statistics of the forecast.

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
                  - the endog of the dataset. If not provided, ask the user. Do not guess.
                * - p
                  - the smoothing parameter for demand, it is optional
                * - q
                  - the smoothing parameter for the time-intervals between intermittent demands, it is optional
                * - forecast_num
                  - the number of forecast steps, it is optional
                * - optimizer
                  - the optimizer for the model chosen from {'lbfgsb', 'brute', 'sim_annealing'}, it is optional
                * - method
                  - the method for the output chosen from {'sporadic', 'constant'}, it is optional
                * - grid_size
                  - specifies the number of steps from the start point to the length of data for grid search, it is optional
                * - optimize_step
                  - specifies the minimum step for each iteration of LBFGSB method, it is optional
                * - accuracy_measure
                  - the metric to quantify how well a model fits input data chosen from 'mse', 'rmse', 'mae', 'mape', 'smape', 'mase', it is optional
                * - ignore_zero
                  - whether to ignore zero values in the dataset to calculate mape, it is optional
                * - expost_flag
                  - whether to output the expost forecast, it is optional
                * - thread_ratio
                  - the ratio of threads to use for parallel processing, it is optional
                * - iter_count
                  - a positive integer that controls the iteration of the simulated annealing, it is optional
                * - random_state
                  - specifies the seed for random number generator and valid for Simulated annealing method, it is optional
                * - penalty
                  - a penalty is applied to the cost function to avoid over-fitting, it is optional
    """
    name: str = "intermittent_forecast"
    """Name of the tool."""
    description: str = "To generate forecast for the intermittent demand dataset. "
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = IntermittentForecastInput
    return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context
        )

    def _run(
        self, table_name: str, key: str, endog: str, p: Optional[int] = None, q: Optional[int] = None,
        forecast_num: Optional[int] = None, optimizer: Optional[str] = None, method: Optional[str] = None,
        grid_size: Optional[int] = None, optimize_step: Optional[int] = None,
        accuracy_measure: Union[Optional[str], Optional[list]] = None, ignore_zero: Optional[bool] = None,
        expost_flag: Optional[bool] = None, thread_ratio: Optional[float] = None, iter_count: Optional[int] = None,
        random_state: Optional[int] = None, penalty: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        df = self.connection_context.table(table_name).select(key, endog)
        result, stats = intermittent_forecast(
            df,
            key=key, endog=endog,
            p=p, q=q, forecast_num=forecast_num, optimizer=optimizer, method=method, grid_size=grid_size,
            optimize_step=optimize_step, accuracy_measure=accuracy_measure, ignore_zero=ignore_zero,
            expost_flag=expost_flag, thread_ratio=thread_ratio, iter_count=iter_count, random_state=random_state,
            penalty=penalty
        )
        predicted_results = f"{table_name}_INTERMITTENT_FORECAST_RESULT"
        result.save(remove_prefix_sharp(predicted_results), force=True)
        outputs = {
            "predicted_result_table": remove_prefix_sharp(predicted_results),
        }
        for _, row in stats.collect().iterrows():
            outputs[row[stats.columns[0]]] = row[stats.columns[1]]
        return json.dumps(outputs)

    async def _run_async(
        self, table_name: str, key: str, endog: str, p: Optional[int] = None, q: Optional[int] = None,
        forecast_num: Optional[int] = None, optimizer: Optional[str] = None, method: Optional[str] = None,
        grid_size: Optional[int] = None, optimize_step: Optional[int] = None,
        accuracy_measure: Union[Optional[str], Optional[list]] = None, ignore_zero: Optional[bool] = None,
        expost_flag: Optional[bool] = None, thread_ratio: Optional[float] = None, iter_count: Optional[int] = None,
        random_state: Optional[int] = None, penalty: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(
            table_name, key, endog, p=p, q=q, forecast_num=forecast_num, optimizer=optimizer, method=method,
            grid_size=grid_size, optimize_step=optimize_step, accuracy_measure=accuracy_measure, ignore_zero=ignore_zero,
            expost_flag=expost_flag, thread_ratio=thread_ratio, iter_count=iter_count, random_state=random_state,
            penalty=penalty, run_manager=run_manager
        )
