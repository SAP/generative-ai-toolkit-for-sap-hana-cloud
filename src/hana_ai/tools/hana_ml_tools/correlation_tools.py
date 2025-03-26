"""
This module define the agent tools for `correlation()` function in hana-ml.
"""
import json
import logging
from typing import Type
from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.algorithms.pal.tsa.correlation_function import correlation
from hana_ai.utility import remove_prefix_sharp

logger = logging.getLogger(__name__)

class CorrelationInput(BaseModel):
    """
    Class for input parameters of the correlation() function.
    """
    input_table : str = Field(description="Table (or view) containing the input data for correlation calculation. If not provided, ask the user, do not guess")
    key : str = Field(description="The key of input table. If not provided, ask the user, do not guess")
    x : str = Field(description="Column name of the 1st time-series data for correlation computation. If not provided, ask the user, do not guess")
    y : str = Field(description="Column name of the 2nd time-series data for correlation computation", default=None)
    thread_ratio : float = Field(description="The ratio of available threads to be used", default=None)
    method : str = Field(description="The method for calculating the correlation coefficiennts, with valid options including 'auto', 'brute_force' and 'fft'.", default=None)
    max_lag : int = Field(description="Maximum number of lags for correlation computation.", default=None)
    calculate_pacf : bool = Field(description="If set as True, calculate the partial autocorrelation coefficient(pacf) instead", default=None)
    calculate_confint : bool = Field(description="If set as True, calculate the confidence interval", default=None)
    alpha : float = Field(description="Specifies the level of confidence for the confidence interval", default=None)
    bartlett : bool = Field(description="If set as True, Bartlett's formula is used to calculate the confidence bound, otherwise standard error is used", default=None)

class Correlation(BaseTool):
    r"""
    This tool computes the correlation coefficients between time-series.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The result string containing the correlation result table name.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - input_table
                  - Table (or view) containing the input data for correlation calculation.
                * - key
                  - The key of input table.
                * - x
                  - Column name of the 1st time-series data for correlation computation.
                * - y
                  - Column name of the 2nd time-series data for correlation computation.
                * - thread_ratio
                  - The ratio of available threads to be used.
                * - method
                  - The method for calculating the correlation coefficients.
                * - max_lag
                  - Maximum number of lags for correlation computation.
                * - calculate_pacf
                  - If set as True, calculate the partial autocorrelation coefficient (pacf) instead.
                * - calculate_confint
                  - If set as True, calculate the confidence interval.
                * - alpha
                  - Specifies the level of confidence for the confidence interval.
                * - bartlett
                  - If set as True, Bartlett's formula is used to calculate the confidence bound.
    """
    name: str = "Correlation between time-series."
    """Name of the tool."""
    description: str = "To compute the auto-correlation of a single time-series, or the correlation between two time-series."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = CorrelationInput
    #return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context
        )

    def _run(#pylint:disable=too-many-positional-arguments
        self,
        input_table : str,
        key : str,
        x : str,
        y : str=None,
        thread_ratio : float=None,
        method : str=None,
        max_lag : int=None,
        calculate_pacf : bool=None,
        calculate_confint : bool=False,
        alpha : float=None,
        bartlett : bool=None,
        run_manager: CallbackManagerForToolRun = None#pylint:disable=unused-argument
        ) -> str:
        cf_coef = correlation(data=self.connection_context.table(input_table),
                              key=key, x=x, y=y, thread_ratio=thread_ratio,
                              method=method, max_lag=max_lag,
                              calculate_pacf=calculate_pacf,
                              calculate_confint=calculate_confint,
                              alpha=alpha, bartlett=bartlett)
        cf_table = remove_prefix_sharp(f"{input_table}_CORRELATION_RESULT")
        cf_coef.save(cf_table, force=True)
        return json.dumps({"correlation_result_table" : cf_table})

    async def _arun(self,#pylint:disable=too-many-positional-arguments
                    input_table : str,
                    key : str,
                    x : str,
                    y : str=None,
                    thread_ratio : float=None,
                    method : str=None,
                    max_lag : int=None,
                    calculate_pacf : bool=None,
                    calculate_confint : bool=False,
                    alpha : float=None,
                    bartlett : bool=None,
                    run_manager: AsyncCallbackManagerForToolRun = None#pylint:disable=unused-argument
                    )-> str:
        """Use the tool asynchronously."""
        return self._run(input_table=input_table,
                         key=key, x=x, y=y,
                         thread_ratio=thread_ratio,
                         method=method,
                         max_lag=max_lag,
                         calculate_pacf=calculate_pacf,
                         calculate_confint=calculate_confint,
                         alpha=alpha,
                         bartlett=bartlett,
                         run_manager=run_manager)
