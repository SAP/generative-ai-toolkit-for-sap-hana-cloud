"""
This module define the agent tools for `correlation()` function in hana-ml.
"""
import json
import logging
from typing import Type, Annotated
from pydantic import BaseModel, Field
from annotated_types import Interval

from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.algorithms.pal.tsa.correlation_function import correlation
from hdbcli.dbapi import ProgrammingError
from hana_ai.utility import remove_prefix_sharp


logger = logging.getLogger(__name__)

class CorrelationInput(BaseModel):
    """
    Class for input parameters of the correlation() function.
    """
    table_name : str = Field(description="Table (or view) containing the input data for correlation computation. " +\
    "If not provided, ask the user, do not guess")
    key : str = Field(description="The key of input table. If not provided, ask the user, do not guess")
    x : str = Field(description="Column name of the 1st time-series data for correlation computation. If not provided, ask the user, do not guess")
    y : str = Field(description="Column name of the 2nd time-series data for correlation computation. " +\
    "If not prvoided, auto-correlation of the 1st time-series data will be computed", default=None)
    thread_ratio : float = Field(description="The ratio of available threads to be used", default=None)
    method : str = Field(description="The method for calculating the correlation coefficiennts, with valid options 'auto', 'brute_force' and 'fft'.", default=None)
    max_lag : int = Field(description="Maximum number of lags for (auto)correlation computation.", default=None)
    calculate_pacf : bool = Field(description="If set as True, calculate the partial autocorrelation coefficient (pacf) as well.", default=None)
    calculate_confint : bool = Field(description="If set as True, calculate the confidence intervals of autocorrelation coefficients",
    default=False)
    alpha : Annotated[float, Interval(gt=0, lt=1)] = Field(description="Specifies the confidence level for confidence interval, which should be a positive value" +\
    " between 0 and 1. For example, the value of 0.1 implies a 90% confidence interval.", default=None)
    bartlett : bool = Field(description="If set as True, Bartlett's formula is used to calculate the confidence bound," +\
    " otherwise standard error is used", default=None)

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
                * - table_name
                  - Table (or view) containing the input data for correlation computation.
                * - key
                  - The key of input table.
                * - x
                  - Column name of the 1st time-series data for correlation computation.
                * - y
                  - Column name of the 2nd time-series data for correlation computation.
                    If not prvoided, auto-correlation of the 1st time-series data will be computed.
                * - thread_ratio
                  - The ratio of available threads to be used.
                * - method
                  - The method for calculating the correlation coefficients.
                * - max_lag
                  - Maximum number of lags for correlation computation.
                * - calculate_pacf
                  - If set as True, calculate the partial autocorrelation coefficient (pacf) as well.
                * - calculate_confint
                  - If set as True, calculate the confidence bounds of autocorrelation coefficients.
                * - alpha
                  - Specifies the confidence level of confidence interval defined by confidence bound,
                    which should be a positive value between 0 and 1. For example, the value of 0.1 implies
                    a 90% confidence interval.
                * - bartlett
                  - If set as True, Bartlett's formula is used to calculate the confidence bounds.
    """
    name: str = "correlation_function"
    """Name of the tool."""
    description: str = "To compute the auto-correlation of a single time-series or the correlation between two time-series."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = CorrelationInput
    return_direct: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct = False
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )

    def _run(#pylint:disable=too-many-return-statements
        self,
        **kwargs) -> str:
        if "kwargs" in kwargs:
            kwargs = kwargs.get("kwargs")
        table_name = kwargs.get("table_name", None)
        if table_name is None:
            return "Input table name (`table_name`) is required"
        key, x = kwargs.get("key", None), kwargs.get("x", None)
        if key is None:
            return "key is required"
        if x is None:
            return "x (the 1st time-series) is required"
        y = kwargs.get("y", None)
        thread_ratio = kwargs.get("thread_ratio", None)
        method = kwargs.get("method", None)
        max_lag = kwargs.get("max_lag", None)
        calculate_pacf = kwargs.get("calculate_pacf", None)
        calculate_confint = kwargs.get("calculate_confint", None)
        alpha = kwargs.get("alpha", None)
        bartlett = kwargs.get("bartlett", None)
        if calculate_confint and y is not None:
            msg = "Since confidence intervals are only applicable to the autocorrelation of one time-series, " +\
            "the value of `calculate_confint` needs to be changed to False to proceed."
            return f'ValueError occurred: {msg}'
        input_data = self.connection_context.table(table_name)
        try:
            cf_coef = correlation(data=input_data,
                                  key=key, x=x, y=y, thread_ratio=thread_ratio,
                                  method=method, max_lag=max_lag,
                                  calculate_pacf=calculate_pacf,
                                  calculate_confint=calculate_confint,
                                  alpha=alpha, bartlett=bartlett)
        except ValueError as verr:
            # Handles invalid parameter values (e.g., alpha not in [0,1])
            return f'ValueError occurred: {str(verr)}'
        except KeyError as kerr:
            # Handles missing columns in the DataFrame
            return f'KeyError occurred: {str(kerr)}'
        except TypeError as terr:
            # Handles type mismatches (e.g., non-numeric input where number expected)
            return f'TypeError occurred: {str(terr)}'
        except ProgrammingError as perr:
            # Handles invalid table name specifically
            if 'invalid table name' in str(perr):
                return f'Invalid table name: Could not find table/view {table_name}'
        cf_table = remove_prefix_sharp(f"{table_name}_CORRELATION_RESULT")
        cf_coef.save(cf_table, force=True)
        return json.dumps({"correlation_result_table" : cf_table})

    async def _arun(self,
                    **kwargs
                    )-> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs)
