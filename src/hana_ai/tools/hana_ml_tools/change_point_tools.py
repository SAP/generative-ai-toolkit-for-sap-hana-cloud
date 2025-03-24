"""
This module contains the tools for change point detection (CPD).
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
from hana_ml.algorithms.pal.tsa.changepoint import BCPD
from hana_ai.utility import remove_prefix_sharp

logger = logging.getLogger(__name__)

class BayesianChangePointInput(BaseModel):
    """
    Class for input arguments of Bayesian change point detection (BCPD).
    """
    input_table : str = Field(description="Table (or view) containing the input time-series data for Bayesian change point detection (BCPD)." +\
    " If not provided, ask the user, do not guess")
    key : str = Field(description="The key of the input data. If not provided, ask the user, do not guess")
    endog : str = Field(description="The Column that contains time-series data for Bayesian change point detection.", default=None)
    max_tcp : int = Field(description="Maximum number of trend change points to be detected. If not provided, ask the user, do not guess")
    max_scp : int = Field(description="Maximum number of season change points to be detected. If not provided, ask the user, do not guess")
    trend_order : int = Field(description="The Order of trend segments that for decomposition", default=None)
    max_harmonic_order : int = Field(description="Maximum order of harmonic waves within seasonal segments", default=None)
    min_period : int = Field(description="Minimum possible period within seasonal segments", default=None)
    max_period : int = Field(description="Maximum possible period within seasonal segments", default=None)
    random_seed : int = Field(description="The seed used to initialize the random number generator", default=None)
    max_iter : int = Field(description="The number of iterations for BCPD, more iterations require longer running time while leading to more precise detection result", default=None)#pylint:disable=line-too-long
    interval_ratio : float = Field(description="This parameter regulates the interval between change points, which should be larger than the corresponding portion of total length", default=None)#pylint:disable=line-too-long

class BayesianChangePoint(BaseTool):
    r"""
    This tool is used to find the change points in time-series data using Bayesian change point detection.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The result string containing the trend change points, seasonal change points, periods, and the decomposed table name.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - input_table
                  - Table (or view) containing the input time-series data for Bayesian change point detection (BCPD).
                * - key
                  - The key of the input data.
                * - endog
                  - The Column that contains time-series data for Bayesian change point detection.
                * - max_tcp
                  - Maximum number of trend change points to be detected.
                * - max_scp
                  - Maximum number of season change points to be detected.
                * - trend_order
                  - The Order of trend segments that for decomposition.
                * - max_harmonic_order
                  - Maximum order of harmonic waves within seasonal segments.
                * - min_period
                  - Minimum possible period within seasonal segments.
                * - max_period
                  - Maximum possible period within seasonal segments.
                * - random_seed
                  - The seed used to initialize the random number generator.
                * - max_iter
                  - The number of iterations for BCPD.
                * - interval_ratio
                  - This parameter regulates the interval between change points.
    """
    name: str = "beyesian_change_point_detection"
    """Name of the tool."""
    description: str = "To find change points in time-series using bayesion change point detection."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = BayesianChangePointInput
    return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext
    ) -> None:
        super().__init__(  # type: ignore[call-arg
            connection_context=connection_context
        )

    def _run(#pylint:disable=too-many-positional-arguments
        self,
        input_table : str,
        key : str,
        max_tcp : int,
        max_scp : int,
        endog : str=None,
        trend_order : int=None,
        max_harmonic_order : int=None,
        min_period : int=None,
        max_period : int=None,
        random_seed : int=None,
        max_iter : int=None,
        interval_ratio : float=None,
        run_manager: CallbackManagerForToolRun = None#pylint:disable=unused-argument
        ) -> str:
        bcpd = BCPD(max_tcp=max_tcp,
                    max_scp=max_scp,
                    trend_order=trend_order,
                    max_harmonic_order=max_harmonic_order,
                    min_period=min_period,
                    max_period=max_period,
                    random_seed=random_seed,
                    max_iter=max_iter,
                    interval_ratio=interval_ratio)
        bcpd_dfs = bcpd.fit_predict(data=self.connection_context.table(input_table),
                                    key=key, endog=endog)
        t_cps = ', '.join([str(stmp) for stmp in list(bcpd_dfs[0].collect().iloc[:,1])])
        s_cps = ', '.join([str(stmp) for stmp in list(bcpd_dfs[1].collect().iloc[:,1])])
        periods = ', '.join([str(prd) for prd in list(bcpd_dfs[2].collect().iloc[:,1])])
        result_dict = {"trend_change_points": t_cps, "seasonal_change_points": s_cps, "periods": periods}
        decompose_tbl = remove_prefix_sharp(input_table + '_BCPD_DECOMPOSED')
        bcpd_dfs[3].save(decompose_tbl, force=True)
        result_dict["bcpd_decomposed_table"] = decompose_tbl
        return json.dumps(result_dict)

    async def _arun(#pylint:disable=too-many-positional-arguments
        self,
        input_table : str,
        key : str,
        max_tcp : int,
        max_scp : int,
        endog : str=None,
        trend_order : int=None,
        max_harmonic_order : int=None,
        min_period : int=None,
        max_period : int=None,
        random_seed : int=None,
        max_iter : int=None,
        interval_ratio : float=None,
        run_manager: AsyncCallbackManagerForToolRun = None#pylint:disable=unused-argument
        ) -> str:
        return self._run(input_table=input_table,
                         key=key,
                         endog=endog,
                         max_tcp=max_tcp,
                         max_scp=max_scp,
                         trend_order=trend_order,
                         max_harmonic_order=max_harmonic_order,
                         min_period=min_period,
                         max_period=max_period,
                         random_seed=random_seed,
                         max_iter=max_iter,
                         interval_ratio=interval_ratio,
                         run_manager=run_manager)
