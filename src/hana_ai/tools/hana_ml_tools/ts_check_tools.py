"""
This module is used to do some checks on the time series dataset.

The following classes are available:

    * :class `TimeSeriesCheck`
    * :class `StationarityTest`
    * :class `TrendTest`
    * :class `SeasonalityTest`
    * :class `WhiteNoiseTest`
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
from hana_ml.algorithms.pal.tsa.stationarity_test import stationarity_test
from hana_ml.algorithms.pal.tsa.trend_test import trend_test
from hana_ml.algorithms.pal.tsa.seasonal_decompose import seasonal_decompose
from hana_ml.algorithms.pal.tsa.white_noise_test import white_noise_test

from hana_ai.tools.hana_ml_tools.utility import _CustomEncoder

logger = logging.getLogger(__name__)

def ts_char(df, key, endog):
    """
    This function is used to get the characteristics of time series data.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    key : str
        The key column of the DataFrame.
    endog : str
        The endogenous column of the DataFrame.
    """
    analysis_result = ''
    # Index info
    analysis_result += f"Index: starts from {df[key].min()} to {df[key].max()}. Time series length is {df.count()}\n"

    key_col_type = df.get_table_structure()[key]
    key_ = key
    df_ = df
    if 'INT' not in key_col_type.upper():
        key_ = "NEW_" + key
        df_ = df.add_id(key_, ref_col=key)

    # Intermitent Test
    zero_values = df_.filter(f'"{endog}" = 0').count()
    total_values = df_.count()
    if total_values == 0:
        zero_proportion = 1
    else:
        zero_proportion = zero_values / total_values
    analysis_result += f"Intermittent Test: proportion of zero values is {zero_proportion}\n"

    # Stationarity Test
    result = stationarity_test(df_, key_, endog).collect()
    analysis_result += "Stationarity Test: "
    for _, row in result.iterrows():
        analysis_result += f"The {row['STATS_NAME']} is {row['STATS_VALUE']}."
    analysis_result += "\n"

    # Trend Test
    result = trend_test(df_, key_, endog)[0].collect()
    for _, row in result.iterrows():
        if row['STAT_NAME'] == 'TREND':
            if row['STAT_VALUE'] == 1:
                analysis_result += 'Trend Test:' + " Upward trend."
            elif row['STAT_VALUE'] == -1:
                analysis_result += 'Trend Test:' + " Downward trend."
            else:
                analysis_result += 'Trend Test:' + " No trend."
    analysis_result += "\n"

    # Seasonality Test
    result = seasonal_decompose(df_, key_, endog)[0].collect()
    analysis_result += "Seasonality Test: "
    for _, row in result.iterrows():
        analysis_result += f"The {row['STAT_NAME']} is {row['STAT_VALUE']}."

    return analysis_result

class TSCheckInput(BaseModel):
    """
    The input schema for the TimeSeriesCheckTool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")

class StationarityTestInput(BaseModel):
    """
    The input schema for the StationarityTestTool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    method: Optional[str] = Field(description="the method of the stationarity test chosen from {'kpss', 'adf'}, it is optional", default=None)
    mode: Optional[str] = Field(description="the mode of the stationarity test chosen from {'level', 'trend', 'no'}, it is optional", default=None)
    lag: Optional[int] = Field(description="the lag of the stationarity test, it is optional", default=None)
    probability: Optional[float] = Field(description="the confidence level for confirming stationarity, it is optional", default=None)

class TrendTestInput(BaseModel):
    """
    The input schema for the TrendTestTool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    method: Optional[str] = Field(description="the method of the trend test chosen from {'mk', 'difference-sign'}, it is optional", default=None)
    alpha: Optional[float] = Field(description="the significance level for the trend test, it is optional", default=None)

class SeasonalityTestInput(BaseModel):
    """
    The input schema for the SeasonalityTestTool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    alpha: Optional[float] = Field(description="the criterion for the autocorrelation coefficient, it is optional", default=None)
    decompose_type: Optional[str] = Field(description="the type of decomposition chosen from {'additive', 'multiplicative', 'auto'}, it is optional", default=None)
    extrapolation: Optional[bool] = Field(description="whether to extrapolate the endpoints or not, it is optional", default=None)
    smooth_width: Optional[int] = Field(description="the width of the smoothing window, it is optional", default=None)
    auxiliary_normalitytest: Optional[bool] = Field(description="specifies whether to use normality test to identify model types, it is optional", default=None)
    periods: Optional[int] = Field(description="the length of the periods, it is optional", default=None)
    decompose_method: Optional[str] = Field(description="the method of decomposition chosen from {'stl', 'traditional'}, it is optional", default=None)
    stl_robust: Optional[bool] = Field(description="whether to use robust decomposition or not only valid for 'stl' decompose method, it is optional", default=None)
    stl_seasonal_average: Optional[bool] = Field(description="whether to use seasonal average or not only valid for 'stl' decompose method, it is optional", default=None)
    smooth_method_non_seasonal: Optional[str] = Field(description="the method of smoothing for non-seasonal component chosen from {'moving_average', 'super_smoother'}, it is optional", default=None)

class WhiteNoiseTestInput(BaseModel):
    """
    The input schema for the WhiteNoiseTestTool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    lag: Optional[int] = Field(description="specifies the lag autocorrelation coefficient that the statistic will be based on, it is optional", default=None)
    probability: Optional[float] = Field(description="the confidence level used for chi-square distribution., it is optional", default=None)
    model_df: Optional[int] = Field(description="the degree of freedom of the model, it is optional", default=None)

class TimeSeriesCheck(BaseTool):
    """
    This tool calls stationarity test, intermittent check, trend test and seasonality test for the given time series data.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The characteristics of the time series data.

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
    name: str = "ts_check"
    """Name of the tool."""
    description: str = "To check the time series data for stationarity, intermittent, trend and seasonality."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = TSCheckInput
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
        self, table_name: str, key: str, endog: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        df = self.connection_context.table(table_name).select(key, endog)
        return ts_char(df, key, endog)

    async def _arun(
        self, table_name: str, key: str, endog: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(table_name, key, endog, run_manager=run_manager)

class StationarityTest(BaseTool):
    """
    This tool calls stationarity test for the given time series data.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The stationarity statistics of the time series data.

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
                * - method
                  - the method of the stationarity test chosen from {'kpss', 'adf'}, it is optional
                * - mode
                  - the mode of the stationarity test chosen from {'level', 'trend', 'no'}, it is optional
                * - lag
                  - the lag of the stationarity test, it is optional
                * - probability
                  - the confidence level for confirming stationarity, it is optional
    """
    name: str = "stationarity_test"
    """Name of the tool."""
    description: str = "To check the stationarity of the time series data."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = StationarityTestInput
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
        table_name: str,
        key: str,
        endog: str,
        method: Optional[str] = None,
        mode: Optional[str] = None,
        lag: Optional[int] = None,
        probability: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        df = self.connection_context.table(table_name).select(key, endog)
        result = stationarity_test(data=df,
                                   key=key,
                                   endog=endog,
                                   method=method,
                                   mode=mode,
                                   lag=lag,
                                   probability=probability).collect()
        analysis_result = {}
        for _, row in result.iterrows():
            analysis_result[row['STATS_NAME']] = row['STATS_VALUE']
        return json.dumps(analysis_result, cls=_CustomEncoder)

    async def _arun(
        self,
        table_name: str,
        key: str,
        endog: str,
        method: Optional[str] = None,
        mode: Optional[str] = None,
        lag: Optional[int] = None,
        probability: Optional[float] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(table_name, key, endog, method, mode, lag, probability, run_manager=run_manager)

class TrendTest(BaseTool):
    """
    This tool calls trend test for the given time series data.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The trend statistics of the time series data.

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
                * - method
                  - the method of the trend test chosen from {'mk', 'difference-sign'}, it is optional
                * - alpha
                  - the significance level for the trend test, it is optional
    """
    name: str = "trend_test"
    """Name of the tool."""
    description: str = "To check the trend of the time series data."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = TrendTestInput
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
        table_name: str,
        key: str,
        endog: str,
        method: Optional[str] = None,
        alpha: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        df = self.connection_context.table(table_name).select(key, endog)
        result = trend_test(data=df,
                            key=key,
                            endog=endog,
                            method=method,
                            alpha=alpha)[0].collect()
        analysis_result = {}
        for _, row in result.iterrows():
            if row['STAT_NAME'] == 'TREND':
                if row['STAT_VALUE'] == 1:
                    analysis_result['Trend'] = "Upward trend."
                elif row['STAT_VALUE'] == -1:
                    analysis_result['Trend'] = "Downward trend."
                else:
                    analysis_result['Trend'] = "No trend."
        return json.dumps(analysis_result, cls=_CustomEncoder)

    async def _arun(
        self,
        table_name: str,
        key: str,
        endog: str,
        method: Optional[str] = None,
        alpha: Optional[float] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(table_name, key, endog, method, alpha, run_manager=run_manager)

class SeasonalityTest(BaseTool):
    """
    This tool calls seasonality test for the given time series data.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The seasonality of the time series data.

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
                * - alpha
                  - the criterion for the autocorrelation coefficient, it is optional
                * - decompose_type
                  - the type of decomposition chosen from {'additive', 'multiplicative', 'auto'}, it is optional
                * - extrapolation
                  - whether to extrapolate the endpoints or not, it is optional
                * - smooth_width
                  - the width of the smoothing window, it is optional
                * - auxiliary_normalitytest
                  - specifies whether to use normality test to identify model types, it is optional
                * - periods
                  - the length of the periods, it is optional
                * - decompose_method
                  - the method of decomposition chosen from {'stl', 'traditional'}, it is optional
                * - stl_robust
                  - whether to use robust decomposition or not only valid for 'stl' decompose method, it is optional
                * - stl_seasonal_average
                  - whether to use seasonal average or not only valid for 'stl' decompose method, it is optional
                * - smooth_method_non_seasonal
                  - the method of smoothing for non-seasonal component chosen from {'moving_average', 'super_smoother'}, it is optional
    """
    name: str = "seasonality_test"
    """Name of the tool."""
    description: str = "To check the seasonality of the time series data."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = SeasonalityTestInput
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
        table_name: str,
        key: str,
        endog: str,
        alpha: Optional[float] = None,
        decompose_type: Optional[str] = None,
        extrapolation: Optional[bool] = None,
        smooth_width: Optional[int] = None,
        auxiliary_normalitytest: Optional[bool] = None,
        periods: Optional[int] = None,
        decompose_method: Optional[str] = None,
        stl_robust: Optional[bool] = None,
        stl_seasonal_average: Optional[bool] = None,
        smooth_method_non_seasonal: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        df = self.connection_context.table(table_name).select(key, endog)
        result = seasonal_decompose(data=df,
                                    key=key,
                                    endog=endog,
                                    alpha=alpha,
                                    decompose_type=decompose_type,
                                    extrapolation=extrapolation,
                                    smooth_width=smooth_width,
                                    auxiliary_normalitytest=auxiliary_normalitytest,
                                    periods=periods,
                                    decompose_method=decompose_method,
                                    stl_robust=stl_robust,
                                    stl_seasonal_average=stl_seasonal_average,
                                    smooth_method_non_seasonal=smooth_method_non_seasonal)[0].collect()
        analysis_result = {}
        for _, row in result.iterrows():
            analysis_result[row['STAT_NAME']] = row['STAT_VALUE']
        return json.dumps(analysis_result, cls=_CustomEncoder)

    async def _arun(
        self,
        table_name: str,
        key: str,
        endog: str,
        alpha: Optional[float] = None,
        decompose_type: Optional[str] = None,
        extrapolation: Optional[bool] = None,
        smooth_width: Optional[int] = None,
        auxiliary_normalitytest: Optional[bool] = None,
        periods: Optional[int] = None,
        decompose_method: Optional[str] = None,
        stl_robust: Optional[bool] = None,
        stl_seasonal_average: Optional[bool] = None,
        smooth_method_non_seasonal: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(table_name, key, endog, alpha, decompose_type, extrapolation, smooth_width, auxiliary_normalitytest, periods, decompose_method, stl_robust, stl_seasonal_average, smooth_method_non_seasonal, run_manager=run_manager)

class WhiteNoiseTest(BaseTool):
    """
    This tool calls white noise test for the given time series data.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The white noise statistics of the time series data.

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
                * - lag
                  - specifies the lag autocorrelation coefficient that the statistic will be based on, it is optional
                * - probability
                  - the confidence level used for chi-square distribution., it is optional
                * - model_df
                  - the degree of freedom of the model, it is optional
    """
    name: str = "white_noise_test"
    """Name of the tool."""
    description: str = "To check the white noise of the time series data."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = WhiteNoiseTestInput
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
        table_name: str,
        key: str,
        endog: str,
        lag: Optional[int] = None,
        probability: Optional[float] = None,
        model_df: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        df = self.connection_context.table(table_name).select(key, endog)
        result = white_noise_test(data=df,
                                  key=key,
                                  endog=endog,
                                  lag=lag,
                                  probability=probability,
                                  model_df=model_df).collect()
        analysis_result = {}
        for _, row in result.iterrows():
            analysis_result[row['STAT_NAME']] = row['STAT_VALUE']
        return json.dumps(analysis_result, cls=_CustomEncoder)

    async def _arun(
        self,
        table_name: str,
        key: str,
        endog: str,
        lag: Optional[int] = None,
        probability: Optional[float] = None,
        model_df: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(table_name, key, endog, lag, probability, model_df, run_manager=run_manager)
