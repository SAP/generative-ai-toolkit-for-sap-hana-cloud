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

from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.algorithms.pal.tsa.stationarity_test import stationarity_test
from hana_ml.algorithms.pal.tsa.trend_test import trend_test
from hana_ml.algorithms.pal.tsa.seasonal_decompose import seasonal_decompose
from hana_ml.algorithms.pal.tsa.white_noise_test import white_noise_test

from hana_ai.tools.hana_ml_tools.utility import _CustomEncoder

logger = logging.getLogger(__name__)

INTERMITTENT_ADI_THRESHOLD = 1.32
INTERMITTENT_CV2_THRESHOLD = 0.49


def _format_metric(value):
    if value is None:
        return "n/a"
    if value == float("inf"):
        return "inf"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _analyze_intermittent_demand(df, key, endog):
    """Classify intermittent demand with the common ADI/CV^2 method."""
    ordered = df.select(key, endog).sort_values(key).collect()
    values = ordered[endog].tolist()
    total_values = len(values)

    if total_values == 0:
        return {
            "zero_proportion": 1.0,
            "occurrence_rate": 0.0,
            "occurrences": 0,
            "adi": float("inf"),
            "cv2": float("inf"),
            "classification": "empty",
            "is_intermittent": False,
        }

    non_zero_values = [value for value in values if value != 0]
    occurrences = len(non_zero_values)
    zero_proportion = (total_values - occurrences) / total_values
    occurrence_rate = occurrences / total_values

    if occurrences == 0:
        return {
            "zero_proportion": zero_proportion,
            "occurrence_rate": occurrence_rate,
            "occurrences": 0,
            "adi": float("inf"),
            "cv2": 0.0,
            "classification": "all_zero",
            "is_intermittent": True,
        }

    adi = total_values / occurrences
    mean_non_zero = sum(non_zero_values) / occurrences
    if occurrences == 1:
        cv2 = 0.0
    else:
        variance = sum((value - mean_non_zero) ** 2 for value in non_zero_values) / occurrences
        cv2 = float("inf") if mean_non_zero == 0 else variance / (mean_non_zero ** 2)

    if adi >= INTERMITTENT_ADI_THRESHOLD and cv2 >= INTERMITTENT_CV2_THRESHOLD:
        classification = "lumpy"
    elif adi >= INTERMITTENT_ADI_THRESHOLD:
        classification = "intermittent"
    elif cv2 >= INTERMITTENT_CV2_THRESHOLD:
        classification = "erratic"
    else:
        classification = "smooth"

    return {
        "zero_proportion": zero_proportion,
        "occurrence_rate": occurrence_rate,
        "occurrences": occurrences,
        "adi": adi,
        "cv2": cv2,
        "classification": classification,
        "is_intermittent": classification in {"intermittent", "lumpy", "all_zero"},
    }


def _format_intermittent_result(intermittent_metrics):
    return (
        "Intermittent Test (ADI/CV^2): "
        f"non-zero occurrence rate is {_format_metric(intermittent_metrics['occurrence_rate'])}, "
        f"zero proportion is {_format_metric(intermittent_metrics['zero_proportion'])}, "
        f"occurrences are {intermittent_metrics['occurrences']}, "
        f"average demand interval (ADI) is {_format_metric(intermittent_metrics['adi'])}, "
        f"CV^2 of non-zero demand sizes is {_format_metric(intermittent_metrics['cv2'])}, "
        f"classification is {intermittent_metrics['classification']}, "
        f"intermittent={intermittent_metrics['is_intermittent']}"
    )


def _get_model_recommendation(intermittent_metrics):
    classification = intermittent_metrics["classification"]

    if classification == "intermittent":
        return {
            "available_algorithms": ["Intermittent Forecast", "Automatic Time Series Forecast"],
            "message": (
                "Model Recommendation: prefer Intermittent Forecast with method=constant. "
                "ADI >= 1.32 and CV^2 < 0.49 indicate sparse demand occurrence with relatively stable "
                "non-zero demand sizes. Alternative: Automatic Time Series Forecast."
            ),
        }
    if classification == "lumpy":
        return {
            "available_algorithms": ["Intermittent Forecast", "Automatic Time Series Forecast"],
            "message": (
                "Model Recommendation: prefer Intermittent Forecast with method=sporadic. "
                "ADI >= 1.32 and CV^2 >= 0.49 indicate sparse demand occurrence with highly variable "
                "non-zero demand sizes. Alternative: Automatic Time Series Forecast."
            ),
        }
    if classification == "all_zero":
        return {
            "available_algorithms": ["Intermittent Forecast", "Automatic Time Series Forecast"],
            "message": (
                "Model Recommendation: validate whether the series is structurally zero before training. "
                "If forecasting is still required, start with Intermittent Forecast and keep "
                "Automatic Time Series Forecast only as a fallback after data review."
            ),
        }
    if classification == "erratic":
        return {
            "available_algorithms": ["Additive Model Forecast", "Automatic Time Series Forecast"],
            "message": (
                "Model Recommendation: prefer Automatic Time Series Forecast. ADI < 1.32 and CV^2 >= 0.49 "
                "indicate frequent demand occurrence but volatile non-zero demand sizes. Alternative: "
                "Additive Model Forecast when you want explicit seasonality and trend control."
            ),
        }
    if classification == "empty":
        return {
            "available_algorithms": ["Intermittent Forecast", "Additive Model Forecast", "Automatic Time Series Forecast"],
            "message": (
                "Model Recommendation: the series is empty, so any model choice is provisional until data is available."
            ),
        }
    return {
        "available_algorithms": ["Additive Model Forecast", "Automatic Time Series Forecast"],
        "message": (
            "Model Recommendation: prefer Automatic Time Series Forecast as the default non-intermittent path. "
            "ADI < 1.32 and CV^2 < 0.49 indicate regular demand occurrence with stable non-zero sizes. "
            "Alternative: Additive Model Forecast when you want explicit seasonality and trend control."
        ),
    }

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

    # Table info
    table_struct = json.dumps(df.get_table_structure())
    analysis_result += f"Table structure: {table_struct}\n"
    analysis_result += f"Key: {key}\n"
    analysis_result += f"Endog: {endog}\n"

    # Index info
    analysis_result += f"Index: starts from {df[key].min()} to {df[key].max()}. Time series length is {df.count()}\n"

    key_col_type = df.get_table_structure()[key]
    key_ = key
    df_ = df
    if 'INT' not in key_col_type.upper():
        key_ = "NEW_" + key
        df_ = df.add_id(key_, ref_col=key)

    # Intermittent Test
    intermittent_metrics = _analyze_intermittent_demand(df_, key_, endog)
    analysis_result += _format_intermittent_result(intermittent_metrics) + "\n"

    # Stationarity Test
    result = stationarity_test(df_, key_, endog).collect()
    analysis_result += "Stationarity Test: "
    for _, row in result.iterrows():
        analysis_result += f"The `{row['STATS_NAME']}` is {row['STATS_VALUE']}."
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
        analysis_result += f"The `{row['STAT_NAME']}` is {row['STAT_VALUE']}."
    analysis_result += "\n"

    model_recommendation = _get_model_recommendation(intermittent_metrics)
    analysis_result += model_recommendation["message"] + "\n"

    # Restrict time series algorithms
    available_algorithms = model_recommendation["available_algorithms"]
    analysis_result += f"Available algorithms: {', '.join(available_algorithms)}\n"

    return analysis_result

def ts_char_massive(df, group_key, key, endog):
    """
    This function is used to get the characteristics of multiple time series data grouped by group_key.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    group_key : str
        The column used to group multiple time series.
    key : str
        The key column (time index) of the DataFrame.
    endog : str
        The endogenous column of the DataFrame.
    """
    # 获取所有分组
    groups = df.select(group_key).distinct().collect()[group_key].to_list()
    analysis_result = f"Time Series Analysis Report ({len(groups)} groups)\n"
    analysis_result += "=" * 60 + "\n\n"

    # 遍历每个分组
    for i, group_val in enumerate(groups):
        if isinstance(group_val, str):
            df_group = df.filter(f'"{group_key}" = \'{group_val}\'')
        else:
            df_group = df.filter(f'"{group_key}" = {group_val}')

        analysis_result += f"Group {i+1}/{len(groups)}: {group_key} = {group_val}\n"
        analysis_result += "-" * 60 + "\n"

        # 表结构信息
        table_struct = json.dumps(df_group.get_table_structure())
        analysis_result += f"• Table structure: {table_struct}\n"
        analysis_result += f"• Key: {key}\n"
        analysis_result += f"• Endog: {endog}\n"

        # 索引信息
        analysis_result += f"Index: starts from {df_group[key].min()} to {df_group[key].max()}. Time series length is {df_group.count()}\n"

        # 处理非整数类型的时间键
        key_col_type = df_group.get_table_structure()[key]
        key_ = key
        df_ = df_group
        if 'INT' not in key_col_type.upper():
            key_ = "NEW_" + key
            df_ = df_group.add_id(key_, ref_col=key)

        # Intermittent Test
        intermittent_metrics = _analyze_intermittent_demand(df_, key_, endog)
        analysis_result += _format_intermittent_result(intermittent_metrics) + "\n"

        # Stationarity Test
        result = stationarity_test(df_, key_, endog).collect()
        analysis_result += "Stationarity Test: "
        for _, row in result.iterrows():
            analysis_result += f"The `{row['STATS_NAME']}` is {row['STATS_VALUE']}."
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
            analysis_result += f"The `{row['STAT_NAME']}` is {row['STAT_VALUE']}."
        analysis_result += "\n"

        model_recommendation = _get_model_recommendation(intermittent_metrics)
        analysis_result += model_recommendation["message"] + "\n"

        # Restrict time series algorithms
        available_algorithms = model_recommendation["available_algorithms"]
        analysis_result += f"Available algorithms: {', '.join(available_algorithms)}\n"

    return analysis_result

class TSCheckInput(BaseModel):
    """
    The input schema for the TimeSeriesCheckTool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    schema_name: Optional[str] = Field(description="the schema_name of the table, it is optional", default=None)

class MassiveTSCheckInput(BaseModel):
    """
    The input schema for the TimeSeriesCheckTool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    group_key: str = Field(description="the group key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    schema_name: Optional[str] = Field(description="the schema_name of the table, it is optional", default=None)

class StationarityTestInput(BaseModel):
    """
    The input schema for the StationarityTestTool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    schema_name: Optional[str] = Field(description="the schema_name of the table, it is optional", default=None)
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
    schema_name: Optional[str] = Field(description="the schema_name of the table, it is optional", default=None)
    method: Optional[str] = Field(description="the method of the trend test chosen from {'mk', 'difference-sign'}, it is optional", default=None)
    alpha: Optional[float] = Field(description="the significance level for the trend test, it is optional", default=None)

class SeasonalityTestInput(BaseModel):
    """
    The input schema for the SeasonalityTestTool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    schema_name: Optional[str] = Field(description="the schema_name of the table, it is optional", default=None)
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
    schema_name: Optional[str] = Field(description="the schema_name of the table, it is optional", default=None)
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
                * - schema_name
                  - the schema_name of the table, it is optional
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
        self,
        **kwargs
    ) -> str:
        """Use the tool."""

        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        table_name = kwargs.get("table_name", None)
        if table_name is None:
            return "Table name is required"
        key = kwargs.get("key", None)
        if key is None:
            return "Key is required"
        endog = kwargs.get("endog", None)
        if endog is None:
            return "Endog is required"
        schema_name = kwargs.get("schema_name", None)
        # check table exists
        if not self.connection_context.has_table(table_name, schema=schema_name):
            return f"Table {table_name} does not exist."
        # check key and endog columns exist
        if key not in self.connection_context.table(table_name, schema=schema_name).columns:
            return f"Key column {key} does not exist in table {table_name}."
        if endog not in self.connection_context.table(table_name, schema=schema_name).columns:
            return f"Endog column {endog} does not exist in table {table_name}."
        df = self.connection_context.table(table_name, schema=schema_name).select(key, endog)
        return ts_char(df, key, endog)

    async def _arun(
        self, **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs)

class MassiveTimeSeriesCheck(BaseTool):
    """
    This tool performs time series analysis for multiple grouped time series data, 
    including stationarity, intermittency, trend and seasonality tests.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The characteristics report for multiple time series groups.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - table_name
                  - The name of the table. If not provided, ask the user. Do not guess.
                * - group_key
                  - The column used to group multiple time series. Required.
                * - key
                  - The time key column. If not provided, ask the user. Do not guess.
                * - endog
                  - The endogenous variable column. If not provided, ask the user. Do not guess.
                * - schema_name
                  - The schema name of the table (optional)
    """
    name: str = "massive_ts_check"
    """Name of the tool."""
    description: str = (
        "Performs comprehensive time series analysis per group(group_key Column), "
        "including stationarity, intermittency, trend and seasonality tests."
    )
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = MassiveTSCheckInput
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
        """Use the tool for massive time series analysis."""
        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]

        # Validate required parameters
        table_name = kwargs.get("table_name")
        if not table_name:
            return "Table name is required"

        group_key = kwargs.get("group_key")
        if not group_key:
            return "Group key is required for massive time series analysis"

        key = kwargs.get("key")
        if not key:
            return "Time key is required"

        endog = kwargs.get("endog")
        if not endog:
            return "Endogenous variable is required"

        schema_name = kwargs.get("schema_name")

        # Check table existence
        if not self.connection_context.has_table(table_name, schema=schema_name):
            return f"Table {table_name} does not exist."

        # Get table reference
        table = self.connection_context.table(table_name, schema=schema_name)

        # Validate columns exist
        required_columns = [group_key, key, endog]
        for col in required_columns:
            if col not in table.columns:
                return f"Column '{col}' does not exist in table {table_name}."

        # Select relevant columns
        df = table.select(group_key, key, endog)

        # Check if group_key has reasonable number of groups
        distinct_groups = df.select(group_key).distinct().count()
        if distinct_groups > 100:
            return (
                f"Too many groups ({distinct_groups}) for analysis. "
                "Consider filtering or using a different group key."
            )

        # Perform massive time series analysis
        return ts_char_massive(df, group_key, key, endog)

    async def _arun(
        self, **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs)

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
                * - schema_name
                  - the schema_name of the table, it is optional
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
        **kwargs
    ) -> str:
        """Use the tool."""

        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        table_name = kwargs.get("table_name", None)
        if table_name is None:
            return "Table name is required"
        key = kwargs.get("key", None)
        if key is None:
            return "Key is required"
        endog = kwargs.get("endog", None)
        if endog is None:
            return "Endog is required"
        schema_name = kwargs.get("schema_name", None)
        method = kwargs.get("method", None)
        mode = kwargs.get("mode", None)
        lag = kwargs.get("lag", None)
        probability = kwargs.get("probability", None)
        # check table exists
        if not self.connection_context.has_table(table_name, schema=schema_name):
            return f"Table {table_name} does not exist."
        # check key and endog columns exist
        if key not in self.connection_context.table(table_name, schema=schema_name).columns:
            return f"Key column {key} does not exist in table {table_name}."
        if endog not in self.connection_context.table(table_name, schema=schema_name).columns:
            return f"Endog column {endog} does not exist in table {table_name}."
        df = self.connection_context.table(table_name, schema=schema_name).select(key, endog)
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
        **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs)

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
                * - schema_name
                  - the schema_name of the table, it is optional
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
        **kwargs
    ) -> str:
        """Use the tool."""

        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        table_name = kwargs.get("table_name", None)
        if table_name is None:
            return "Table name is required"
        key = kwargs.get("key", None)
        if key is None:
            return "Key is required"
        endog = kwargs.get("endog", None)
        if endog is None:
            return "Endog is required"
        method = kwargs.get("method", None)
        alpha = kwargs.get("alpha", None)
        schema_name = kwargs.get("schema_name", None)
        if not self.connection_context.has_table(table_name, schema=schema_name):
            return f"Table {table_name} does not exist."
        # check key and endog columns exist
        if key not in self.connection_context.table(table_name, schema=schema_name).columns:
            return f"Key column {key} does not exist in table {table_name}."
        if endog not in self.connection_context.table(table_name, schema=schema_name).columns:
            return f"Endog column {endog} does not exist in table {table_name}."
        df = self.connection_context.table(table_name, schema=schema_name).select(key, endog)
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
        **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs)

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
                * - schema_name
                  - the schema_name of the table, it is optional
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
        **kwargs
    ) -> str:
        """Use the tool."""

        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        table_name = kwargs.get("table_name", None)
        if table_name is None:
            return "Table name is required"
        key = kwargs.get("key", None)
        if key is None:
            return "Key is required"
        endog = kwargs.get("endog", None)
        if endog is None:
            return "Endog is required"
        schema_name = kwargs.get("schema_name", None)
        alpha = kwargs.get("alpha", None)
        decompose_type = kwargs.get("decompose_type", None)
        extrapolation = kwargs.get("extrapolation", None)
        smooth_width = kwargs.get("smooth_width", None)
        auxiliary_normalitytest = kwargs.get("auxiliary_normalitytest", None)
        periods = kwargs.get("periods", None)
        decompose_method = kwargs.get("decompose_method", None)
        stl_robust = kwargs.get("stl_robust", None)
        stl_seasonal_average = kwargs.get("stl_seasonal_average", None)
        smooth_method_non_seasonal = kwargs.get("smooth_method_non_seasonal", None)
        if not self.connection_context.has_table(table_name, schema=schema_name):
            return f"Table {table_name} does not exist."
        # check key and endog columns exist
        if key not in self.connection_context.table(table_name, schema=schema_name).columns:
            return f"Key column {key} does not exist in table {table_name}."
        if endog not in self.connection_context.table(table_name, schema=schema_name).columns:
            return f"Endog column {endog} does not exist in table {table_name}."
        df = self.connection_context.table(table_name, schema=schema_name).select(key, endog)
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
        **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs)

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
                * - schema_name
                  - the schema_name of the table, it is optional
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
        **kwargs
    ) -> str:
        """Use the tool."""

        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        table_name = kwargs.get("table_name", None)
        if table_name is None:
            return "Table name is required"
        key = kwargs.get("key", None)
        if key is None:
            return "Key is required"
        endog = kwargs.get("endog", None)
        if endog is None:
            return "Endog is required"
        lag = kwargs.get("lag", None)
        probability = kwargs.get("probability", None)
        model_df = kwargs.get("model_df", None)
        schema_name = kwargs.get("schema_name", None)
        # check table exists
        if not self.connection_context.has_table(table_name, schema=schema_name):
            return f"Table {table_name} does not exist."
        # check key and endog columns exist
        if key not in self.connection_context.table(table_name, schema=schema_name).columns:
            return f"Key column {key} does not exist in table {table_name}."
        if endog not in self.connection_context.table(table_name, schema=schema_name).columns:
            return f"Endog column {endog} does not exist in table {table_name}."
        df = self.connection_context.table(table_name, schema=schema_name).select(key, endog)
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
        **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs)
