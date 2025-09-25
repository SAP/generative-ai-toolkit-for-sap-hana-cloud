"""
Create Predict Table for Time Series

The following classes are available:

    * :class `TSMakeFutureTableTool`
    * :class `TSMakeFutureTableForMassiveForecastTool`
"""

import logging
from typing import Optional, Type
import uuid
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from hana_ml import ConnectionContext

logger = logging.getLogger(__name__)


def make_future_dataframe(data, key=None, periods=1, increment_type='seconds'):
    """
    Create a new dataframe for time series prediction.

    Parameters
    ----------
    data : DataFrame, optional
        The training data contains the index.

        Defaults to the data used in the fit().

    key : str, optional
        The index defined in the training data.

        Defaults to the specified key in fit function or the data.index or the first column of the data.

    periods : int, optional
        The number of rows created in the predict dataframe.

        Defaults to 1.

    increment_type : {'seconds', 'days', 'months', 'years'}, optional
        The increment type of the time series.

        Defaults to 'seconds'.

    Returns
    -------
    DataFrame

    """

    if key is None:
        if data.index is None:
            key = data.columns[0]
        else:
            key = data.index
    max_ = data.select(key).max()
    sec_max_ = data.select(key).distinct().sort_values(key, ascending=False).head(2).collect().iat[1, 0]
    delta = max_ - sec_max_
    is_int = 'INT' in data.get_table_structure()[key]
    if is_int:
        forecast_start, timedelta = max_ + delta, delta
    else:
        forecast_start, timedelta = max_ + delta, delta.total_seconds()
    timeframe = []
    if not is_int:
        if 'day' in increment_type.lower():
            increment_type = 'days'
            timedelta = round(timedelta / 86400)
            if timedelta == 0:
                raise ValueError("The interval between the training time series is less than one day.")
        elif 'month' in increment_type.lower():
            increment_type = 'months'
            timedelta = round(timedelta / 2592000)
            if timedelta == 0:
                raise ValueError("The interval between the training time series is less than one month.")
        elif 'year' in increment_type.lower():
            increment_type = 'years'
            timedelta = round(timedelta / 31536000)
            if timedelta == 0:
                raise ValueError("The interval between the training time series is less than one year.")
        else:
            increment_type = 'seconds'
    for period in range(0, periods):
        if is_int:
            timeframe.append("SELECT TO_INT({} + {} * {}) AS \"{}\" FROM DUMMY".format(forecast_start, timedelta, period, key))
        else:
            timeframe.append("SELECT ADD_{}('{}', {} * {}) AS \"{}\" FROM DUMMY".format(increment_type.upper(), forecast_start, timedelta, period, key))
    sql = ' UNION ALL '.join(timeframe)
    return data.connection_context.sql(sql).sort_values(key)

def make_future_dataframe_for_massive_forecast(data=None, key=None, group_key=None, periods=1, increment_type='seconds'):
    """
    Create a new dataframe for time series prediction.

    Parameters
    ----------
    data : DataFrame, optional
        The training data contains the index.

        Defaults to the data used in the fit().

    key : str, optional
        The index defined in the training data.

        Defaults to the specified key in fit() or the value in data.index or the PAL's default key column position.

    group_key : str, optional
        Specify the group id column.

        This parameter is only valid when ``massive`` is True.

        Defaults to the specified group_key in fit() or the first column of the dataframe.

    periods : int, optional
        The number of rows created in the predict dataframe.

        Defaults to 1.

    increment_type : {'seconds', 'days', 'months', 'years'}, optional
        The increment type of the time series.

        Defaults to 'seconds'.

    Returns
    -------
    DataFrame

    """
    if group_key is None:
        group_key = data.columns[0]
    if key is None:
        if data.index is None:
            key = data.columns[1]
        else:
            key = data.index
    group_id_type = data.get_table_structure()[group_key]
    group_list = data.select(group_key).distinct().collect()[group_key]
    timeframe = []
    for group in group_list:
        if 'INT' in group_id_type.upper():
            m_data = data.filter(f"{group_key}={group}")
        else:
            m_data = data.filter(f"{group_key}='{group}'")
        max_ = m_data.select(key).max()
        sec_max_ = m_data.select(key).distinct().sort_values(key, ascending=False).head(2).collect().iat[1, 0]
        delta = max_ - sec_max_
        is_int = 'INT' in m_data.get_table_structure()[key]
        if is_int:
            forecast_start, timedelta = max_ + delta, delta
        else:
            forecast_start, timedelta = max_ + delta, delta.total_seconds()

        if not is_int:
            if 'day' in increment_type.lower():
                increment_type = 'days'
                timedelta = round(timedelta / 86400)
                if timedelta == 0:
                    raise ValueError("The interval between the training time series is less than one day.")
            elif 'month' in increment_type.lower():
                increment_type = 'months'
                timedelta = round(timedelta / 2592000)
                if timedelta == 0:
                    raise ValueError("The interval between the training time series is less than one month.")
            elif 'year' in increment_type.lower():
                increment_type = 'years'
                timedelta = round(timedelta / 31536000)
                if timedelta == 0:
                    raise ValueError("The interval between the training time series is less than one year.")
            else:
                increment_type = 'seconds'

        increment_type = increment_type.upper()
        for period in range(0, periods):
            if 'INT' in group_id_type.upper():
                if is_int:
                    timeframe.append(f"SELECT {group} AS \"{group_key}\", TO_INT({forecast_start} + {timedelta} * {period}) AS \"{key}\" FROM DUMMY")
                else:
                    timeframe.append(f"SELECT {group} AS \"{group_key}\", ADD_{increment_type}('{forecast_start}', {timedelta} * {period}) AS \"{key}\" FROM DUMMY")
            else:
                if is_int:
                    timeframe.append(f"SELECT '{group}' AS \"{group_key}\", TO_INT({forecast_start} + {timedelta} * {period}) AS \"{key}\" FROM DUMMY")
                else:
                    timeframe.append(f"SELECT '{group}' AS \"{group_key}\", ADD_{increment_type}('{forecast_start}', {timedelta} * {period}) AS \"{key}\" FROM DUMMY")
    sql = ' UNION ALL '.join(timeframe)

    return data.connection_context.sql(sql).sort_values([group_key, key])

class MakeFutureTableToolInput(BaseModel):
    """
    The input schema for the TSMakeFutureTableTool.
    """
    train_table: str = Field(description="The name of the training table in HANA")
    train_schema: Optional[str] = Field(default=None, description="The schema of the training table, it is optional")
    key: Optional[str] = Field(default=None, description="The index defined in the training data.")
    periods: Optional[int] = Field(default=1, description="The number of rows created in the predict dataframe.")
    increment_type: Optional[str] = Field(default='seconds', description="The increment type of the time series. Options are 'seconds', 'days', 'months', 'years'.")
    predict_table: Optional[str] = Field(default=None, description="The name of the target table to store the predict dataframe in HANA")

class MakeFutureTableForMassiveForecastToolInput(BaseModel):
    """
    The input schema for the TSMakeFutureTableForMassiveForecastTool.
    """
    train_table: str = Field(description="The name of the training table in HANA")
    train_schema: Optional[str] = Field(default=None, description="The schema of the training table, it is optional")
    key: Optional[str] = Field(default=None, description="The index defined in the training data.")
    group_key: Optional[str] = Field(default=None, description="Specify the group id column.")
    periods: Optional[int] = Field(default=1, description="The number of rows created in the predict dataframe.")
    increment_type: Optional[str] = Field(default='seconds', description="The increment type of the time series. Options are 'seconds', 'days', 'months', 'years'.")
    predict_table: Optional[str] = Field(default=None, description="The name of the target table to store the predict dataframe in HANA")

class TSMakeFutureTableTool(BaseTool):
    """
    This tool creates a predict table for time series forecasting in HANA.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        Operation result message

    """
    name: str = "ts_make_future_table"
    """Name of the tool."""
    description: str = "Create a predict table for time series forecasting in HANA."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = MakeFutureTableToolInput
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
        self, **kwargs
    ) -> str:
        """Use the tool."""
        # 从kwargs字典中提取参数
        train_table = kwargs.get('train_table')
        train_schema = kwargs.get('train_schema', None)
        key = kwargs.get('key', None)
        periods = kwargs.get('periods', 1)
        increment_type = kwargs.get('increment_type', 'seconds')
        # predict_table_gen from uuid
        uuid_str = str(uuid.uuid4()).replace('-', '_').upper()
        predict_table_gen = f"#FUTURE_INPUT_TABLE_{uuid_str}"
        predict_table = kwargs.get('predict_table', predict_table_gen)
        try:
            # 读取训练数据
            train_data = self.connection_context.table(train_table, schema=train_schema)
            # 创建预测数据
            predict_data = make_future_dataframe(train_data, key, periods, increment_type)
            # 将预测数据保存到HANA表中
            predict_data.smart_save(predict_table)
            return f"Successfully created the forecast input table '{predict_table}' with {periods} rows."
        except Exception as e:
            logger.error("Error creating the forecast input table: %s", str(e))
            return f"Operation failed: {str(e)}"

    async def _arun(
        self, **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs)

class TSMakeFutureTableForMassiveForecastTool(BaseTool):
    """
    This tool creates a predict table for massive time series forecasting in HANA.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        Operation result message

    """
    name: str = "ts_make_future_table_for_massive_forecast"
    """Name of the tool."""
    description: str = "Create a predict table for massive time series forecasting in HANA."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = MakeFutureTableForMassiveForecastToolInput
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
        self, **kwargs
    ) -> str:
        """Use the tool."""
        # 从kwargs字典中提取参数
        train_table = kwargs.get('train_table')
        train_schema = kwargs.get('train_schema', None)
        key = kwargs.get('key', None)
        group_key = kwargs.get('group_key', None)
        periods = kwargs.get('periods', 1)
        increment_type = kwargs.get('increment_type', 'seconds')
        # predict_table_gen from uuid
        uuid_str = str(uuid.uuid4()).replace('-', '_').upper()
        predict_table_gen = f"#FUTURE_INPUT_TABLE_{uuid_str}"
        predict_table = kwargs.get('predict_table', predict_table_gen)
        try:
            # 读取训练数据
            train_data = self.connection_context.table(train_table, schema=train_schema)
            # 创建预测数据
            predict_data = make_future_dataframe_for_massive_forecast(train_data, key, group_key, periods, increment_type)
            # 将预测数据保存到HANA表中
            predict_data.smart_save(predict_table)
            return f"Successfully created the forecast input table '{predict_table}' with {periods} rows per group."
        except Exception as e:
            logger.error("Error creating the forecast input table: %s", str(e))
            return f"Operation failed: {str(e)}"

    async def _arun(
        self, **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs)
