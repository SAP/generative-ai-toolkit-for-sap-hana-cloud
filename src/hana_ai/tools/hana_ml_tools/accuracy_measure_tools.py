"""
This module define the agent tools for `accuracy_measure` function in hana-ml.
"""
import json
import logging
from typing import Type, List, Union
from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.algorithms.pal.tsa.accuracy_measure import accuracy_measure

logger = logging.getLogger(__name__)

class AccuracyMeasureInput(BaseModel):
    """Class of input arguments for accuracy measure"""
    input_table : str = Field(description="Table (or view) containing the input data for computing the accuracy measure, should have at least 3 columns:" +\
    "ID, real data and forecast data. If not provided, ask the user, do not guess")
    evaluation_metric : Union[str, List[str]] = Field(description="Specifies the accuracy measures to compute, it could be one or a list of the" +\
    " following options : 'mpe', 'mse', 'rmse', 'et', 'mad', 'mase', 'wmape', 'smape', 'mape' and 'spec'." + \
    " If not provided, ask the user, do not guess")
    key : str = Field(description="Specifies the name of the ID column in input_table. If not provided," +\
    " the 1st column of input_table will be used.", default=None)
    real_col : str = Field(description="Name of the column that contains the real data in input_table." +\
    " If not provided, name of the 1st non-ID column of input_table will be used", default=None)
    forecast_col : str = Field(description="Name of the column that contains the forecast data in input_table." +\
    " If not provided, name of the 2nd non-ID column of input_table will be used", default=None)
    ignore_zero : bool = Field(description="Specifies whether or not to ignore zero values when calculating accuracy measure 'mpe' or 'mape', it is optional", default=None)#pylint:disable=line-too-long
    alpha2 : float = Field(description="Specifies the unit stock-keeping cost parameter of accuracy measure 'spec'", default=None)#pylint:disable=line-too-long
    alpha1 : float = Field(description="Specifies unit opportunity cost parameter of accuracy measure 'spec'", default=None)#pylint:disable=line-too-long

class AccuracyMeasure(BaseTool):
    r"""
    This tool calculate the specified accuracy measures using true and predicted data.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The result string containing the training table name, model storage name, and model storage version.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - name
                  - The name of the model in model storage.
                * - version
                  - The version of the model in model storage.
                * - fit_table
                  - The table to fit the model.
    """
    name : str = "accuracy_measure"
    """Name of the tool."""
    description : str = "To compute the accuracy measure using true and predict values."
    """Description of the tool."""
    connection_context : ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = AccuracyMeasureInput
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
        evaluation_metric : Union[str, List[str]],
        key : str=None,
        real_col : str=None,
        forecast_col : str=None,
        ignore_zero : bool=None,
        alpha1 : float=None,
        alpha2 : float=None,
        run_manager: CallbackManagerForToolRun=None#pylint:disable=unused-argument
        )-> str:
        input_data = self.connection_context.table(input_table)
        table_type = len(input_data.columns)
        if isinstance(evaluation_metric, str):
            evaluation_metric = [evaluation_metric]
        if 'spec' in evaluation_metric and table_type < 3:
            msg = "When 'spec' is specified as the evaluation metric, input_table should" +\
            "have at least 3 columns, and the ID column must present"
            raise ValueError(msg)
        if table_type == 2:
            input_data = input_data.add_id("ACC_MS_GEN_ID")
        key = input_data.columns[0] if key is None else key
        for in_col in [key, real_col, forecast_col]:
            if in_col is not None and in_col not in input_data.columns:
                msg = f"Column {in_col} not found in table {input_table}"
                raise ValueError(msg)
        input_cols = input_data.columns
        input_cols.remove(key)
        real_col = input_cols[0] if real_col is None else real_col
        forecast_col = input_cols[1] if forecast_col is None else forecast_col
        accm_res = accuracy_measure(data=input_data[[key, real_col, forecast_col]],
                                    evaluation_metric=evaluation_metric,
                                    ignore_zero=ignore_zero,
                                    alpha1=alpha1,
                                    alpha2=alpha2)
        out_dict = {}
        for _, row in accm_res.collect().iterrows():
            out_dict[row['STAT_NAME']] = row['STAT_VALUE']
        return json.dumps(out_dict)

    async def _arun(#pylint:disable=too-many-positional-arguments
        self,
        input_table : str,
        evaluation_metric : Union[str, List[str]],
        key : str=None,
        real_col : str=None,
        forecast_col : str=None,
        ignore_zero : bool=None,
        alpha1 : float=None,
        alpha2 : float=None,
        run_manager: AsyncCallbackManagerForToolRun=None#pylint:disable=unused-argument
        ) -> str:
        return self._run(input_table=input_table,
                         evaluation_metric=evaluation_metric,
                         key=key,
                         real_col=real_col,
                         forecast_col=forecast_col,
                         ignore_zero=ignore_zero,
                         alpha1=alpha1,
                         alpha2=alpha2,
                         run_manager=run_manager)
