"""
This tool calculates the accuracy measure for the time series forecast.
It is used to evaluate the performance of the forecast model.

The following class is available:

    * :class `AccuracyMeasure`
"""

# build custom tool for accuracy measure from PAL function
import json
from typing import Type, List, Union
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.algorithms.pal.tsa.accuracy_measure import accuracy_measure

from hana_ai.tools.hana_ml_tools.utility import _CustomEncoder

class AccuracyMeasureInput(BaseModel):
    """
    Input schema for the AccuracyMeasure tool.
    """
    predict_table: str = Field(description="should be the table name of the prediction result. If not provided, ask the user, do not guess")
    actual_table: str = Field(description="should be the table name of the actual result. If not provided, ask the user, do not guess")
    predict_key: str = Field(description="should be the key column name of the prediction result. If not provided, ask the user, do not guess")
    actual_key: str = Field(description="should be the key column name of the actual result. If not provided, ask the user, do not guess")
    predict_target: str = Field(description="should be the target column name of the prediction result. If not provided, ask the user, do not guess")
    actual_target: str = Field(description="should be the target column name of the actual result. If not provided, ask the user, do not guess")
    evaluation_metric : Union[str,  List[str]] = Field(description="Specifies the accuracy measures to compute, it could be one or a list of the" +\
    " following options : 'mpe', 'mse', 'rmse', 'et', 'mad', 'mase', 'wmape', 'smape', 'mape' and 'spec'." + \
    " If not provided, ask the user, do not guess")
    ignore_zero : bool = Field(description="Specifies whether or not to ignore zero values when calculating accuracy measure 'mpe' or 'mape', it is optional", default=None)#pylint:disable=line-too-long
    alpha2 : float = Field(description="Specifies the unit stock-keeping cost parameter of accuracy measure 'spec'", default=None)#pylint:disable=line-too-long
    alpha1 : float = Field(description="Specifies unit opportunity cost parameter of accuracy measure 'spec'", default=None)#pylint:disable=line-too-long

class AccuracyMeasure(BaseTool):
    """
    This tool calculates the accuracy measure for the time series forecast.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The accuracy measure result in JSON format.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - predict_table
                  - The name of the prediction result table.
                * - actual_table
                  - The name of the actual result table.
                * - predict_key
                  - The key column name of the prediction result.
                * - actual_key
                  - The key column name of the actual result.
                * - predict_target
                  - The target column name of the prediction result.
                * - actual_target
                  - The target column name of the actual result.
                * - evaluation_metric
                  - The accuracy measures to compute. It could be one or a list of the following options: 'mpe', 'mse', 'rmse', 'et', 'mad', 'mase', 'wmape', 'smape', 'mape' and 'spec'.
                * - ignore_zero
                  - Specifies whether or not to ignore zero values when calculating accuracy measure 'mpe' or 'mape'. It is optional.
                * - alpha2
                  - Specifies the unit stock-keeping cost parameter of accuracy measure 'spec'.
                * - alpha1
                  - Specifies the unit opportunity cost parameter of accuracy measure 'spec'.                  
       
    """
    name : str = "accuracy_measure"
    """Name of the tool."""
    description : str = "To compute the accuracy measure using true and predict tables."
    """Description of the tool."""
    connection_context : ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = AccuracyMeasureInput
    return_direct: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False,
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

        predict_table = kwargs.get("predict_table", None)
        if predict_table is None:
            return "Prediction table is required"
        actual_table = kwargs.get("actual_table", None)
        if actual_table is None:
            return "Actual table is required"
        predict_key = kwargs.get("predict_key", None)
        if predict_key is None:
            return "Prediction key is required"
        actual_key = kwargs.get("actual_key", None)
        if actual_key is None:
            return "Actual key is required"
        predict_target = kwargs.get("predict_target", None)
        if predict_target is None:
            return "Prediction target is required"
        actual_target = kwargs.get("actual_target", None)
        if actual_target is None:
            return "Actual target is required"
        evaluation_metric = kwargs.get("evaluation_metric", None)
        if evaluation_metric is None:
            return "Evaluation metric is required"
        ignore_zero = kwargs.get("ignore_zero", None)
        alpha1 = kwargs.get("alpha1", None)
        alpha2 = kwargs.get("alpha2", None)
        err_msg = []
        # check table existence
        if not self.connection_context.has_table(predict_table):
            err_msg.append(f"predict_table error: Table {predict_table} does not exist.")
        if not self.connection_context.has_table(actual_table):
            err_msg.append(f"actual_table error: Table {actual_table} does not exist.")
        if len(err_msg) > 0:
            return "\n".join(err_msg)
        # check column existence
        err_msg = []
        if predict_key not in self.connection_context.table(predict_table).columns:
            err_msg.append(f"predict_key error: Column {predict_key} does not exist in table {predict_table}.")
        if actual_key not in self.connection_context.table(actual_table).columns:
            err_msg.append(f"actual_key error: Column {actual_key} does not exist in table {actual_table}.")
        if predict_target not in self.connection_context.table(predict_table).columns:
            err_msg.append(f"predict_target error: Column {predict_target} does not exist in table {predict_table}.")
        if actual_target not in self.connection_context.table(actual_table).columns:
            err_msg.append(f"actual_target error: Column {actual_target} does not exist in table {actual_table}.")
        if len(err_msg) > 0:
            return "\n".join(err_msg)

        m_actual_target = actual_target + "_actual"
        m_predict_target = predict_target + "_predict"
        m_actual_key = actual_key + "_actual"
        m_predict_key = predict_key + "_predict"
        m_actual_key_int = m_actual_key + "_int"
        prepared_input = self.connection_context.table(predict_table)\
            .rename_columns({predict_target: m_predict_target, predict_key: m_predict_key})\
                .join(self.connection_context.table(actual_table)\
                    .rename_columns({actual_target: m_actual_target, actual_key: m_actual_key}),\
                         f'"{m_actual_key}"="{m_predict_key}"')[[m_actual_key, m_actual_target, m_predict_target]]\
                             .cast({m_actual_target: 'DOUBLE', m_predict_target: 'DOUBLE'})\
                                 .add_id(m_actual_key_int, ref_col=m_actual_key)
        accm_res = accuracy_measure(data=prepared_input[[m_actual_key_int, m_actual_target, m_predict_target]],
                                    evaluation_metric=evaluation_metric,
                                    ignore_zero=ignore_zero,
                                    alpha1=alpha1,
                                    alpha2=alpha2)
        out_dict = {}
        for _, row in accm_res.collect().iterrows():
            out_dict[row['STAT_NAME']] = row['STAT_VALUE']
        return json.dumps(out_dict, cls=_CustomEncoder)

    async def _arun(
        self,
        **kwargs,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(
            **kwargs
        )
