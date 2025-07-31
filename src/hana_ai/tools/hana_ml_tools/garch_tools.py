"""
The module defines some agent tools for GARCH class in hana-ml.
"""
import json
import logging
from typing import Type
from pydantic import BaseModel, Field, ConfigDict

from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.algorithms.pal.tsa.garch import GARCH
from hdbcli.dbapi import ProgrammingError
from hana_ai.utility import remove_prefix_sharp

logger = logging.getLogger(__name__)

class GARCHInput(BaseModel):
    """
    Class for GARCH parameters (inclusive of fit and predict).
    """
    table_name : str = Field(description="The table (or view) containing the data to fit a GARCH model and forecast" +\
    " the future data variance (risk). If not provided, ask the user, do not guess")
    # init args
    key : str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog : str = Field(description="the column of sequence data to fit the GARCH model and make future predictions." +\
    " If not provided, ask the user. Do not guess.")
    p : int = Field(description="Specifies the number of lagged error terms in GARCH model", default=None)
    q : int = Field(description="Specifies the number of lagged variance terms in GARCH model", default=None)
    model_type : str = Field(description="Specifies the variant of GARCH model, including 'garch', " +\
    "'igarch', 'tgarch' and 'egarch'", default=None)
    thread_ratio : float = Field(description="The ratio of available threads used to fit the GARCH model", default=None)
    # predict args
    forecast_length : int = Field(description="The length of future steps to forecast", default=None)
    model_config = ConfigDict(protected_namespaces=())

class GARCHFitPredict(BaseTool):
    r"""
    This tool is used to fit a GARCH model using the input data and then forecast its future variances.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The result string containing the forecast result table name and statistics.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - table_name
                  - The table (or view) containing the sequence data to fit a GARCH model and make future forecasts on data's volatility.
                * - key
                  - The key of the dataset. If not provided, ask the user. Do not guess.
                * - endog
                  - The column of sequence data to fit a GARCH model and make future forecasts on data volatility.
                    If not provided, ask the user. Do not guess
                * - p
                  - Specifies the number of lagged error terms in GARCH model.
                * - q
                  - Specifies the number of lagged variance terms in GARCH model.
                * - model_type
                  - Specifies the variant of GARCH model, including 'garch', 'igarch', 'tgarch' and 'egarch'.
                * - thread_ratio
                  - The ratio of available threads used to fit the GARCH model.
                * - forecast_length
                  - The length of future steps to forecast.
    """
    name: str = "garch_fit_predict"
    """Name of the tool."""
    description: str = "To fit a GARCH model on a dataset and then forecast its future volatility."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = GARCHInput
    return_direct: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct : bool = False
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )

    def _run(self,#pylint:disable=too-many-return-statements
             **kwargs) -> str:
        """Use the tool."""
        if "kwargs" in kwargs:
            kwargs = kwargs.get("kwargs")
        table_name = kwargs.get("table_name", None)
        if table_name is None:
            return "Input table name is required"
        key = kwargs.get("key", None)
        if key is None:
            return "key is required"
        endog = kwargs.get("endog", None)
        if endog is None:
            return "Endogenous variable (`endog`) is required"
        p = kwargs.get("p", None)
        q = kwargs.get("q", None)
        model_type = kwargs.get("model_type", None)
        thread_ratio = kwargs.get("thread_ratio", None)
        forecast_length = kwargs.get("forecast_length", None)
        try:
            garch = GARCH(p=p, q=q, model_type=model_type)
            garch.fit(data=self.connection_context.table(table_name)[[key, endog]],
                      key=key, endog=endog,
                      thread_ratio=thread_ratio)
            out_tabs = garch.predict(horizon=forecast_length)
        except ValueError as verr:
            return 'ValueError occurred: ' + str(verr)
        except TypeError as terr:
            return 'TypeError occurred: ' + str(terr)
        except KeyError as kerr:
            return 'KeyError occurred: ' + str(kerr)
        except ProgrammingError as perr:
            if 'invalid table name' in str(perr):
                return f'Invalid table name: Could not find table/view {table_name}'
        predict_result = remove_prefix_sharp(f"{table_name}_PREDICT_RESULT")
        out_tabs[0].save(predict_result, force=True)
        out_dict = {"garch_predict_result_table" : predict_result}
        if out_tabs[1].count() > 0:
            for _, row in out_tabs[1].collect().iterrows():
                out_dict[row["STAT_NAME"]] = row["STAT_VALUE"]
        return json.dumps(out_dict)

    async def _arun(self, **kwargs) -> str:
        """The asynchronous run"""
        return self._run(**kwargs)
