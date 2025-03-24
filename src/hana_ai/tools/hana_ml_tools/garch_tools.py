"""
The module defines some agent tools for GARCH class in hana-ml.
"""
import json
import logging
from typing import Type
from pydantic import BaseModel, Field, ConfigDict

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.model_storage import ModelStorage
from hana_ml.algorithms.pal.tsa.garch import GARCH
from hana_ai.utility import remove_prefix_sharp

logger = logging.getLogger(__name__)

class GARCHFitInput(BaseModel):
    """
    Class for GARCH initialize & fit parameters.
    """
    name : str = Field(description="The name of the model in model storage. If not provided, ask the user, do not guess")
    version : int = Field(description="the version of the model in model storage", default=None)
    fit_table : str = Field(description="The table (or view) containing the data to fit a GARCH model. If not provided, ask the user, do not guess")
    # init args
    p : int = Field(description="Specifies the number of lagged error terms in GARCH model", default=None)
    q : int = Field(description="Specifies the number of lagged variance terms in GARCH model", default=None)
    model_type : str = Field(description="Specifies the variant of GARCH model, including 'garch', " +\
    "'igarch', 'tgarch' and 'egarch'", default=None)
    # fit args
    key : str = Field(description="The key of the train data. If not provided, ask the user, do not guess")
    endog : str = Field(description="The endog of the fit data", default=None)
    thread_ratio : float = Field(description="The ratio of available threads used to fit the GARCH model", default=None)
    model_config = ConfigDict(protected_namespaces=())

class GARCHPredictInput(BaseModel):
    """
    Class for GRACH predict parameters.
    """
    name : str = Field(description="The name of the model. If not provided, ask the user, do not guess")
    version : int = Field(description="The version of the model", default=None)
    horizon : int = Field(description="The forecasting horizon", default=None)

class GARCHFitAndSave(BaseTool):
    r"""
    This tool is used to fit and save a GARCH model.

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
                  - The table (or view) containing the data to fit a GARCH model.
                * - p
                  - Specifies the number of lagged error terms in GARCH model.
                * - q
                  - Specifies the number of lagged variance terms in GARCH model.
                * - model_type
                  - Specifies the variant of GARCH model, including 'garch', 'igarch', 'tgarch' and 'egarch'.
                * - key
                  - The key of the train data.
                * - endog
                  - The endog of the fit data.
                * - thread_ratio
                  - The ratio of available threads used to fit the GARCH model.
    """
    name: str = "garch_fit_and_save"
    """Name of the tool."""
    description: str = "To fit a GARCH model and save the model in model storage."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = GARCHFitInput
    return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context
        )

    def _run(#pylint:disable=too-many-positional-arguments
        self,
        name: str,
        fit_table : str,
        key : str,
        version: int = None,
        p : int = None,
        q : int = None,
        model_type : str = None,
        endog : str = None,
        thread_ratio : float = None,
        run_manager: CallbackManagerForToolRun = None#pylint:disable=unused-argument
    ) -> str:
        """Use the tool."""
        ms = ModelStorage(connection_context=self.connection_context)
        garch = GARCH(p=p, q=q, model_type=model_type)
        garch.fit(data=self.connection_context.table(fit_table),
                  key=key,
                  endog=endog,
                  thread_ratio=thread_ratio)
        garch.name = name
        if version is None:
            version = ms._get_new_version_no(name)
            if version is None:
                version = 1
            else:
                version = int(version)
        garch.version = version
        ms.save_model(model=garch, if_exists='replace')
        return json.dumps({"train_table/view": fit_table, "model_storage_name": name, "model_storage_version": version})

    async def _arun(#pylint:disable=too-many-positional-arguments
        self,
        name: str,
        fit_table : str,
        key : str,
        version: int = None,
        p : int = None,
        q : int = None,
        model_type : str = None,
        endog : str = None,
        thread_ratio : float = None,
        run_manager: AsyncCallbackManagerForToolRun = None
    ) -> str:
        return self._run(name=name,
                         version=version,
                         fit_table=fit_table,
                         p=p, q=q,
                         model_type=model_type,
                         key=key,
                         endog=endog,
                         thread_ratio=thread_ratio,
                         run_manager=run_manager)

class GARCHLoadModelAndPredict(BaseTool):
    """
    This tool is used to load the additive model forecast from model storage and predict.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The result string containing the prediction result table.

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
                * - horizon
                  - The forecasting horizon.
    """
    name: str = "garch_load_model_and_predict"
    """Name of the tool."""
    description: str = "To load a garch model from model storage and do predict."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = GARCHPredictInput
    return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context
        )

    def _run(
        self,
        name : str,
        version : int = None,
        horizon : int = None,
        run_manager : CallbackManagerForToolRun = None#pylint:disable=unused-argument
    ) -> str:
        ms = ModelStorage(connection_context=self.connection_context)
        model = ms.load_model(name=name, version=version)
        if hasattr(model, 'version'):
            if model.version is not None:
                version = model.version
        model.name = name
        model.version = version
        ms.save_model(model=model, if_exists='replace_meta')
        out_tabs = model.predict(horizon=horizon)
        predict_result = remove_prefix_sharp(f"{name}_{model.version}_PREDICT_RESULT")
        out_tabs[0].save(predict_result, force=True)
        out_dict = {"predict_result_table" : predict_result}
        if out_tabs[1].count() > 0:
            for _, row in out_tabs[1].collect().iterrows():
                out_dict[row["STATS_NAME"]] = row["STATS_VALUE"]
        return json.dumps(out_dict)

    async def _arun(
        self,
        name : str,
        version : int = None,
        horizon : int = None,
        run_manager: AsyncCallbackManagerForToolRun = None
    ) -> str:
        return self._run(name=name,
                         version=version,
                         horizon=horizon,
                         run_manager=run_manager)
