"""
This module contains the tools for managing model storage.

The following class are available:

    * :class `ListModels`: List all models in the model storage.
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
from hana_ml.model_storage import ModelStorage

class ListModelsInput(BaseModel):
    """
    Input for the ListModels tool.
    """
    name: Optional[str] = Field(description="Name of the model to search for, it is optional.", default=None)
    version: Optional[int] = Field(description="Version of the model to search for, it is optional.", default=None)
    display_type: Optional[str] = Field(description="Display type of the searched model information chosen from {'complete', 'simple', 'no_reports'}, it is optional.", default=None)

class ListModels(BaseTool):
    """
    Tool to list all models in the model storage.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    pandas.DataFrame
        The list of models in the model storage.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - name
                  - Name of the model to search for, it is optional.
                * - version
                  - Version of the model to search for, it is optional.
                * - display_type
                  - Display type of the searched model information chosen from {'complete', 'simple', 'no_reports'}, it is optional.
    """

    name: str = "list_models"
    description: str = "List all models in the model storage."
    args_schema: Type[ListModelsInput] = ListModelsInput
    connection_context: ConnectionContext
    return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = True
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )
   
    def _run(
        self,
        name: Optional[str] = None,
        version: Optional[int] = None,
        display_type: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Run the tool.

        Parameters
        ----------
        name : str, optional
            Name of the model to search for, it is optional.
        version : int, optional
            Version of the model to search for, it is optional.
        display_type : str, optional
            Display type of the searched model information chosen from {'complete', 'simple', 'no_reports'}, it is optional.
        run_manager : CallbackManagerForToolRun, optional
            Callback manager for tool run.

        Returns
        -------
        pandas.DataFrame
            The list of models in the model storage.
        """
        ms = ModelStorage(self.connection_context)
        return ms.list_models(
            name=name,
            version=version,
            display_type=display_type
        )

    async def _run_async(
        self,
        name: Optional[str] = None,
        version: Optional[int] = None,
        display_type: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """
        Asynchronous run of the tool.

        Parameters
        ----------
        name : str, optional
            Name of the model to search for, it is optional.
        version : int, optional
            Version of the model to search for, it is optional.
        display_type : str, optional
            Display type of the searched model information chosen from {'complete', 'simple', 'no_reports'}, it is optional.
        run_manager : AsyncCallbackManagerForToolRun, optional
            Callback manager for tool run.

        Returns
        -------
        pandas.DataFrame
            The list of models in the model storage.
        """
        return self._run(
            name=name,
            version=version,
            display_type=display_type,
            run_manager=run_manager,
        )
