"""
This module contains the tools for managing model storage.

The following class are available:

    * :class `ListModels`: List all models in the model storage.
"""

import logging
from typing import Optional, Type
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.model_storage import ModelStorage

logger = logging.getLogger(__name__)

class ListModelsInput(BaseModel):
    """
    Input for the ListModels tool.
    """
    name: Optional[str] = Field(description="Name of the model to search for, it is optional.", default=None)
    version: Optional[int] = Field(description="Version of the model to search for, it is optional.", default=None)
    display_type: Optional[str] = Field(description="Display type of the searched model information chosen from {'complete', 'simple', 'no_reports'}, it is optional.", default=None)

class DeleteModelInput(BaseModel):
    """
    Input for the DeleteModel tool.
    """
    name: str = Field(description="Name of the model to delete.")
    version: Optional[int] = Field(description="Version of the model to delete, it is optional.", default=None)

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
        """
        Run the tool.

        Parameters
        ----------
        kwargs : dict
            Dictionary containing the parameters for the tool.

        Returns
        -------
        pandas.DataFrame
            The list of models in the model storage.
        """
        name = kwargs.get("name", None)
        version = kwargs.get("version", None)
        display_type = kwargs.get("display_type", None)

        if display_type not in {None, "complete", "simple", "no_reports"}:
            return "Invalid display_type. Choose from {'complete', 'simple', 'no_reports'}."

        ms = ModelStorage(self.connection_context)
        return ms.list_models(
            name=name,
            version=version,
            display_type=display_type
        )

    async def _run_async(
        self,
        **kwargs
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
            **kwargs
        )

class DeleteModels(BaseTool):
    """
    Tool to delete a model from the model storage.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        Confirmation message indicating the model has been deleted.
    """

    name: str = "delete_models"
    description: str = "Delete models from the model storage."
    args_schema: Type[DeleteModelInput] = DeleteModelInput
    connection_context: ConnectionContext

    def _run(self, **kwargs) -> str:
        name = kwargs.get("name")
        version = kwargs.get("version", None)

        ms = ModelStorage(self.connection_context)
        if version:
            ms.delete_model(name=name, version=version)
            return f"Model {name} with version {version} has been deleted."
        else:
            ms.delete_models(name=name)
            return f"Model {name} has been deleted."
