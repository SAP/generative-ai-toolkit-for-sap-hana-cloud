"""
This module contains the functions to fetch data from HANA.

The following classes are available:

    * :class `DummyTool`
"""

import logging
from typing import Optional, Type
from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext

logger = logging.getLogger(__name__)

class UnsupportedToolInput(BaseModel):
    """
    The input schema for the FetchDataTool.
    """
    name: Optional[str] = Field(description="Any string. It is not mandatory.", default=None)

class ClassificationTool(BaseTool):
    """
    This tool is to handle unsupported tools.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The message.
    """
    name: str = "classification_tool"
    """Name of the tool."""
    description: str = "To train the classification model or to predict on the classification model."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = UnsupportedToolInput
    """Input schema of the tool."""
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
        self, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs: str
    ) -> str:
        """Use the tool."""
        return "Currently, the machine learning models in hana.ai tools only support time series-related models."

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None, **kwargs: str
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(run_manager=run_manager)
    
    class RegressionTool(BaseTool):
        """
        This tool is to handle unsupported tools.

        Parameters
        ----------
        connection_context : ConnectionContext
            Connection context to the HANA database.

        Returns
        -------
        str
            The message.
        """
        name: str = "regression_tool"
        """Name of the tool."""
        description: str = "To train the regression model or to predict on the regression model."
        """Description of the tool."""
        connection_context: ConnectionContext = None
        """Connection context to the HANA database."""
        args_schema: Type[BaseModel] = UnsupportedToolInput
        """Input schema of the tool."""
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
            self, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs: str
        ) -> str:
            """Use the tool."""
            return "Currently, the machine learning models in hana.ai tools only support time series-related models."

        async def _arun(
            self,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None, **kwargs: str
        ) -> str:
            """Use the tool asynchronously."""
            return self._run(run_manager=run_manager)
