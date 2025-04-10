"""
This module contains the functions to fetch data from HANA.

The following classes are available:

    * :class `FetchDataTool`
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

from hana_ai.tools.hana_ml_tools.utility import _CustomEncoder

logger = logging.getLogger(__name__)

class FetchDataInput(BaseModel):
    """
    The input schema for the FetchDataTool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    top_n: Optional[int] = Field(description="the number of rows to fetch, it is optional", default=None)
    last_n: Optional[int] = Field(description="the number of rows to fetch from the end of the table, it is optional", default=None)

class FetchDataTool(BaseTool):
    """
    This tool fetches data from a given table.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    pandas.DataFrame
        The fetched data.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - table_name
                  - The name of the table. If not provided, ask the user. Do not guess.
                * - top_n
                  - The number of rows to fetch, it is optional
                * - last_n
                  - The number of rows to fetch from the end of the table, it is optional
    """
    name: str = "fetch_data"
    """Name of the tool."""
    description: str = "Fetch data from a given table."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = FetchDataInput
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
        self, table_name: str, top_n: Optional[int] = None, last_n: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        if top_n:
            results = self.connection_context.table(table_name).head(top_n).collect()
        elif last_n:
            results = self.connection_context.table(table_name).tail(last_n).collect()
        else:
            results = self.connection_context.table(table_name).collect()
        # serialize the results
        
        return results

    async def _arun(
        self, table_name: str, top_n: Optional[int] = None, last_n: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(table_name, top_n, last_n, run_manager=run_manager
        )
