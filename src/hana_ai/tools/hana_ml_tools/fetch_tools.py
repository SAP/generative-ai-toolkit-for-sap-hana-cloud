"""
This module contains the functions to fetch data from HANA.

The following classes are available:

    * :class `FetchDataTool`
"""

import logging
from typing import Optional, Type
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext

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
        self, **kwargs
    ) -> str:
        """Use the tool."""
        # 从kwargs字典中提取参数
        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        table_name = kwargs.get("table_name", None)
        top_n = kwargs.get("top_n")
        last_n = kwargs.get("last_n")

        # 参数校验
        if table_name is None:
            return "table_name is required"
        if top_n:
            results = self.connection_context.table(table_name).head(top_n).collect()
        elif last_n:
            results = self.connection_context.table(table_name).tail(last_n).collect()
        else:
            results = self.connection_context.table(table_name).collect()
        if not self.return_direct:
            results = results.to_string(index=False)
        return results

    async def _arun(
        self, **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs
        )
