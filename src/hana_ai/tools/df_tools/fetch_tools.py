"""
This module contains the functions to fetch data from HANA.

The following classes are available:

    * :class `FetchDataTool`
"""
import copy
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
    select_statement: str = Field(description="the select statement of dataframe. If not provided, ask the user. Do not guess.")
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
                * - select_statement
                  - The select_statement of dataframe. If not provided, ask the user. Do not guess.
                * - top_n
                  - The number of rows to fetch, it is optional
                * - last_n
                  - The number of rows to fetch from the end of the table, it is optional
    """
    name: str = "fetch_data"
    """Name of the tool."""
    description: str = "Fetch data from a select statement of dataframe."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = FetchDataInput
    """Input schema of the tool."""
    return_direct: bool = False
    """Used for transform"""
    is_transform: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False,
        is_transform: bool = False
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct,
            is_transform=is_transform
        )

    def set_transform(self, is_transform: bool):
        """
        Return a copy of the tool with the is_transform flag set.

        Parameters
        ----------
        is_transform : bool
            Whether to set the tool to transform mode.
        """
        new_tool = copy.copy(self)
        new_tool.is_transform = is_transform
        return new_tool

    def _run(
        self, **kwargs
    ) -> str:
        """Use the tool."""
        # 从kwargs字典中提取参数
        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        select_statement = kwargs.get("select_statement", None)
        top_n = kwargs.get("top_n")
        last_n = kwargs.get("last_n")

        # 参数校验
        if select_statement is None:
            return "select_statement is required"
        if top_n:
            results = self.connection_context.sql(select_statement).head(top_n)
        elif last_n:
            results = self.connection_context.sql(select_statement).tail(last_n)
        else:
            results = self.connection_context.sql(select_statement)
        if self.is_transform is True:
            return results.select_statement
        else:
            results = results.collect()
        if not self.return_direct:
            results = results.to_markdown(index=False)
        return results

    async def _arun(
        self, **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs
        )
