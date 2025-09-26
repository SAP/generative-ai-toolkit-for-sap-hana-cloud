"""
This module contains the functions to store SelectStatement data to HANA tables.

The following classes are available:

    * :class `SelectStatementToTableTool`
"""

import logging
from typing import Type, Optional
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from hana_ml import ConnectionContext

logger = logging.getLogger(__name__)

class SelectStatementToTableInput(BaseModel):
    """
    The input schema for the SelectStatementToTableTool.
    """
    table_name: str = Field(description="The name of the target table in HANA")
    select_statement: str = Field(description="The SQL select statement. It must be provided. ")
    schema_name: Optional[str] = Field(description="the schema_name of the table, it is optional", default=None)
    force: Optional[str] = Field(description="Whether to overwrite the table if it already exists. Default is False.", default=False)

class SelectStatementToTableTool(BaseTool):
    """
    This tool stores SelectStatement data into a HANA table.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        Operation result message

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - table_name
                  - The name of the target table in HANA
                * - SelectStatement_data
                  - List of SelectStatement objects to store in the table
    """
    name: str = "SelectStatement_to_table"
    """Name of the tool."""
    description: str = "Save the SQL select statement results to a HANA table."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = SelectStatementToTableInput
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
        table_name = kwargs.get("table_name")
        select_statement = kwargs.get("select_statement")
        schema_name = kwargs.get("schema_name", None)
        force = kwargs.get("force", False)
        # 参数校验
        if not table_name:
            return "Error: table_name is required"
        if not select_statement:
            return "Error: select_statement is required"

        # 调用核心存储函数
        return SelectStatement_to_table(select_statement, self.connection_context, table_name, schema_name, force)

    async def _arun(
        self, **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs)

def SelectStatement_to_table(select_statement: str,
                             connection_context: ConnectionContext,
                             table_name: str,
                             schema_name: str,
                             force: bool) -> str:
    """
    Stores SelectStatement data (list of dictionaries) into a HANA table
    
    Parameters
    ----------
    select_statement : str
        The SQL select statement to store the results from
    connection_context : ConnectionContext
        HANA database connection context
    table_name : str
        Target table name in HANA
    schema_name : str
        The schema_name of the table, it is optional
    force : bool
        Whether to overwrite the table if it already exists. Default is False.

    Returns
    -------
    str
        Operation result message
    """
    try:
        # 将SelectStatement数据转换为Pandas DataFrame

        connection_context.sql(select_statement).smart_save(table_name, schema=schema_name, force=force)

        return f"Successfully save the data to '{table_name}'"

    except Exception as e:
        logger.error("Error storing data to HANA: %s", str(e))
        return f"Operation failed: {str(e)}"
