"""
This module is used to discover HANA objects via knowledge graph.

The following classes are available:

    * :class `DiscoveryAgentTool`
    * :class `DataAgentTool`
    * :class `CreateRemoteSourceTool`
"""

from typing import Optional, Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ai.agents.hana_agent.agent_base import AgentBase
from hana_ai.agents.hana_agent.discovery_agent import DiscoveryAgent
from hana_ai.agents.hana_agent.data_agent import DataAgent

class HANAAgentToolInput(BaseModel):
    """
    Input schema for DiscoveryAgent.
    """
    query : str = Field(description="The query to discover HANA objects via knowledge graph.")
    remote_source_name : Optional[str] = Field(description="The name of the remote source to connect to AI Core. Default is 'HANA_DISCOVERY_AGENT_CREDENTIALS'.", default="HANA_DISCOVERY_AGENT_CREDENTIALS")
    rag_schema_name: Optional[str] = Field(description="The schema name where RAG tables are stored. Default is 'SYSTEM'.", default="SYSTEM")
    rag_table_name: Optional[str] = Field(description="The table name where RAG data is stored. Default is 'RAG'.", default="RAG")
    graph_name: Optional[str] = Field(description="The name of the knowledge graph to use. Default is 'HANA_OBJECTS'.", default="HANA_OBJECTS")

class CreateRemoteSourceInput(BaseModel):
    """
    Input schema for creating remote source.
    """
    credentials_file: str = Field(description="The filepath of the credentials for AI Core service.")
    remote_source_name : Optional[str] = Field(description="The name of the remote source to create. Default is 'HANA_DISCOVERY_AGENT_CREDENTIALS'.", default="HANA_DISCOVERY_AGENT_CREDENTIALS")
    pse_name : Optional[str] = Field(description="The name of the PSE to create. Default is 'AI_CORE_PSE'.", default="AI_CORE_PSE")
    create_pse : Optional[bool] = Field(description="Whether to create PSE. Default is False.", default=False)

class CreateRemoteSourceTool(BaseTool):
    """
    Tool for creating remote source and PSE for AI Core.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The result of the remote source creation.
    """
    name: str = "create_hana_agent_remote_source"
    description: str = "Tool for creating remote source and PSE for AI Core."
    connection_context : ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = CreateRemoteSourceInput
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
        """Use the tool."""

        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        credentials_file= kwargs.get("credentials_file", None)
        if credentials_file is None:
            return "Credentials file is required"
        remote_source_name = kwargs.get("remote_source_name", "HANA_DISCOVERY_AGENT_CREDENTIALS")
        pse_name = kwargs.get("pse_name", "AI_CORE_PSE")
        create_pse = kwargs.get("create_pse", False)
        da = AgentBase(
            connection_context=self.connection_context,
            agent_type="DISCOVERY_AGENT"
        )
        try:
            da.create_remote_source(
                credentials=credentials_file,
                pse_name=pse_name,
                remote_source_name=remote_source_name,
                create_pse=create_pse
            )
        except Exception as err:
            # Handles invalid parameter values (e.g., alpha not in [0,1])
            return f"Error occurred: {str(err)}"
        return f"Remote source '{remote_source_name}' created successfully."

    async def _arun(
        self,
        **kwargs
    ) -> str:
        return self._run(**kwargs
        )

class DiscoveryAgentTool(BaseTool):
    """
    Tool for discovering HANA objects via knowledge graph.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The discovery result as a string.
    """
    name: str = "discovery_agent"
    description: str = "Tool for discovering HANA objects via knowledge graph."
    connection_context : ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = HANAAgentToolInput
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
        """Use the tool."""

        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        query= kwargs.get("query", None)
        if query is None:
            return "Query is required"
        remote_source_name = kwargs.get("remote_source_name", "HANA_DISCOVERY_AGENT_CREDENTIALS")
        rag_schema_name = kwargs.get("rag_schema_name", "SYSTEM")
        rag_table_name = kwargs.get("rag_table_name", "RAG")
        graph_name = kwargs.get("graph_name", "HANA_OBJECTS")
        additional_config = {
            "ragSchemaName": rag_schema_name,
            "ragTableName": rag_table_name,
            "graphName": graph_name
        }
        da = DiscoveryAgent(
            connection_context=self.connection_context
        )
        da.remote_source_name = remote_source_name
        try:
            result = da.run(query=query, additional_config=additional_config)
        except Exception as err:
            # Handles invalid parameter values (e.g., alpha not in [0,1])
            return f"Error occurred: {str(err)}"
        return result

    async def _arun(
        self,
        **kwargs
    ) -> str:
        return self._run(**kwargs
        )

class DataAgentTool(BaseTool):
    """
    Tool for interacting with Data Agent.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.
    Returns
    -------
    str
        The Data Agent query result as a string.
    """
    name: str = "data_agent"
    description: str = "Tool for interacting with Data Agent."
    connection_context : ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = HANAAgentToolInput
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
        """Use the tool."""

        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        query= kwargs.get("query", None)
        if query is None:
            return "Query is required"
        remote_source_name = kwargs.get("remote_source_name", "HANA_DISCOVERY_AGENT_CREDENTIALS")
        rag_schema_name = kwargs.get("rag_schema_name", "SYSTEM")
        rag_table_name = kwargs.get("rag_table_name", "RAG")
        graph_name = kwargs.get("graph_name", "HANA_OBJECTS")
        additional_config = {
            "ragSchemaName": rag_schema_name,
            "ragTableName": rag_table_name,
            "graphName": graph_name
        }
        da = DataAgent(
            connection_context=self.connection_context
        )
        da.remote_source_name = remote_source_name
        try:
            result = da.run(query=query, additional_config=additional_config)
        except Exception as err:
            # Handles invalid parameter values (e.g., alpha not in [0,1])
            return f"Error occurred: {str(err)}"
        return result

    async def _arun(
        self,
        **kwargs
    ) -> str:
        return self._run(**kwargs
        )
