"""
This module is used to discover HANA objects via knowledge graph.

The following classes are available:

    * :class `DiscoveryAgentTool`
    * :class `DataAgentTool`
    * :class `CreateRemoteSourceTool`
"""

from typing import Optional, Type

from cv2 import add
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
    model_name: Optional[str] = Field(description="The name of the AI Core model to use. Default is None.", default='gpt-4.1')

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
    remote_source_name: str = "HANA_DISCOVERY_AGENT_CREDENTIALS"
    rag_schema_name: str = "SYSTEM"
    rag_table_name: str = "RAG"
    knowledge_graph_name: str = "HANA_OBJECTS"
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

    def configure(self,
                  remote_source_name: str,
                  rag_schema_name: str,
                  rag_table_name: str,
                  knowledge_graph_name: str):
        """
        Configure the additional settings for Data Agent.

        Parameters
        ----------
        remote_source_name : str
            The name of the remote source to connect to AI Core.
        rag_schema_name : str
            The schema name where RAG tables are stored.
        rag_table_name : str
            The table name where RAG data is stored.
        knowledge_graph_name : str
            The name of the knowledge graph to use.
        model_name : str
            The name of the AI Core model to use.
        """
        self.remote_source_name = remote_source_name
        self.rag_schema_name = rag_schema_name
        self.rag_table_name = rag_table_name
        self.knowledge_graph_name = knowledge_graph_name

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

        additional_config = {
            "model": {
                "name": kwargs.get("model_name", "gpt-4.1")
            }
        }
        da = DiscoveryAgent(
            connection_context=self.connection_context,
            remote_source_name=self.remote_source_name,
            knowledge_graph_name=self.knowledge_graph_name,
            rag_schema_name=self.rag_schema_name,
            rag_table_name=self.rag_table_name
        )

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
    remote_source_name: str = "HANA_DISCOVERY_AGENT_CREDENTIALS"
    rag_schema_name: str = "SYSTEM"
    rag_table_name: str = "RAG"
    knowledge_graph_name: str = "HANA_OBJECTS"
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

    def configure(self,
                  remote_source_name: str,
                  rag_schema_name: str,
                  rag_table_name: str,
                  knowledge_graph_name: str):
        """
        Configure the additional settings for Data Agent.

        Parameters
        ----------
        remote_source_name : str
            The name of the remote source to connect to AI Core.
        rag_schema_name : str
            The schema name where RAG tables are stored.
        rag_table_name : str
            The table name where RAG data is stored.
        knowledge_graph_name : str
            The name of the knowledge graph to use.
        model_name : str
            The name of the AI Core model to use.
        """
        self.remote_source_name = remote_source_name
        self.rag_schema_name = rag_schema_name
        self.rag_table_name = rag_table_name
        self.knowledge_graph_name = knowledge_graph_name

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

        additional_config = {
            "model": {
                "name": kwargs.get("model_name", "gpt-4.1")
            }
        }
        da = DataAgent(
            connection_context=self.connection_context,
            remote_source_name=self.remote_source_name,
            knowledge_graph_name=self.knowledge_graph_name,
            rag_schema_name=self.rag_schema_name,
            rag_table_name=self.rag_table_name
        )

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
