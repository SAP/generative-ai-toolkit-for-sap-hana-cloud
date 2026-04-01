"""
This module exposes HANA AI retrieval tools.

The following classes are available:

    * :class `AIObjectRetrievalTool`
    * :class `AIDataRetrievalTool`
"""

from typing import Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ai.agents.hana_agent.ai_object_retrieval import AIObjectRetrieval
from hana_ai.agents.hana_agent.ai_data_retrieval import AIDataRetrieval

class HANAAgentToolInput(BaseModel):
    """
    Input schema for HANA retrieval agents.
    """
    query : str = Field(description="The query to discover HANA objects via knowledge graph.")

class AIObjectRetrievalTool(BaseTool):
    """
    Tool for running AI_OBJECT_RETRIEVAL.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The retrieval result as a string.
    """
    name: str = "ai_object_retrieval"
    description: str = "Tool for running AI_OBJECT_RETRIEVAL."
    connection_context : ConnectionContext = None
    """Connection context to the HANA database."""
    remote_source_name: str = "HANA_DISCOVERY_AGENT_CREDENTIALS"
    ai_metadata_schema_name: str = "SYSTEM"
    ai_metadata_object_prefix: str = "HANA_OBJECTS"
    schema_name: str = "SYS"
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
                  ai_metadata_schema_name: str,
                  ai_metadata_object_prefix: str,
                  schema_name: str = "SYS"):
        """
        Configure the additional settings for AI_OBJECT_RETRIEVAL.

        Parameters
        ----------
        remote_source_name : str
            The name of the configured remote source.
        ai_metadata_schema_name : str
            The schema name where AI metadata objects are stored.
        ai_metadata_object_prefix : str
            Prefix used to resolve AI metadata objects.
        schema_name : str, optional
            The schema name where the retrieval procedure is located, by default "SYS".
        """
        self.remote_source_name = remote_source_name
        self.ai_metadata_schema_name = ai_metadata_schema_name
        self.ai_metadata_object_prefix = ai_metadata_object_prefix
        self.schema_name = schema_name

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
        retrieval = AIObjectRetrieval(
            connection_context=self.connection_context,
            remote_source_name=self.remote_source_name,
            ai_metadata_schema_name=self.ai_metadata_schema_name,
            ai_metadata_object_prefix=self.ai_metadata_object_prefix,
            schema_name=self.schema_name,
        )

        try:
            result = retrieval.run(query=query)
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

class AIDataRetrievalTool(BaseTool):
    """
    Tool for running AI_DATA_RETRIEVAL.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.
    Returns
    -------
    str
        The retrieval result as a string.
    """
    name: str = "ai_data_retrieval"
    description: str = "Tool for running AI_DATA_RETRIEVAL."
    connection_context : ConnectionContext = None
    """Connection context to the HANA database."""
    remote_source_name: str = "HANA_DISCOVERY_AGENT_CREDENTIALS"
    ai_metadata_schema_name: str = "SYSTEM"
    ai_metadata_object_prefix: str = "HANA_OBJECTS"
    schema_name: str = "SYS"
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
                  ai_metadata_schema_name: str,
                  ai_metadata_object_prefix: str,
                  schema_name: str = "SYS"):
        """
        Configure the additional settings for AI_DATA_RETRIEVAL.

        Parameters
        ----------
        remote_source_name : str
            The name of the configured remote source.
        ai_metadata_schema_name : str
            The schema name where AI metadata objects are stored.
        ai_metadata_object_prefix : str
            Prefix used to resolve AI metadata objects.
        schema_name : str, optional
            The schema name where the retrieval procedure is located, by default "SYS".
        """
        self.remote_source_name = remote_source_name
        self.ai_metadata_schema_name = ai_metadata_schema_name
        self.ai_metadata_object_prefix = ai_metadata_object_prefix
        self.schema_name = schema_name

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

        retrieval = AIDataRetrieval(
            connection_context=self.connection_context,
            remote_source_name=self.remote_source_name,
            ai_metadata_schema_name=self.ai_metadata_schema_name,
            ai_metadata_object_prefix=self.ai_metadata_object_prefix,
            schema_name=self.schema_name,
        )

        try:
            result = retrieval.run(query=query)
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
