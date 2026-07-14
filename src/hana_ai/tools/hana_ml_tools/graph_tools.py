"""
This module is used to discover HANA objects via knowledge graph.

The following classes are available:

    * :class `ObjectDiscoveryTool`
    * :class `DataRetrievalTool`
"""

from typing import Optional, Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ai.retrieval.object_discovery import ObjectDiscovery
from hana_ai.retrieval.data_retrieval import DataRetrieval

class HANARetrievalToolInput(BaseModel):
    """
    Input schema for ObjectDiscovery.
    """
    query : str = Field(description="The query to discover HANA objects via knowledge graph.")
    model_name: Optional[str] = Field(
        description=(
            "The name of the AI Core model to use. Leave unset for procedures "
            "with the simple (IN query, OUT output) signature; setting this "
            "adds a config JSON positional argument to the CALL."
        ),
        default=None,
    )

class ObjectDiscoveryTool(BaseTool):
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
    name: str = "object_discovery"
    description: str = "Tool for discovering HANA objects via knowledge graph."
    connection_context : ConnectionContext = None
    """Connection context to the HANA database."""
    remote_source_name: str = None
    rag_schema_name: str = None
    rag_table_name: str = None
    knowledge_graph_name: str = None
    schema_name: str = "SYS"
    procedure_name: Optional[str] = "AI_OBJECT_RETRIEVAL"
    metadata_schema_name: Optional[str] = None
    metadata_object_prefix: Optional[str] = None
    args_schema: Type[BaseModel] = HANARetrievalToolInput
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
                  remote_source_name: str = None,
                  rag_schema_name: str = None,
                  rag_table_name: str = None,
                  knowledge_graph_name: str = None,
                  schema_name: str = "SYS",
                  procedure_name: str | None = "AI_OBJECT_RETRIEVAL",
                  metadata_schema_name: str | None = None,
                  metadata_object_prefix: str | None = None):
        """
        Configure the additional settings for Object Discovery.

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
        schema_name : str, optional
            The schema name where the Object Discovery stored procedure is located, by default "SYS".
        procedure_name : str | None, optional
            The name of the Object Discovery stored procedure, by default "AI_OBJECT_RETRIEVAL".
        metadata_schema_name : str | None, optional
            Value for the procedure's ``AI_METADATA_SCHEMA_NAME`` IN parameter
            (no-wrapper deployments only). When set together with
            ``metadata_object_prefix``, the CALL switches to named-parameter
            syntax. Leave unset for customer wrapper procedures.
        metadata_object_prefix : str | None, optional
            Value for ``AI_METADATA_OBJECT_PREFIX``.
        """
        self.remote_source_name = remote_source_name
        self.rag_schema_name = rag_schema_name
        self.rag_table_name = rag_table_name
        self.knowledge_graph_name = knowledge_graph_name
        self.schema_name = schema_name
        self.procedure_name = procedure_name
        self.metadata_schema_name = metadata_schema_name
        self.metadata_object_prefix = metadata_object_prefix

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

        # Only attach a model config when the caller explicitly asked for one,
        # otherwise leave additional_config=None so _call_procedure_sql can emit
        # the 2-arg CALL form for procedures that don't accept a config blob.
        model_name = kwargs.get("model_name")
        additional_config = {"model": {"name": model_name}} if model_name else None
        retriever = ObjectDiscovery(
            connection_context=self.connection_context,
            remote_source_name=self.remote_source_name,
            knowledge_graph_name=self.knowledge_graph_name,
            rag_schema_name=self.rag_schema_name,
            rag_table_name=self.rag_table_name,
            schema_name=self.schema_name,
            procedure_name=self.procedure_name,
            metadata_schema_name=self.metadata_schema_name,
            metadata_object_prefix=self.metadata_object_prefix,
        )

        try:
            result = retriever.run(query=query, additional_config=additional_config)
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

class DataRetrievalTool(BaseTool):
    """
    Tool for interacting with Data Retrieval.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.
    Returns
    -------
    str
        The Data Retrieval query result as a string.
    """
    name: str = "data_retrieval"
    description: str = "Tool for interacting with Data Retrieval."
    connection_context : ConnectionContext = None
    """Connection context to the HANA database."""
    remote_source_name: str = None
    rag_schema_name: str = None
    rag_table_name: str = None
    knowledge_graph_name: str = None
    schema_name: str = "SYS"
    procedure_name: Optional[str] = "AI_DATA_RETRIEVAL"
    metadata_schema_name: Optional[str] = None
    metadata_object_prefix: Optional[str] = None
    args_schema: Type[BaseModel] = HANARetrievalToolInput
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
                  remote_source_name: str = None,
                  rag_schema_name: str = None,
                  rag_table_name: str = None,
                  knowledge_graph_name: str = None,
                  schema_name: str = "SYS",
                  procedure_name: str | None = "AI_DATA_RETRIEVAL",
                  metadata_schema_name: str | None = None,
                  metadata_object_prefix: str | None = None):
        """
        Configure the additional settings for Data Retrieval.

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
        schema_name : str, optional
            The schema name where the Data Retrieval stored procedure is located, by default "SYS".
        procedure_name : str | None, optional
            The name of the Data Retrieval stored procedure, by default None.
        metadata_schema_name : str | None, optional
            Value for the procedure's ``AI_METADATA_SCHEMA_NAME`` IN parameter
            (no-wrapper deployments only). See :meth:`ObjectDiscoveryTool.configure`.
        metadata_object_prefix : str | None, optional
            Value for ``AI_METADATA_OBJECT_PREFIX``.
        """
        self.remote_source_name = remote_source_name
        self.rag_schema_name = rag_schema_name
        self.rag_table_name = rag_table_name
        self.knowledge_graph_name = knowledge_graph_name
        self.schema_name = schema_name
        self.procedure_name = procedure_name
        self.metadata_schema_name = metadata_schema_name
        self.metadata_object_prefix = metadata_object_prefix

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

        # Only attach a model config when the caller explicitly asked for one;
        # see ObjectDiscoveryTool._run for rationale.
        model_name = kwargs.get("model_name")
        additional_config = {"model": {"name": model_name}} if model_name else None

        retriever = DataRetrieval(
            connection_context=self.connection_context,
            remote_source_name=self.remote_source_name,
            knowledge_graph_name=self.knowledge_graph_name,
            rag_schema_name=self.rag_schema_name,
            rag_table_name=self.rag_table_name,
            schema_name=self.schema_name,
            procedure_name=self.procedure_name,
            metadata_schema_name=self.metadata_schema_name,
            metadata_object_prefix=self.metadata_object_prefix,
        )

        try:
            result = retriever.run(query=query, additional_config=additional_config)
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
