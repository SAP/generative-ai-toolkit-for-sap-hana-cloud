"""
hana_ai.retrieval.data_retrieval

The following classes are available:

    * :class:`DataRetrieval`
"""
import logging

from .retrieval_base import RetrievalBase


logger = logging.getLogger(__name__)

class DataRetrieval(RetrievalBase):
    """
    Data Retrieval for interacting with AI Core services.

    The user has the below privileges to create/drop remote source and PSE as well as call the Data Retrieval SQL:

    - EXECUTE privilege on the DATA_AGENT_DEV or DATA_AGENT stored procedure.
    - CREATE REMOTE SOURCE privilege.
    - TRUST ADMIN privilege.
    - CERTIFICATE ADMIN privilege.
    """
    def __init__(
        self,
        connection_context,
        agent_type: str = "AI_DATA_RETRIEVAL",
        *,
        schema_name: str = "SYS",
        procedure_name: str | None = None,
        remote_source_name: str = None,
        knowledge_graph_name: str = None,
        rag_schema_name: str = None,
        rag_table_name: str = None,
        metadata_schema_name: str | None = None,
        metadata_object_prefix: str | None = None,
    ):
        """
        Initialize the DataRetrieval.

        Parameters
        ----------
        connection_context : ConnectionContext
            The HANA connection context.
        """
        super().__init__(
            connection_context,
            agent_type=agent_type,
            schema_name=schema_name,
            procedure_name=procedure_name,
            remote_source_name=remote_source_name,
            knowledge_graph_name=knowledge_graph_name,
            rag_schema_name=rag_schema_name,
            rag_table_name=rag_table_name,
            metadata_schema_name=metadata_schema_name,
            metadata_object_prefix=metadata_object_prefix,
        )
        self.conn_context = connection_context
