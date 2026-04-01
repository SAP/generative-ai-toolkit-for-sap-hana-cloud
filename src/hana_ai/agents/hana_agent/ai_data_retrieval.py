"""
hana_ml.agents.ai_data_retrieval

The following classes are available:

    * :class:`AIDataRetrieval`
"""
import logging

from .agent_base import AgentBase


logger = logging.getLogger(__name__)


class AIDataRetrieval(AgentBase):
    """
    Data retrieval agent for calling AI_DATA_RETRIEVAL.

    The user needs EXECUTE privilege on the AI_DATA_RETRIEVAL procedure.
    The remote source and metadata objects are expected to be prepared already.
    """

    PROCEDURE_NAME = "AI_DATA_RETRIEVAL"

    def __init__(
        self,
        connection_context,
        *,
        schema_name: str = "SYS",
        remote_source_name: str = "HANA_DISCOVERY_AGENT_CREDENTIALS",
        ai_metadata_schema_name: str = "SYSTEM",
        ai_metadata_object_prefix: str = "HANA_OBJECTS",
        remote_source_schema_name: str | None = None,
        model_and_version: str | None = None,
    ):
        """
        Initialize the AIDataRetrieval agent.

        Parameters
        ----------
        connection_context : ConnectionContext
            The HANA connection context.
        """
        super().__init__(
            connection_context,
            schema_name=schema_name,
            procedure_name=self.PROCEDURE_NAME,
            remote_source_name=remote_source_name,
            ai_metadata_schema_name=ai_metadata_schema_name,
            ai_metadata_object_prefix=ai_metadata_object_prefix,
            remote_source_schema_name=remote_source_schema_name,
            model_and_version=model_and_version,
        )