"""
hana_ml.agents.data_agent

The following classes are available:

    * :class:`DataAgent`
"""
import logging

from .agent_base import AgentBase


logger = logging.getLogger(__name__)

class DataAgent(AgentBase):
    """
    Data Agent for interacting with AI Core services.

    The user has the below privileges to create/drop remote source and PSE as well as call the Data Agent SQL:

    - EXECUTE privilege on the DATA_AGENT_DEV or DATA_AGENT stored procedure.
    - CREATE REMOTE SOURCE privilege.
    - TRUST ADMIN privilege.
    - CERTIFICATE ADMIN privilege.
    """
    def __init__(self, connection_context, agent_type="DATA_AGENT_DEV"):
        """
        Initialize the DataAgent.

        Parameters
        ----------
        connection_context : ConnectionContext
            The HANA connection context.
        """
        super().__init__(connection_context, agent_type=agent_type)
        self.conn_context = connection_context
        self.remote_source_name = "HANA_DISCOVERY_AGENT_CREDENTIALS"
        self.pse_name = "AI_CORE_PSE"
