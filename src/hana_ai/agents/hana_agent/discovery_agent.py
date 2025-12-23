"""
hana_ml.agents.discovery_agent

The following classes are available:
    * :class:`DiscoveryAgent`
"""
import logging

from .agent_base import AgentBase


logger = logging.getLogger(__name__)

class DiscoveryAgent(AgentBase):
    """
    Discovery Agent for interacting with AI Core services.

    The user has the below privileges to create/drop remote source and PSE as well as call the Discovery Agent SQL:

    - EXECUTE privilege on the DISCOVERY_AGENT_DEV or DISCOVERY_AGENT stored procedure.
    - CREATE REMOTE SOURCE privilege.
    - TRUST ADMIN privilege.
    - CERTIFICATE ADMIN privilege.

    """
    def __init__(self, connection_context, agent_type="DISCOVERY_AGENT_DEV"):
        """
        Initialize the DiscoveryAgent.

        Parameters
        ----------
        connection_context : ConnectionContext
            The HANA connection context.
        """
        super().__init__(connection_context, agent_type=agent_type)
        self.conn_context = connection_context
        self.remote_source_name = "HANA_DISCOVERY_AGENT_CREDENTIALS"
        self.pse_name = "AI_CORE_PSE"
