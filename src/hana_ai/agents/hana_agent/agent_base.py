"""
hana_ml.agents.agent_base
"""
import json
import logging
import threading
import time

from hana_ml.ml_base import MLBase
from hana_ml.visualizers.shared import EmbeddedUI

from .progress_monitor import TextProgressMonitor
from .utility import _call_agent_sql, _create_ai_core_remote_source, _delete_ai_core_pse, _drop_ai_core_remote_source, _drop_certificate

logger = logging.getLogger(__name__)

class AgentBase(MLBase):
    """
    Discovery Agent for interacting with AI Core services.
    """
    def __init__(self, connection_context, agent_type=None,
                 schema_name: str = "SYS",
                 procedure_name: str | None = None,
                 remote_source_name: str = "HANA_DISCOVERY_AGENT_CREDENTIALS",
                 knowledge_graph_name: str = "HANA_OBJECTS",
                 rag_schema_name: str = "SYSTEM",
                 rag_table_name: str = "RAG"):
        """
        Initialize the AgentBase.

        Parameters
        ----------
        connection_context : ConnectionContext
            The HANA connection context.
        agent_type : str, optional
            Deprecated. Previously restricted to fixed values. If provided and
            `procedure_name` is not set, it will be used as the `procedure_name`
            for backward compatibility.
        schema_name : str, optional
            Schema for the target procedure. Default is "SYS".
        procedure_name : str, optional
            The procedure name to call. If omitted, falls back to `agent_type`.
        remote_source_name : str, optional
            The name of the remote source to be used. Default is "HANA_DISCOVERY_AGENT_CREDENTIALS".
        knowledge_graph_name : str, optional
            The name of the knowledge graph to be used. Default is "HANA_OBJECTS".
        rag_schema_name : str, optional
            The name of the RAG schema to be used. Default is "SYSTEM".
        rag_table_name : str, optional
            The name of the RAG table to be used. Default is "RAG".
        """
        super().__init__(connection_context)
        self.conn_context = connection_context
        self.remote_source_name = "HANA_DISCOVERY_AGENT_CREDENTIALS"
        self.pse_name = "AI_CORE_PSE"
        # Backward compatibility: if `procedure_name` not given, use `agent_type` as procedure
        if procedure_name is None and agent_type is not None:
            logging.getLogger(__name__).warning(
                "`agent_type` parameter is deprecated. Use schema_name/procedure_name instead."
            )
            procedure_name = agent_type

        self.schema_name = schema_name
        self.procedure_name = procedure_name
        self.knowledge_graph_name = knowledge_graph_name
        self.rag_schema_name = rag_schema_name
        self.rag_table_name = rag_table_name
        self.remote_source_name = remote_source_name

    def drop_remote_source(self, remote_source_name: str, cascade: bool = True):
        """
        Drop the remote source for AI Core.

        Parameters
        ----------
        remote_source_name : str, optional
            The name of the remote source to be dropped. Default is "AI_CORE_REMOTE_SOURCE".
        cascade : bool, optional
            Whether to drop dependent objects as well. Default is True.
        """
        try:
            _drop_ai_core_remote_source(
                connection_context=self.conn_context,
                remote_source_name=remote_source_name,
                cascade=cascade
            )
        except Exception as exc:
            logger.warning("Failed to drop remote source %s: %s", remote_source_name, str(exc))

    def drop_pse(self, pse_name: str, cascade: bool = True):
        """
        Drop the PSE for AI Core.

        Parameters
        ----------
        pse_name : str, optional
            The name of the PSE to be dropped. Default is "AI_CORE_PSE".
        cascade : bool, optional
            Whether to drop dependent objects as well. Default is True.
        """
        try:
            _delete_ai_core_pse(
                connection_context=self.conn_context,
                pse_name=pse_name,
                cascade=cascade
            )
        except Exception as exc:
            logger.warning("Failed to drop PSE %s: %s", pse_name, str(exc))

    def drop_certificates(self):
        """
        Drop the certificate X1ROOT and DIGICERTG5.
        """
        try:
            _drop_certificate(
                connection_context=self.conn_context,
                certificate_name="X1ROOT"
            )
        except Exception as exc:
            logger.warning("Failed to drop certificate X1ROOT: %s", str(exc))
        try:
            _drop_certificate(
                connection_context=self.conn_context,
                certificate_name="DIGICERTG5"
            )
        except Exception as exc:
            logger.warning("Failed to drop certificate DIGICERTG5: %s", str(exc))

    def create_remote_source(self, credentials, pse_name: str = "AI_CORE_PSE", remote_source_name: str = "HANA_DISCOVERY_AGENT_CREDENTIALS", create_pse: bool = True):
        """
        Configure the Discovery/Data Agent by creating necessary PSE and remote source.

        Parameters
        ----------
        credentials : dict or str
            The credentials for AI Core service. If str, it should be a credentials filepath in JSON format.
        pse_name : str, optional
            The name of the PSE to be created. Default is "AI_CORE_PSE".
        remote_source_name : str, optional
            The name of the remote source to be created. Default is "HANA_DISCOVERY_AGENT_CREDENTIALS".
        create_pse : bool, optional
            Whether to create the PSE. Default is True.

        Remarks
        -------
        The X1ROOT and DIGICERTG5 certificates will be created if they do not already exist. One can provide the enviroment variables DIGICERTG5_PATH and X1ROOT_PATH to specify the certificate file paths, otherwise the default certificate files will be used.
        """
        if isinstance(credentials, str):
            with open(credentials, 'r') as file:
                credentials = json.load(file)
        _create_ai_core_remote_source(
            connection_context=self.conn_context,
            credentials=credentials,
            pse_name=pse_name,
            remote_source_name=remote_source_name,
            create_pse=create_pse
        )
        self.remote_source_name = remote_source_name
        self.pse_name = pse_name

    def check_remote_source_detailed(self, remote_source_name):
        """
        Check if the remote source exists and retrieve detailed information.
        """
        try:
            cursor = self.conn_context.connection.cursor()

            # Query more detailed information
            sql = """
            SELECT *
            FROM SYS.REMOTE_SOURCES
            WHERE REMOTE_SOURCE_NAME = ?
            """
            cursor.execute(sql, (remote_source_name,))
            result = cursor.fetchone()

            if result:
                return {
                    'exists': True,
                    'details': {
                        'remote_source_name': result[0],
                        'adapter_name': result[1],
                        'connection_info': result[2],
                        'created': result[3],
                        'owner': result[4]
                    }
                }
            else:
                return {'exists': False, 'details': None}

        except Exception as exc:
            return {'exists': False, 'error': str(exc)}

    def run(self, query: str, additional_config: dict = None, show_progress: bool = True):
        """
        Run a query using the Discovery/Data Agent.

        Parameters
        ----------
        query : str
            The query string to be executed.

        additional_config : dict, optional
            Additional configuration parameters for the Discovery/Data Agent.
        Returns
        -------
        result : DataFrame
            The result of the query execution.
        """
        config = {
            "remoteSourceName": self.remote_source_name,
            "knowledgeGraphName": self.knowledge_graph_name,
            "ragSchemaName": self.rag_schema_name,
            "ragTableName": self.rag_table_name
        }
        if additional_config:
            config.update(additional_config)

        if not self.procedure_name:
            raise ValueError("procedure_name must be specified for AgentBase to run()")

        sql_query = _call_agent_sql(
            query=query,
            config=config,
            schema_name=self.schema_name,
            procedure_name=self.procedure_name,
        )

        logger.info("Executing Discovery Agent SQL: %s", sql_query)

        # Get current connection ID
        connection_id = int(self.conn_context.get_connection_id())

        # Used to store result
        result = None
        execution_error = None
        execution_completed = threading.Event()

        def execute_query():
            nonlocal result, execution_error
            try:
                with self.conn_context.connection.cursor() as cursor:
                    cursor.execute(sql_query)
                    logger.info("SQL executed successfully.")
                    logger.info("Fetching result...")
                    query_result = cursor.fetchone()
                    result = query_result[0] if query_result else None
                    logger.info("Result fetched successfully.")
            except Exception as exc:
                execution_error = exc
                logger.error("Error executing query: %s", exc)
            finally:
                execution_completed.set()

        if show_progress:
            # Create progress monitor
            monitor = TextProgressMonitor(
                connection=EmbeddedUI.create_connection_context(self.conn_context).connection,
                connection_id=connection_id,
                show_progress=show_progress
            )

            # Start progress monitoring
            monitor.start()

            try:
                # Start query thread
                query_thread = threading.Thread(target=execute_query)
                query_thread.daemon = True
                query_thread.start()

                # Poll progress until query completes
                while not execution_completed.is_set():
                    monitor.update()
                    time.sleep(monitor.refresh_interval)

                # Wait for query thread to finish
                query_thread.join(timeout=5)

                # Query completed
                if execution_error:
                    monitor.complete(success=False, final_message="Query failed: %s" % str(execution_error)[:100])
                else:
                    monitor.complete(success=True, final_message="Query completed successfully.")

            except KeyboardInterrupt:
                # User interruption
                logger.warning("Query execution interrupted by user")
                monitor.complete(success=False, final_message="interrupted by user")
                raise

            except Exception as exc:
                # Other exceptions
                monitor.complete(success=False, final_message="Error: %s" % str(exc)[:100])
                raise

            finally:
                # Ensure monitor stops
                monitor.stop()

                # Store monitor for later progress history
                self._progress_monitor = monitor

        else:
            # No progress display
            execute_query()

        return result
