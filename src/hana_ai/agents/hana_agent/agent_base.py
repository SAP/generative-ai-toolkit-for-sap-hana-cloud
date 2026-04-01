"""
hana_ml.agents.agent_base
"""
import logging
import threading
import time

from hana_ml.ml_base import MLBase
from hana_ml.visualizers.shared import EmbeddedUI

from .progress_monitor import TextProgressMonitor
from .utility import _call_agent_sql

logger = logging.getLogger(__name__)

class AgentBase(MLBase):
    """
    Base class for calling HANA AI retrieval procedures.
    """
    def __init__(self,
                 connection_context,
                 *,
                 schema_name: str = "SYS",
                 procedure_name: str | None = None,
                 remote_source_name: str = "HANA_DISCOVERY_AGENT_CREDENTIALS",
                 ai_metadata_schema_name: str = "SYSTEM",
                 ai_metadata_object_prefix: str = "HANA_OBJECTS",
                 remote_source_schema_name: str | None = None,
                 model_and_version: str | None = None):
        """
        Initialize the AgentBase.

        Parameters
        ----------
        connection_context : ConnectionContext
            The HANA connection context.
        schema_name : str, optional
            Schema for the target procedure. Default is "SYS".
        procedure_name : str, optional
            The procedure name to call.
        remote_source_name : str, optional
            The name of the remote source to be used. Default is "HANA_DISCOVERY_AGENT_CREDENTIALS".
        ai_metadata_schema_name : str, optional
            The schema that contains the AI metadata objects.
        ai_metadata_object_prefix : str, optional
            Prefix used by the retrieval procedure to resolve metadata objects.
        remote_source_schema_name : str, optional
            Remote source schema name. The current procedures expect this to be null.
        model_and_version : str, optional
            Model and version identifier. The current procedures expect this to be null.
        """
        super().__init__(connection_context)
        self.conn_context = connection_context
        self.schema_name = schema_name
        self.procedure_name = procedure_name
        self.remote_source_name = remote_source_name
        self.ai_metadata_schema_name = ai_metadata_schema_name
        self.ai_metadata_object_prefix = ai_metadata_object_prefix
        self.remote_source_schema_name = remote_source_schema_name
        self.model_and_version = model_and_version

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

    def run(self, query: str, options: dict | str | None = None, show_progress: bool = True):
        """
        Run a query using the Discovery/Data Agent.

        Parameters
        ----------
        query : str
            The query string to be executed.
        options : dict or str, optional
            Optional procedure options passed as NCLOB.
        Returns
        -------
        result : str | None
            The result of the query execution.
        """
        if not self.procedure_name:
            raise ValueError("procedure_name must be specified for AgentBase to run()")

        sql_query = _call_agent_sql(
            remote_source_schema_name=self.remote_source_schema_name,
            remote_source_name=self.remote_source_name,
            ai_metadata_schema_name=self.ai_metadata_schema_name,
            ai_metadata_object_prefix=self.ai_metadata_object_prefix,
            model_and_version=self.model_and_version,
            query=query,
            options=options,
            schema_name=self.schema_name,
            procedure_name=self.procedure_name,
        )

        logger.info("Executing retrieval SQL: %s", sql_query)

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
