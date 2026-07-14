"""
hana_ai.retrieval.retrieval_base
"""
import logging
import threading
import time

from hana_ml.ml_base import MLBase
from hana_ml.visualizers.shared import EmbeddedUI

from .progress_monitor import TextProgressMonitor
from .utility import _call_procedure_sql

logger = logging.getLogger(__name__)

class RetrievalBase(MLBase):
    """
    Base class for HANA AI Core retrieval procedures (object discovery, data retrieval).
    """
    def __init__(self, connection_context, agent_type=None,
                 schema_name: str = "SYS",
                 procedure_name: str | None = None,
                 remote_source_name: str = None,
                 knowledge_graph_name: str = None,
                 rag_schema_name: str = None,
                 rag_table_name: str = None,
                 metadata_schema_name: str | None = None,
                 metadata_object_prefix: str | None = None):
        """
        Initialize the RetrievalBase.

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
            The name of the remote source to be used.
        knowledge_graph_name : str, optional
            The name of the knowledge graph to be used.
        rag_schema_name : str, optional
            The name of the RAG schema to be used. When unset and
            ``metadata_schema_name`` is provided, this defaults to
            ``metadata_schema_name``.
        rag_table_name : str, optional
            The name of the RAG table to be used. When unset and
            ``metadata_object_prefix`` is provided, this defaults to
            ``f"{metadata_object_prefix}_INDEX"``.
        metadata_schema_name : str, optional
            Value for the procedure's ``AI_METADATA_SCHEMA_NAME`` IN parameter
            (no-wrapper deployments only). When set together with
            ``metadata_object_prefix``, the CALL switches to named-parameter
            form **and** the unset ones of ``rag_schema_name`` /
            ``rag_table_name`` / ``knowledge_graph_name`` are auto-derived
            from the SYSTEM-side naming convention:

                rag_schema_name      <- metadata_schema_name
                rag_table_name       <- f"{metadata_object_prefix}_INDEX"
                knowledge_graph_name <- f"{metadata_schema_name}.{metadata_object_prefix}_GRAPH"

            Leave as ``None`` to keep the original positional CALL used by
            customer wrapper procedures.
        metadata_object_prefix : str, optional
            Value for ``AI_METADATA_OBJECT_PREFIX``. See ``metadata_schema_name``.
        """
        super().__init__(connection_context)
        self.conn_context = connection_context
        self.schema_name = schema_name
        self.procedure_name = procedure_name
        self.remote_source_name = remote_source_name
        self.metadata_schema_name = metadata_schema_name
        self.metadata_object_prefix = metadata_object_prefix

        # Auto-derive RAG / KG names from the metadata schema + prefix when
        # the caller did not override them explicitly. The procedure itself
        # also reads these names from the metadata catalog, but emitting
        # them in the CONFIG blob keeps the wire format symmetric with the
        # wrapper path and gives downstream auditing concrete values.
        derived_rag_schema = metadata_schema_name
        derived_rag_table = (
            f"{metadata_object_prefix}_INDEX" if metadata_object_prefix else None
        )
        derived_kg = None
        if metadata_schema_name and metadata_object_prefix:
            derived_kg = f"{metadata_schema_name}.{metadata_object_prefix}_GRAPH"

        self.rag_schema_name = rag_schema_name if rag_schema_name is not None else derived_rag_schema
        self.rag_table_name = rag_table_name if rag_table_name is not None else derived_rag_table
        self.knowledge_graph_name = knowledge_graph_name if knowledge_graph_name is not None else derived_kg

    def run(self, query: str, additional_config: dict = None, show_progress: bool = True):
        """
        Run a query using the Object Discovery / Data Retrieval procedure.

        Parameters
        ----------
        query : str
            The query string to be executed.

        additional_config : dict, optional
            Additional configuration parameters for the retrieval procedure.
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
            raise ValueError("procedure_name must be specified for RetrievalBase to run()")

        sql_query = _call_procedure_sql(
            query=query,
            config=config,
            schema_name=self.schema_name,
            procedure_name=self.procedure_name,
            metadata_schema_name=self.metadata_schema_name,
            metadata_object_prefix=self.metadata_object_prefix,
        )

        logger.info("Executing retrieval procedure SQL: %s", sql_query)

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

        monitor = None
        if show_progress:
            # Try to create progress monitor. EmbeddedUI.create_connection_context
            # clones the cc with password="" when sslKeyStore is set on the
            # original — that path assumes keystore-based auth and breaks when
            # the keystore is only used for CA trust (e.g. certifi.where()) and
            # the user authenticates with username/password. If the clone or
            # monitor setup fails for any reason, degrade gracefully to the
            # no-progress path rather than aborting the whole query.
            try:
                monitor = TextProgressMonitor(
                    connection=EmbeddedUI.create_connection_context(self.conn_context).connection,
                    connection_id=connection_id,
                    show_progress=show_progress
                )
            except Exception as exc:
                logger.warning(
                    "Progress monitor unavailable (%s); running without progress display.",
                    exc,
                )

        if show_progress and monitor is not None:
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

        if execution_error is not None:
            # Re-raise the underlying DB error so callers (and the MCP tool
            # wrapper) see the real cause instead of an empty result.
            raise execution_error

        return result
