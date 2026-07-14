"""
Utility functions for the HANA ML tools.
"""
import os
import shutil
import json
import re
import socket
from pathlib import Path
import logging
from datetime import datetime, date
from typing import Optional, Union, Any
import pandas as pd
from pandas import Timestamp
from numpy import int64
from hana_ml.model_storage import ModelStorage
#pylint: disable=too-many-nested-blocks, unexpected-keyword-arg, invalid-name

logger = logging.getLogger(__name__)

DEFAULT_MCP_SESSION_CONTEXT_KEYS = [
    "EVENT_TYPE",
    "OCCURRED_AT",
    "MCP_SESSION_ID",
    "CLIENT_IP",
    "CLIENT_DECLARED_NAME",
    "CLIENT_DECLARED_AGENT_NAME",
    "CLIENT_DECLARED_MODEL_NAME",
    # HANA-authenticated identity of the connection the tool ran on. Sourced from
    # HANA built-ins (CURRENT_USER / SESSION_USER / CURRENT_CONNECTION) plus the
    # driver's setclientinfo channel (APPLICATION / APPLICATION_USER / CLIENT_HOST).
    # These sit alongside CLIENT_DECLARED_* so an auditor can compare
    # "who the client says it is" vs "who HANA authenticated".
    "HANA_DB_USER",
    "HANA_DB_SESSION_USER",
    "HANA_CONNECTION_ID",
    "HANA_APPLICATION_USER",
    "HANA_APPLICATION",
    "HANA_CLIENT_HOST",
    "TOOL_NAME",
    "TARGET_TABLES",
    "TOOL_ARGS_JSON",
    "RESPONSE_SIZE",
    "MODEL_STORAGE_NAME",
    "MODEL_STORAGE_VERSION",
    "STATUS",
    "DURATION_MS",
    "HANA_CORRELATION_ID",
    "INVOCATION_ID",
    "MCP_CLIENT_NAME",
    "MCP_CLIENT_ID",
    "AI_AGENT_NAME",
    "AI_MODEL_NAME",
]

def convert_cap_to_hdi(source_dir, target_dir, archive=True):
    """
    Convert a CAP project structure to an HDI structure.
    Parameters
    ----------
    source_dir : str
        The source directory containing the CAP project files.
    target_dir : str
        The target directory where the HDI structure will be created.
    archive : bool, optional
        If True, the function will create an archive of the source directory.
        Default is True.
    """
    target_path = Path(target_dir)
    if target_path.exists() and target_path.is_dir():
        if any(target_path.iterdir()):
            if archive:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                archive_path = f"archive_{target_dir}_{timestamp}.tar.gz"
                shutil.make_archive(archive_path, 'gztar', target_dir)
                # delete the target directory after archiving including subdirectories except the archive
                for item in target_path.iterdir():
                    if item.name != f"{target_dir}.tar.gz":
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                logger.info("Created archive: %s", archive_path)
            else:
                logger.info("Target directory %s already exists and is not empty.", target_dir)
                raise FileExistsError(f"The target_dir {target_dir} is not empty. Please provide an empty directory.")
    db_src = os.path.join(Path(target_dir), "db", "src")
    db_cfg = os.path.join(Path(target_dir), "db", "cfg")
    srv_dir = os.path.join(Path(target_dir), "srv")
    os.makedirs(db_src, exist_ok=True)
    os.makedirs(db_cfg, exist_ok=True)
    os.makedirs(srv_dir, exist_ok=True)
    cap_db = Path(os.path.join(Path(source_dir), "db"))
    src_files = Path(os.path.join(cap_db, "src")).glob("*")
    for file in src_files:
        if file.suffix == ".cds":
            target_file = os.path.join(db_src, f"{file.stem}.hdbcds")
            shutil.copy2(file, target_file)
        else:
            shutil.copy2(file, os.path.join(db_src, file.name))
    for cds_file in cap_db.glob("*.cds"):
        target_file = os.path.join(db_src, f"{cds_file.stem}.hdbcds")
        shutil.copy2(cds_file, target_file)
    srv_source = Path(os.path.join(Path(source_dir), "srv"))
    if srv_source.exists():
        shutil.copytree(srv_source, srv_dir, dirs_exist_ok=True)
    hdi_config = os.path.join(db_cfg, ".hdiconfig")
    with open(hdi_config, "w") as f:
        json.dump({
            "file": {
                "path": os.path.join("db", "src"),
                "build_plugins": [
                    {"plugin": "com.sap.hana.di.cds"},
                    {"plugin": "com.sap.hana.di.procedure"},
                    {"plugin": "com.sap.hana.di.synonym"},
                    {"plugin": "com.sap.hana.di.grant"}
                ]
            }
        }, f, indent=2)

class _CustomEncoder(json.JSONEncoder):
    """
    This class is used to encode the model attributes into JSON string.
    """
    def default(self, obj): #pylint: disable=arguments-renamed
        if isinstance(obj, (Timestamp, datetime, date)):
            # Convert Timestamp, datetime or date to ISO string
            return obj.isoformat()
        elif isinstance(obj, (int64, int)):
            # Convert numpy int64 or Python int to Python int
            return int(obj)
        # Let other types use the default handler
        return super().default(obj)


def add_stopping_hint(x : str):
    """Added the hint for stopping the execution when an error message is returned."""
    return (x + ". Please stop the execution and return.").replace("..", ".")


def _hana_safe_identifier(text: Any) -> str:
    """Normalize a segment used to build a HANA table identifier.

    HANA folds unquoted identifiers to upper case at parse time, but the
    ``smart_save``/``save`` helpers in ``hana_ml`` quote the target table name.
    That means when a user-supplied fragment like ``my_hana_ai_model`` is
    embedded verbatim into a table identifier, the table is created
    case-sensitively (``..._my_hana_ai_model_8``) yet any downstream
    ``SELECT ... FROM PREDICT_RESULT_..._my_hana_ai_model_8`` written without
    quotes is folded to ``..._MY_HANA_AI_MODEL_8`` and no longer matches.

    Uppercasing every fragment before assembling the identifier keeps the
    stored table name aligned with HANA's default folding so that downstream
    unquoted references (issued by the agent, tools, or SQL written by the
    user) resolve without needing to double-quote the name. Already-uppercase
    inputs are idempotent under this transform.
    """
    return str(text).upper() if text is not None else text

def generate_model_storage_version(ms : ModelStorage, version: Union[int, str, None], name: str) -> int:
    """Generate the model storage version."""
    ms._create_metadata_table()
    if version is None:
        version = ms._get_new_version_no(name)
        if version is None:
            version = 1
        else:
            version = int(version)
    return version

def _create_temp_table(conn, select_statement: str, tool_name: str, additional_info: str = None) -> str:
    """
    Create a temporary table in the HANA database.
    Parameters
    ----------
    conn : Connection
        The HANA connection object.
    select_statement : str
        The SQL select statement to create the temporary table.
    tool_name : str
        The name of the tool to create a unique temporary table name.
    additional_info : str, optional
        Additional information to append to the table name.
    Returns
    -------
    str
        The SQL statement to select from the temporary table.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    if additional_info:
        additional_info = f"_{additional_info}_"
    else:
        additional_info = "_"
    table_name = f"#{tool_name}{additional_info}{timestamp}".upper()
    create_temp_table_sql = f"CREATE LOCAL TEMPORARY TABLE {table_name} AS ({select_statement})"
    conn.execute_sql(create_temp_table_sql)
    return f"SELECT * FROM {table_name}"


def normalize_column_list(columns: Union[None, str, list, tuple]) -> list[str]:
    """Normalize optional column input into a flat ordered list of column names."""
    if columns is None:
        return []
    if isinstance(columns, str):
        parts = [part.strip() for part in re.split(r"[;,]", columns) if part.strip()]
        return parts if parts else [columns.strip()]
    normalized: list[str] = []
    for column in columns:
        text = str(column).strip()
        if text:
            normalized.append(text)
    return normalized


def is_predict_feature_mismatch_error(exc: Exception) -> bool:
    """Detect PAL/HANA errors that indicate predict-table features do not match the trained model."""
    text = str(exc).lower()
    markers = (
        "feature number of predict table does not match the trained model",
        "predict table features do not match the trained model",
        "predict table does not match the trained model",
        "invalid table:$tab$",
        "73001007",
    )
    return any(marker in text for marker in markers)


def build_repaired_predict_dataframe(predict_df, *, key: str, exog=None, group_key: Optional[str] = None, add_placeholder: bool = False):
    """Return a predict DataFrame containing only the columns required for inference.

    The returned tuple is (dataframe, kept_columns, missing_columns).
    """
    required_columns: list[str] = []
    for column in [group_key, key, *normalize_column_list(exog)]:
        if column and column not in required_columns:
            required_columns.append(column)

    missing_columns = [column for column in required_columns if column not in predict_df.columns]
    if missing_columns:
        return predict_df, required_columns, missing_columns

    repaired_df = predict_df.select(*required_columns)
    if add_placeholder and len(repaired_df.columns) == 1:
        repaired_df = repaired_df.add_constant("PLACEHOLDER", 0)
    return repaired_df, required_columns, []


def format_predict_mismatch_diagnostic(*, predict_table: str, predict_schema: Optional[str], original_columns: list[str], kept_columns: list[str], missing_columns: list[str], key: str, exog=None, group_key: Optional[str] = None, original_error: Optional[str] = None) -> str:
    """Build a structured error payload for predict-table schema mismatches."""
    context_columns = [column for column in [group_key, key, *normalize_column_list(exog)] if column]
    analysis = (
        "The predict table structure does not match the trained model. "
        "For forecasting prediction, the predict input should usually contain only the time key"
        + (", the group key" if group_key else "")
        + (", and any explicit exogenous columns." if context_columns else ".")
    )
    payload = {
        "error": "Prediction table features do not match the trained model.",
        "error_category": "predict_table_feature_mismatch",
        "input_predict_table": predict_table,
        "input_predict_schema": predict_schema,
        "predict_table_columns": original_columns,
        "columns_required_for_retry": kept_columns,
        "missing_required_columns": missing_columns,
        "analysis": analysis,
        "suggested_fix": "Create or use a predict table that contains only the required columns and retry the prediction.",
    }
    if original_error:
        payload["original_error"] = original_error
    return json.dumps(payload, cls=_CustomEncoder)


def find_free_port(start: int = 8600, end: int = 8700) -> int:
    """Return an available localhost TCP port."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def ensure_mcp_audit_log(audit_log_path: str = "logs/mcp-audit.jsonl") -> Path:
    """Ensure the MCP audit JSONL file exists and return its resolved path."""
    log_path = Path(audit_log_path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    return log_path


def fetch_mcp_audit_rows(audit_log_path: str, session_id: str):
    """Fetch audit rows for a given MCP session id from the JSONL audit log."""
    log_path = ensure_mcp_audit_log(audit_log_path)
    rows: list[dict[str, Any]] = []

    with log_path.open("r", encoding="utf-8") as log_file:
        for raw_line in log_file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            session = event.get("session", {}) or {}
            if session.get("mcp_session_id") != session_id:
                continue

            correlation = event.get("correlation", {}) or {}
            payload = event.get("payload", {}) or {}
            rows.append(
                {
                    "EVENT_TYPE": event.get("event_type"),
                    "OCCURRED_AT": event.get("occurred_at"),
                    "MCP_SESSION_ID": session.get("mcp_session_id"),
                    "CLIENT_IP": session.get("client_ip"),
                    "CLIENT_DECLARED_NAME": session.get("client_declared_name"),
                    "CLIENT_DECLARED_AGENT_NAME": session.get("client_declared_agent_name"),
                    "CLIENT_DECLARED_MODEL_NAME": session.get("client_declared_model_name"),
                    "HANA_DB_USER": session.get("hana_db_user"),
                    "HANA_DB_SESSION_USER": session.get("hana_db_session_user"),
                    "HANA_CONNECTION_ID": session.get("hana_connection_id"),
                    "HANA_APPLICATION_USER": session.get("hana_application_user"),
                    "HANA_APPLICATION": session.get("hana_application"),
                    "HANA_CLIENT_HOST": session.get("hana_client_host"),
                    "TOOL_NAME": payload.get("tool_name"),
                    "TARGET_TABLES": payload.get("target_tables"),
                    "TOOL_ARGS_JSON": payload.get("tool_args_json"),
                    "RESPONSE_SIZE": payload.get("response_size"),
                    "MODEL_STORAGE_NAME": payload.get("model_storage_name"),
                    "MODEL_STORAGE_VERSION": payload.get("model_storage_version"),
                    "STATUS": payload.get("status"),
                    "DURATION_MS": payload.get("duration_ms"),
                    "HANA_CORRELATION_ID": correlation.get("hana_correlation_id"),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "EVENT_TYPE",
                "OCCURRED_AT",
                "MCP_SESSION_ID",
                "CLIENT_IP",
                "CLIENT_DECLARED_NAME",
                "CLIENT_DECLARED_AGENT_NAME",
                "CLIENT_DECLARED_MODEL_NAME",
                "HANA_DB_USER",
                "HANA_DB_SESSION_USER",
                "HANA_CONNECTION_ID",
                "HANA_APPLICATION_USER",
                "HANA_APPLICATION",
                "HANA_CLIENT_HOST",
                "TOOL_NAME",
                "TARGET_TABLES",
                "TOOL_ARGS_JSON",
                "RESPONSE_SIZE",
                "MODEL_STORAGE_NAME",
                "MODEL_STORAGE_VERSION",
                "STATUS",
                "DURATION_MS",
                "HANA_CORRELATION_ID",
            ]
        )

    audit_rows = pd.DataFrame(rows)
    audit_rows["OCCURRED_AT"] = pd.to_datetime(audit_rows["OCCURRED_AT"], errors="coerce")
    return audit_rows.sort_values(by="OCCURRED_AT", ascending=False).reset_index(drop=True)


def fetch_hana_session_context(connection, keys: Optional[list[str]] = None) -> pd.DataFrame:
    """Fetch selected HANA SESSION_CONTEXT values into a single-row DataFrame."""
    selected_keys = [str(key) for key in (keys or DEFAULT_MCP_SESSION_CONTEXT_KEYS)]
    if not selected_keys:
        raise ValueError("keys must contain at least one session context name.")

    select_sql = "SELECT " + ", ".join(
        "SESSION_CONTEXT('{literal}') AS \"{identifier}\"".format(
            literal=key.replace("'", "''"),
            identifier=key.replace('"', '""'),
        )
        for key in selected_keys
    ) + " FROM DUMMY"

    cursor = connection.cursor()
    try:
        cursor.execute(select_sql)
        row = cursor.fetchone()
    finally:
        cursor.close()

    if row is None:
        return pd.DataFrame([{key: None for key in selected_keys}])

    return pd.DataFrame(
        [{
            key: (str(row[idx]) if row[idx] is not None else None)
            for idx, key in enumerate(selected_keys)
        }]
    )


class SyncHTTPMCPClient:
    """Synchronous JSON-RPC MCP client for notebook/demo flows."""

    def __init__(self, base_url: str, timeout: int = 60):
        import httpx

        normalized = base_url.rstrip("/")
        if not normalized.endswith("/mcp"):
            normalized = normalized + "/mcp"
        self.base_url = normalized.rstrip("/")
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self.tools: dict[str, dict[str, Any]] = {}
        self._httpx = httpx
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            follow_redirects=True,
            trust_env=False,
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "mcp-protocol-version": "2024-11-05",
            },
        )

    def initialize(
        self,
        *,
        client_name: str,
        client_version: str = "0.1",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Send the MCP initialize request and refresh the local tool cache."""
        client_name = (client_name or "").strip()
        if not client_name:
            raise ValueError("client_name is required and must be a non-empty user name.")

        identity_metadata = dict(metadata or {})
        headers = {
            "x-mcp-client-name": client_name,
            "x-mcp-client-version": client_version,
        }
        if identity_metadata.get("client_id"):
            headers["x-mcp-client-id"] = str(identity_metadata["client_id"])
        if identity_metadata.get("agent_name"):
            headers["x-ai-agent-name"] = str(identity_metadata["agent_name"])
        if identity_metadata.get("model_name"):
            headers["x-ai-model-name"] = str(identity_metadata["model_name"])
        if identity_metadata.get("model_version"):
            headers["x-ai-model-version"] = str(identity_metadata["model_version"])
        self.client.headers.update(headers)

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": client_name,
                    "version": client_version,
                },
                "metadata": identity_metadata,
            },
        }
        response = self.client.post("", json=payload)
        response.raise_for_status()
        self.session_id = response.headers.get("mcp-session-id")
        if self.session_id:
            self.client.headers["mcp-session-id"] = self.session_id
        self.list_tools(force_refresh=True)

    def list_tools(self, force_refresh: bool = False) -> list[dict[str, Any]]:
        """Return the cached MCP tool list, optionally refreshing it from the server."""
        if self.tools and not force_refresh:
            return list(self.tools.values())

        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": ({"session": {"id": self.session_id}} if self.session_id else {}),
        }
        response = self.client.post("", json=payload)
        response.raise_for_status()
        result = response.json().get("result", {})
        tool_list = result.get("tools", []) or []
        self.tools = {tool["name"]: tool for tool in tool_list}
        return tool_list

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invoke ``tool_name`` on the MCP server with ``arguments`` and return the flattened text result."""
        payload = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
                **({"session": {"id": self.session_id}} if self.session_id else {}),
            },
        }
        response = self.client.post("", json=payload)
        response.raise_for_status()
        rpc_response = response.json()
        if "error" in rpc_response:
            raise RuntimeError(str(rpc_response["error"]))

        result_data = rpc_response.get("result", {})
        content = result_data.get("content", [])
        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(part for part in text_parts if part) or json.dumps(result_data, ensure_ascii=False)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()


def _json_type_to_python(spec: dict[str, Any]) -> Any:
    json_type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    json_type = spec.get("type")
    if json_type == "array":
        return list[Any]
    if json_type == "object":
        return dict[str, Any]
    return json_type_map.get(json_type, Any)


def _schema_to_model(tool_name: str, input_schema: dict[str, Any]):
    from pydantic import Field, create_model

    properties = input_schema.get("properties", {}) or {}
    required = set(input_schema.get("required", []) or [])
    fields: dict[str, tuple[Any, Any]] = {}

    for field_name, field_spec in properties.items():
        annotation = _json_type_to_python(field_spec)
        description = field_spec.get("description", "")
        if field_name in required:
            fields[field_name] = (annotation, Field(..., description=description))
        else:
            default = field_spec.get("default", None)
            fields[field_name] = (annotation | None, Field(default=default, description=description))

    if not fields:
        return create_model(f"{tool_name.title().replace('_', '')}Input")

    return create_model(f"{tool_name.title().replace('_', '')}Input", **fields)


def build_context_agent_mcp_tools(
    base_url: str,
    *,
    timeout: int = 120,
    client_name: str,
    client_version: str = "0.1",
    client_metadata: Optional[dict[str, Any]] = None,
    skip_admin_tools: bool = True,
):
    """Build LangChain-compatible tools backed by an HTTP MCP server.

    Returns a tuple of (tools, client).
    """
    from langchain_core.tools import StructuredTool

    client = SyncHTTPMCPClient(base_url=base_url, timeout=timeout)
    client.initialize(
        client_name=client_name,
        client_version=client_version,
        metadata=client_metadata,
    )

    tools = []
    for remote_tool in client.list_tools():
        tool_name = remote_tool["name"]
        if skip_admin_tools and tool_name.startswith("admin_"):
            continue

        args_schema = _schema_to_model(tool_name, remote_tool.get("inputSchema", {}))

        def _invoke(_tool_name: str = tool_name, **kwargs):
            filtered_kwargs = {key: value for key, value in kwargs.items() if value is not None}
            return client.call_tool(_tool_name, filtered_kwargs)

        structured_tool = StructuredTool.from_function(
            func=_invoke,
            name=tool_name,
            description=remote_tool.get("description", tool_name),
            args_schema=args_schema,
        )
        tools.append(structured_tool)

    return tools, client
