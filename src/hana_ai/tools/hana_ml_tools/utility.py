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

# ---------------------------------------------------------------------------
# APPLICATIONSOURCE pack (setclientinfo channel)
# ---------------------------------------------------------------------------
# HANA setclientinfo 'APPLICATIONSOURCE' is capped at 256 wire bytes and the
# server silently truncates past that (verified against HANA Cloud 4.00 with
# hdbcli 2.27). Non-ASCII UTF-8 input gets cut mid-character; astral / surrogate
# pairs behave erratically. We therefore pack ASCII-only content, honour a hard
# byte budget, and let the JSONL / HANA sink carry the full audit event.
#
# Layout (pipe-delimited K=V, order fixed so cheap SUBSTR_BEFORE parsing works):
#
#   mcp=hana-ai/<ver>|sess=<id>|agent=<n>|model=<n>|cli=<n>|mcp_ip=<ip>
#     |tool=<n>|inv=<id>|corr=<id>|resp=<int>?
#
# All values are ASCII tokens (a conservative whitelist below); ``resp`` is a
# decimal integer only emitted on the post-completion "beacon" write when the
# tool's response_size is known. The redacted tool-args payload used to be
# packed here too (base64) but the 254-byte cap made it None in most real
# traffic — auditors now read the full args from the JSONL sink instead.
APPLICATIONSOURCE_MAX_BYTES = 254  # 256 wire cap minus a 2-byte cushion.
APPLICATIONSOURCE_FIELD_ORDER = (
    "mcp", "sess", "agent", "model", "cli", "mcp_ip",
    "tool", "inv", "corr", "resp",
)
_APP_SOURCE_SAFE_VALUE_RE = re.compile(r"[^A-Za-z0-9._:/@\-]")


def _sanitize_appsource_value(value: Any) -> str:
    """Coerce an arbitrary value to an ASCII token safe for the pack."""
    if value is None:
        return ""
    text = str(value)
    # Replace anything outside a conservative whitelist with '_' so a malicious
    # or accidental '|' / '=' / space cannot break the pack framing.
    return _APP_SOURCE_SAFE_VALUE_RE.sub("_", text)


def _sanitize_response_size(value: Any) -> str:
    """Coerce ``response_size`` to a non-negative decimal token or empty string.

    Mirrors the semantics of ``ToolkitMCPMixin._normalize_response_size`` in
    ``toolkit.py`` — accepts ``int`` (but not ``bool``), int-valued ``float``,
    and digit-only ``str``. Any other value returns "" so the pack builder
    omits the ``resp=`` segment.
    """
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, int):
        as_int = value
    elif isinstance(value, float) and value.is_integer():
        as_int = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        as_int = int(value.strip())
    else:
        return ""
    if as_int < 0:
        return ""
    return str(as_int)


def build_appsource_pack(
    *,
    mcp_version: Optional[str] = None,
    mcp_session_id: Optional[str] = None,
    client_declared_name: Optional[str] = None,
    client_declared_agent_name: Optional[str] = None,
    client_declared_model_name: Optional[str] = None,
    client_ip: Optional[str] = None,
    tool_name: Optional[str] = None,
    invocation_id: Optional[str] = None,
    hana_correlation_id: Optional[str] = None,
    response_size: Optional[int] = None,
    max_bytes: int = APPLICATIONSOURCE_MAX_BYTES,
) -> str:
    """Build the pipe-delimited pack for setclientinfo('APPLICATIONSOURCE').

    All identity fields (``mcp``, ``sess``, ``agent``, ``model``, ``cli``,
    ``mcp_ip``, ``tool``, ``inv``, ``corr``) are packed in fixed order; each
    is sanitized to an ASCII token (`[A-Za-z0-9._:/@-]`). ``resp`` (a decimal
    integer, tool response_size) is optional and packed last; it is only known
    after the tool body runs, so the started-path pack has no ``resp`` and the
    success-path "beacon" pack adds it. When identity plus ``resp`` overflows
    the 254-byte cap, ``resp`` is dropped first (tail-first truncation via
    ``_truncate_on_pipe``); auditors can still recover the value from the
    JSONL sink.

    The returned string is ASCII and its byte length is guaranteed to be
    ``<= max_bytes`` (verified against HANA's 256-byte setclientinfo cap).
    """
    identity_pairs = [
        ("mcp", _sanitize_appsource_value(f"hana-ai/{mcp_version}") if mcp_version else ""),
        ("sess", _sanitize_appsource_value(mcp_session_id)),
        ("agent", _sanitize_appsource_value(client_declared_agent_name)),
        ("model", _sanitize_appsource_value(client_declared_model_name)),
        ("cli", _sanitize_appsource_value(client_declared_name)),
        ("mcp_ip", _sanitize_appsource_value(client_ip)),
        ("tool", _sanitize_appsource_value(tool_name)),
        ("inv", _sanitize_appsource_value(invocation_id)),
        ("corr", _sanitize_appsource_value(hana_correlation_id)),
        ("resp", _sanitize_response_size(response_size)),
    ]
    parts = [f"{k}={v}" for k, v in identity_pairs if v]
    pack = "|".join(parts)

    # Tail-first truncation on '|' boundaries — drops ``resp`` first when the
    # identity block itself is already near the cap. This preserves the
    # invariant that every emitted pack is ASCII, byte-length <= max_bytes,
    # and every K=V segment is complete (never a half-written tail).
    return _truncate_on_pipe(pack, max_bytes)


def _truncate_on_pipe(pack: str, max_bytes: int) -> str:
    """Truncate a pipe-delimited pack on a '|' boundary within ``max_bytes``.

    Guarantees the returned string is valid ASCII, byte-length <= max_bytes,
    and never leaves a half-written ``K=V`` segment at the tail.
    """
    if len(pack.encode("ascii")) <= max_bytes:
        return pack
    parts = pack.split("|")
    kept: list[str] = []
    running = 0
    for part in parts:
        # +1 for the '|' separator between parts (skipped before the first one).
        sep = 1 if kept else 0
        if running + sep + len(part.encode("ascii")) > max_bytes:
            break
        kept.append(part)
        running += sep + len(part.encode("ascii"))
    return "|".join(kept)


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
    """Fetch audit rows for a given MCP session id from the JSONL audit log.

    In addition to the flat MCP audit event columns, each row includes
    ``APPLICATIONSOURCE_PACK`` — the pipe-delimited pack that would land in
    HANA's setclientinfo APPLICATIONSOURCE for that invocation. This makes the
    JSONL sink self-describing for the "what did HANA see?" question without a
    separate live connection.
    """
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
            appsource_pack = build_appsource_pack(
                mcp_session_id=session.get("mcp_session_id"),
                client_declared_name=session.get("client_declared_name"),
                client_declared_agent_name=session.get("client_declared_agent_name"),
                client_declared_model_name=session.get("client_declared_model_name"),
                client_ip=session.get("client_ip"),
                tool_name=payload.get("tool_name"),
                invocation_id=correlation.get("invocation_id"),
                hana_correlation_id=correlation.get("hana_correlation_id"),
                tool_args_json=payload.get("tool_args_json"),
            )
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
                    "INVOCATION_ID": correlation.get("invocation_id"),
                    "HANA_CORRELATION_ID": correlation.get("hana_correlation_id"),
                    "APPLICATIONSOURCE_PACK": appsource_pack or None,
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
                "INVOCATION_ID",
                "HANA_CORRELATION_ID",
                "APPLICATIONSOURCE_PACK",
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


def parse_appsource_pack(pack: Optional[str]) -> dict:
    """Parse the pipe-delimited APPLICATIONSOURCE pack into a dict.

    Returns a dict of the identity K=V pairs (``mcp``, ``sess``, ``agent``,
    ``model``, ``cli``, ``mcp_ip``, ``tool``, ``inv``, ``corr``) and, when
    present, ``response_size`` (parsed from the ``resp=`` segment written by
    the post-completion beacon; missing on started-path packs).
    """
    result: dict[str, Any] = {}
    if not pack:
        return result
    for segment in pack.split("|"):
        if "=" not in segment:
            continue
        key, _, value = segment.partition("=")
        key = key.strip()
        if not key:
            continue
        if key == "resp":
            if value.isdigit():
                result["response_size"] = int(value)
            continue
        result[key] = value
    return result


def fetch_hana_appsource_pack(connection) -> dict:
    """Read APPLICATIONSOURCE off the current HANA connection and parse it.

    Convenience wrapper around ``SESSION_CONTEXT('APPLICATIONSOURCE')`` +
    ``parse_appsource_pack``. Returns an empty dict when APPLICATIONSOURCE
    is unset. Useful when hand-inspecting a live MCP session's HANA-side
    context without opening the JSONL / HANA audit sink.
    """
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT SESSION_CONTEXT('APPLICATIONSOURCE') FROM DUMMY")
        row = cursor.fetchone()
    finally:
        cursor.close()
    if not row or row[0] is None:
        return {}
    value = row[0]
    if isinstance(value, (memoryview, bytes, bytearray)):
        value = bytes(value).decode("utf-8", errors="replace")
    return parse_appsource_pack(str(value))


HANA_MCP_AUDIT_VIEW_COLUMNS = [
    "LAST_EXECUTION_TIMESTAMP",
    "MCP_SESSION_ID",
    "INVOCATION_ID",
    "HANA_CORRELATION_ID",
    "TOOL_NAME",
    "RESPONSE_SIZE",
    "AGENT_NAME",
    "MODEL_NAME",
    "CLIENT_DECLARED_NAME",
    "CLIENT_IP",
    "MCP_VERSION",
    "HANA_AUTHENTICATED_USER",
    "HANA_SESSION_USER",
    "STATEMENT_HASH",
    "EXECUTION_COUNT",
    "APPLICATION_NAME",
    "APPLICATION_SOURCE",
]


# Marker embedded in the beacon SQL comment so we can filter beacon rows
# out of the returned audit view. Any statement-string containing this token
# is a synthetic plan-cache entry emitted by ``_emit_beacon_sql`` in
# ``toolkit.py`` after a successful tool call — its only purpose is to carry
# ``resp=<N>`` in its APPLICATION_SOURCE pack for the ``fetch_hana_mcp_audit_view``
# to project back onto the real tool-call rows via INVOCATION_ID.
MCP_BEACON_SQL_MARKER = "mcp-audit-beacon"


def fetch_hana_mcp_audit_view(
    connection,
    *,
    mcp_session_id: Optional[str] = None,
    application_name: Optional[str] = None,
    tool_name: Optional[str] = None,
    since_seconds: Optional[int] = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Query ``M_SQL_PLAN_CACHE`` for MCP-tagged SQL and decode the pack.

    Returns a DataFrame of :data:`HANA_MCP_AUDIT_VIEW_COLUMNS`, one row per
    plan-cache entry produced by an MCP tool call visible on this HANA
    instance. Each row's identity fields (``MCP_SESSION_ID``, ``TOOL_NAME``,
    ``INVOCATION_ID``, ``AGENT_NAME`` etc.) are decoded from the pack in
    ``M_SQL_PLAN_CACHE.APPLICATION_SOURCE``.

    This gives an auditor sitting on the HANA side, with only ``SELECT`` on
    ``M_SQL_PLAN_CACHE``, a picture of which MCP session / agent / tool /
    invocation first produced each cached SQL plan — without needing HANA
    audit policies to be enabled and without touching the MCP host at all.

    Parameters
    ----------
    connection
        A live HANA ``dbapi.Connection``. Any connection with SELECT on
        ``M_SQL_PLAN_CACHE`` works — it does not have to be the same
        connection that ran the MCP tools.
    mcp_session_id : str, optional
        Filter to a single MCP session id. Matches the ``sess=`` field.
    application_name : str, optional
        Filter by the setclientinfo ``APPLICATION`` value (MCP client's
        declared name). Useful when several MCP fleets share one HANA
        landscape.
    tool_name : str, optional
        Filter to a specific tool invocation. Matches the pack's ``tool=``
        field via ``APPLICATION_SOURCE LIKE`` so schema differences across
        HANA builds don't matter.
    since_seconds : int, optional
        Only rows with ``LAST_EXECUTION_TIMESTAMP`` newer than this many
        seconds ago. ``None`` returns everything the plan cache still holds.
    limit : int
        Maximum rows returned. Default 200. Set to a high number if you
        expect to paginate on the client side.

    Notes
    -----
    * **First-execution-wins semantics**: HANA populates
      ``M_SQL_PLAN_CACHE.APPLICATION_SOURCE`` from the first process that
      caused a given plan to be cached. Subsequent executions of the
      *same* SQL update ``EXECUTION_COUNT`` and ``LAST_EXECUTION_TIMESTAMP``
      but leave the pack untouched. Different tools produce different SQL
      so each tool's rows carry that tool's own pack, but repeated calls
      to the *same* tool from *different* MCP sessions all appear under
      the first session's pack. This is a HANA server-side property; no
      client-side merge with local files is done here (audit is
      server-side by contract).
    * **``RESPONSE_SIZE`` population**: only known *after* the tool body
      runs, so it cannot travel in the started-path pack that HANA freezes
      onto the tool's own plan-cache rows. Instead, after each successful
      tool call the MCP host executes a synthetic "beacon" statement
      (``SELECT ... mcp-audit-beacon inv=<invocation_id> ...``) whose
      APPLICATION_SOURCE carries ``resp=<N>``. This function extracts that
      ``resp=`` per invocation and left-joins it back onto every row of the
      same ``INVOCATION_ID`` (so the tool's real SQL rows show
      ``RESPONSE_SIZE`` too). The beacon rows themselves are filtered out
      of the returned DataFrame — auditors see one row per real cached
      plan, with ``RESPONSE_SIZE`` populated when the beacon fired.
      Failed tool calls have no beacon, so ``RESPONSE_SIZE`` stays
      ``None`` — check the JSONL sink or the audit sink to distinguish
      "beacon evicted / not yet fired" from "tool failed before returning".
    * **Plan cache eviction**: entries age out on their own schedule. For
      long-term retention on the HANA side, enable HANA audit policies so
      the pack is recorded in ``AUDIT_LOG`` at each SQL execution rather
      than only at plan-caching time.
    * **APPLICATION_USER_NAME**: not exposed on every HANA build; this
      function uses ``USER_NAME`` (HANA-authenticated) and
      ``APPLICATION_NAME`` (setclientinfo APPLICATION) which are portable.
    """
    predicates = ["APPLICATION_SOURCE LIKE 'mcp=hana-ai/%'"]
    params: list[Any] = []
    if mcp_session_id:
        predicates.append("APPLICATION_SOURCE LIKE ?")
        params.append(f"%sess={mcp_session_id}%")
    if application_name:
        predicates.append("APPLICATION_NAME = ?")
        params.append(application_name)
    if tool_name:
        predicates.append("APPLICATION_SOURCE LIKE ?")
        params.append(f"%tool={tool_name}%")
    if since_seconds is not None:
        predicates.append("LAST_EXECUTION_TIMESTAMP > ADD_SECONDS(CURRENT_TIMESTAMP, ?)")
        params.append(-int(since_seconds))

    where_clause = " AND ".join(predicates)
    # STATEMENT_STRING is projected so we can filter out synthetic beacon
    # statements (see MCP_BEACON_SQL_MARKER); it is intentionally NOT
    # returned in the DataFrame — auditors who want the SQL text should
    # query M_SQL_PLAN_CACHE directly.
    sql = (
        "SELECT LAST_EXECUTION_TIMESTAMP, "
        "       USER_NAME, "
        "       SESSION_USER_NAME, "
        "       APPLICATION_NAME, "
        "       STATEMENT_HASH, "
        "       EXECUTION_COUNT, "
        "       APPLICATION_SOURCE, "
        "       STATEMENT_STRING "
        "FROM   M_SQL_PLAN_CACHE "
        f"WHERE  {where_clause} "
        "ORDER  BY LAST_EXECUTION_TIMESTAMP DESC "
        f"LIMIT  {int(limit)}"
    )

    cursor = connection.cursor()
    try:
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        rows = cursor.fetchall()
    finally:
        cursor.close()

    records: list[dict[str, Any]] = []
    # First pass: parse every row, remember which rows are beacons, and
    # collect the beacon-provided response_size per invocation id.
    beacon_indices: set[int] = set()
    resp_by_invocation: dict[str, int] = {}
    for idx, row in enumerate(rows):
        (last_ts, user_name, session_user, app_name,
         stmt_hash, exec_count, app_source, stmt_string) = row
        if isinstance(app_source, (memoryview, bytes, bytearray)):
            app_source = bytes(app_source).decode("utf-8", errors="replace")
        if isinstance(stmt_string, (memoryview, bytes, bytearray)):
            stmt_string = bytes(stmt_string).decode("utf-8", errors="replace")
        parsed = parse_appsource_pack(str(app_source) if app_source else "")

        invocation_id = parsed.get("inv")
        response_size = parsed.get("response_size")
        if invocation_id and response_size is not None:
            # First-write-wins: don't clobber if the same invocation already
            # has a beacon (defensive — beacon SQL text embeds inv so the
            # plan is unique per invocation).
            resp_by_invocation.setdefault(invocation_id, response_size)

        is_beacon = bool(stmt_string) and MCP_BEACON_SQL_MARKER in str(stmt_string)
        if is_beacon:
            beacon_indices.add(idx)

        records.append({
            "LAST_EXECUTION_TIMESTAMP": last_ts,
            "MCP_SESSION_ID": parsed.get("sess"),
            "INVOCATION_ID": invocation_id,
            "HANA_CORRELATION_ID": parsed.get("corr"),
            "TOOL_NAME": parsed.get("tool"),
            "RESPONSE_SIZE": None,   # filled in second pass
            "AGENT_NAME": parsed.get("agent"),
            "MODEL_NAME": parsed.get("model"),
            "CLIENT_DECLARED_NAME": parsed.get("cli"),
            "CLIENT_IP": parsed.get("mcp_ip"),
            "MCP_VERSION": parsed.get("mcp"),
            "HANA_AUTHENTICATED_USER": user_name,
            "HANA_SESSION_USER": session_user,
            "STATEMENT_HASH": stmt_hash,
            "EXECUTION_COUNT": exec_count,
            "APPLICATION_NAME": app_name,
            "APPLICATION_SOURCE": app_source,
        })

    # Second pass: fan out beacon-provided response_size to every row of the
    # same INVOCATION_ID, then drop beacon rows.
    visible: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        if idx in beacon_indices:
            continue
        inv = rec.get("INVOCATION_ID")
        if inv and inv in resp_by_invocation:
            rec["RESPONSE_SIZE"] = resp_by_invocation[inv]
        visible.append(rec)

    if not visible:
        return pd.DataFrame(columns=HANA_MCP_AUDIT_VIEW_COLUMNS)

    df = pd.DataFrame(visible, columns=HANA_MCP_AUDIT_VIEW_COLUMNS)
    df["LAST_EXECUTION_TIMESTAMP"] = pd.to_datetime(
        df["LAST_EXECUTION_TIMESTAMP"], errors="coerce"
    )
    return df


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

    Opens an HTTP MCP session against ``base_url`` via
    :class:`SyncHTTPMCPClient`, calls ``initialize`` to hand the server the
    end-user / agent / model identity that will show up in every HANA audit
    row (see :doc:`mcp_audit`), enumerates the server's tool catalogue via
    ``tools/list``, and wraps each one as a
    :class:`langchain_core.tools.StructuredTool` whose ``args_schema`` is
    derived from the tool's JSON Schema. The returned tools are drop-in
    replacements for local ``BaseTool`` instances — hand them to
    :class:`~hana_ai.iagents.context_agent.ContextAgent` (or any LangChain
    agent) and every invocation transparently round-trips through the MCP
    server, so the HANA-side audit path (setclientinfo, plan cache, audit
    log) is exercised end-to-end.

    Parameters
    ----------
    base_url : str
        MCP server URL. Trailing ``/mcp`` is added automatically if absent
        (e.g. ``http://127.0.0.1:8001`` and ``http://127.0.0.1:8001/mcp``
        are both accepted). Only ``http``/``https`` transports are
        supported by this helper — for stdio, use the stdio client
        directly.
    timeout : int, default 120
        Per-request HTTP timeout in seconds. Applied to every ``tools/list``
        and ``tools/call`` request.
    client_name : str
        End-user identity or client name passed to the server via
        ``clientInfo.name`` and the ``x-mcp-client-name`` header. Lands on
        HANA's ``APPLICATIONUSER`` channel and inside the pack's ``cli=``
        segment.
    client_version : str, default "0.1"
        Client version tag; lands on HANA's ``APPLICATIONVERSION``.
    client_metadata : dict, optional
        Optional identity metadata forwarded to the server. Recognised
        keys:

        * ``client_id`` — sets the ``x-mcp-client-id`` header
        * ``agent_name`` — sets ``x-ai-agent-name``; lands in pack ``agent=``
        * ``model_name`` — sets ``x-ai-model-name``; lands in pack ``model=``
        * ``model_version`` — sets ``x-ai-model-version``

        Values are stringified. Unknown keys are ignored.
    skip_admin_tools : bool, default True
        When ``True``, tools whose name starts with ``admin_`` (e.g.
        ``admin_update_connection_context``) are excluded from the returned
        list. Set to ``False`` if the agent needs to manage the server's
        connection context at runtime.

    Returns
    -------
    tuple[list[StructuredTool], SyncHTTPMCPClient]
        ``(tools, client)`` — the LangChain-ready tools, and the live MCP
        client. Keep the client alive for the duration of the agent
        session; when done, call ``client.close()`` to release the HTTP
        session and its MCP session id.

    Examples
    --------
    Launch the MCP server on one process, then wire it into a
    ``ContextAgent`` from another:

    >>> from hana_ai.tools.hana_ml_tools.utility import build_context_agent_mcp_tools
    >>> tools, client = build_context_agent_mcp_tools(
    ...     "http://127.0.0.1:8001/mcp",
    ...     client_name="context-agent-notebook",
    ...     client_metadata={
    ...         "agent_name": "context-agent",
    ...         "model_name": "gpt-4.1",
    ...     },
    ... )
    >>> [t.name for t in tools][:3]
    ['fetch_data', 'list_models', 'display_config_dict']
    >>> # Hand `tools` to any LangChain agent; every call now carries the
    >>> # context-agent identity all the way into HANA's audit trail.

    See Also
    --------
    SyncHTTPMCPClient : the underlying JSON-RPC client, exposed for callers
        that want direct access (session id inspection, manual ``call_tool``
        invocations, etc.).
    fetch_hana_mcp_audit_view : query the HANA-side audit view populated by
        every tool call made through the returned tools.
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
