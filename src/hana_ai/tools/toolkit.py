"""
Toolkit for interacting with hana-ml.

The following class is available:

    * :class `HANAMLToolkit`
"""
# pylint: disable=ungrouped-imports
import os
import sys
import socket
import json
import hashlib
import ipaddress
import re
from contextlib import closing
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Annotated, Any, ClassVar
import inspect
from uuid import uuid4
try:
    from pydantic import Field as PydField
except Exception:
    PydField = None
try:
    from typing_extensions import Doc as TxtDoc  # PEP 727 style doc metadata
except Exception:
    TxtDoc = None
try:
    from mcp.server.fastmcp import FastMCP, Context as LegacyMCPContext
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mcp"])
    from mcp.server.fastmcp import FastMCP, Context as LegacyMCPContext

# For HTTP transport support via fastmcp (separate package)
try:
    from fastmcp import FastMCP as FastMCPHTTP, Context as FastMCPContext
    from fastmcp.server.dependencies import get_http_request, get_context as get_fastmcp_context
    from fastmcp.server.middleware.middleware import Middleware
except ImportError:
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastmcp"])
        from fastmcp import FastMCP as FastMCPHTTP, Context as FastMCPContext
        from fastmcp.server.dependencies import get_http_request, get_context as get_fastmcp_context
        from fastmcp.server.middleware.middleware import Middleware
    except Exception:
        FastMCPHTTP = None
        FastMCPContext = None
        get_http_request = None
        get_fastmcp_context = None
        Middleware = None

from hana_ml import ConnectionContext
from hana_ai.langchain_compat import BaseToolkit, BaseTool

from hana_ai.tools.code_template_tools import GetCodeTemplateFromVectorDB
from hana_ai.tools.hana_ml_tools.fetch_tools import FetchDataTool
from hana_ai.tools.hana_ml_tools.model_storage_tools import DeleteModels, ListModels
from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine
from hana_ai.tools.hana_ml_tools.additive_model_forecast_tools import AdditiveModelForecastFitAndSave, AdditiveModelForecastLoadModelAndPredict, MassiveAdditiveModelForecastFitAndSave, MassiveAdditiveModelForecastLoadModelAndPredict
from hana_ai.tools.hana_ml_tools.cap_artifacts_tools import CAPArtifactsForBASTool, CAPArtifactsTool
from hana_ai.tools.hana_ml_tools.intermittent_forecast_tools import IntermittentForecast
from hana_ai.tools.hana_ml_tools.ts_visualizer_tools import ForecastLinePlot, TimeSeriesDatasetReport
from hana_ai.tools.hana_ml_tools.automatic_timeseries_tools import AutomaticTimeSeriesFitAndSave, AutomaticTimeSeriesLoadModelAndPredict, AutomaticTimeSeriesLoadModelAndScore
from hana_ai.tools.hana_ml_tools.ts_check_tools import TimeSeriesCheck, MassiveTimeSeriesCheck
from hana_ai.tools.hana_ml_tools.ts_outlier_detection_tools import TSOutlierDetection
from hana_ai.tools.hana_ml_tools.ts_accuracy_measure_tools import AccuracyMeasure
from hana_ai.tools.hana_ml_tools.hdi_artifacts_tools import HDIArtifactsTool
from hana_ai.tools.hana_ml_tools import dataset_prep_tools as dataset_prep_tools_module
from hana_ai.tools.hana_ml_tools.unsupported_tools import ClassificationTool, RegressionTool
from hana_ai.tools.hana_ml_tools.ts_make_predict_table import TSMakeFutureTableTool, TSMakeFutureTableForMassiveForecastTool
from hana_ai.tools.hana_ml_tools.select_statement_to_table_tools import SelectStatementToTableTool
from hana_ai.tools.hana_ml_tools.massive_automatic_timeseries_tools import MassiveAutomaticTimeSeriesFitAndSave, MassiveAutomaticTimeSeriesLoadModelAndPredict, MassiveAutomaticTimeSeriesLoadModelAndScore
from hana_ai.tools.hana_ml_tools.massive_ts_outlier_detection_tools import MassiveTSOutlierDetection
from hana_ai.tools.hana_ml_tools.python_exec_tools import PythonHanaMLExecTool

ImportCSVToTableTool = dataset_prep_tools_module.ImportCSVToTableTool
SplitTableForForecastingTool = getattr(
    dataset_prep_tools_module,
    "SplitTableForForecastingTool",
    dataset_prep_tools_module.SplitTableForModelingTool,
)


def _is_sensitive_key(key: str) -> bool:
    k = (key or "").lower()
    return any(x in k for x in ("password", "passwd", "secret", "token", "key"))


def _redact_dict(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        out[k] = "***" if _is_sensitive_key(str(k)) and v is not None else v
    return out


def _env_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _build_cc_params_from_env() -> dict[str, Any]:
    address = os.environ.get("HANA_ADDRESS")
    port_raw = os.environ.get("HANA_PORT", "443")
    user = os.environ.get("HANA_USER")
    password = os.environ.get("HANA_PASSWORD")
    encrypt = _env_bool("HANA_ENCRYPT", default=None)
    # Default to False to match existing test infra (`RaysKey` uses sslValidateCertificate=False)
    # and to avoid failing in environments without a complete trust store.
    ssl_validate = _env_bool("HANA_SSL_VALIDATE", default=False)

    missing = [k for k, v in {"HANA_ADDRESS": address, "HANA_USER": user, "HANA_PASSWORD": password}.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    try:
        port = int(port_raw)
    except Exception as e:
        raise ValueError(f"Invalid HANA_PORT: {port_raw}") from e

    params: dict[str, Any] = {
        "address": address,
        "port": port,
        "user": user,
        "password": password,
    }
    if encrypt is not None:
        params["encrypt"] = bool(encrypt)
    if ssl_validate is not None:
        params["sslValidateCertificate"] = bool(ssl_validate)
    return params


def _refresh_tools_for_new_context(toolkit: "HANAMLToolkit") -> dict[str, Any]:
    """Propagate the current toolkit.connection_context into tools and rebuild defaults."""
    updated_tools = 0
    recreated_default_tools = 0

    def _update_tool_ctx(t: Any) -> bool:
        try:
            if hasattr(t, "connection_context"):
                setattr(t, "connection_context", toolkit.connection_context)
                return True
        except Exception:
            return False
        return False

    if toolkit.used_tools:
        for t in list(toolkit.used_tools):
            if _update_tool_ctx(t):
                updated_tools += 1

    if toolkit.default_tools:
        for t in list(toolkit.default_tools):
            if _update_tool_ctx(t):
                updated_tools += 1

    try:
        selected_names = None
        if toolkit.used_tools is not None:
            selected_names = [getattr(t, "name", None) for t in toolkit.used_tools]
            selected_names = [n for n in selected_names if n]

        toolkit.default_tools = [
            AccuracyMeasure(connection_context=toolkit.connection_context),
            AdditiveModelForecastFitAndSave(connection_context=toolkit.connection_context),
            AdditiveModelForecastLoadModelAndPredict(connection_context=toolkit.connection_context),
            AutomaticTimeSeriesFitAndSave(connection_context=toolkit.connection_context),
            AutomaticTimeSeriesLoadModelAndPredict(connection_context=toolkit.connection_context),
            AutomaticTimeSeriesLoadModelAndScore(connection_context=toolkit.connection_context),
            CAPArtifactsTool(connection_context=toolkit.connection_context),
            DeleteModels(connection_context=toolkit.connection_context),
            FetchDataTool(connection_context=toolkit.connection_context),
            ImportCSVToTableTool(connection_context=toolkit.connection_context),
            ForecastLinePlot(connection_context=toolkit.connection_context),
            IntermittentForecast(connection_context=toolkit.connection_context),
            ListModels(connection_context=toolkit.connection_context),
            HDIArtifactsTool(connection_context=toolkit.connection_context),
            SplitTableForForecastingTool(connection_context=toolkit.connection_context),
            TimeSeriesDatasetReport(connection_context=toolkit.connection_context),
            TimeSeriesCheck(connection_context=toolkit.connection_context),
            TSOutlierDetection(connection_context=toolkit.connection_context),
            ClassificationTool(connection_context=toolkit.connection_context),
            RegressionTool(connection_context=toolkit.connection_context),
            TSMakeFutureTableTool(connection_context=toolkit.connection_context),
            SelectStatementToTableTool(connection_context=toolkit.connection_context),
            MassiveAutomaticTimeSeriesFitAndSave(connection_context=toolkit.connection_context),
            MassiveAutomaticTimeSeriesLoadModelAndPredict(connection_context=toolkit.connection_context),
            MassiveAutomaticTimeSeriesLoadModelAndScore(connection_context=toolkit.connection_context),
            MassiveAdditiveModelForecastFitAndSave(connection_context=toolkit.connection_context),
            MassiveAdditiveModelForecastLoadModelAndPredict(connection_context=toolkit.connection_context),
            MassiveTimeSeriesCheck(connection_context=toolkit.connection_context),
            TSMakeFutureTableForMassiveForecastTool(connection_context=toolkit.connection_context),
            MassiveTSOutlierDetection(connection_context=toolkit.connection_context),
            PythonHanaMLExecTool(connection_context=toolkit.connection_context),
        ]
        recreated_default_tools = len(toolkit.default_tools)

        if selected_names:
            toolkit.used_tools = [t for t in toolkit.default_tools if getattr(t, "name", None) in selected_names]
        else:
            toolkit.used_tools = toolkit.default_tools
    except Exception as e:
        logging.warning("Failed to rebuild tool instances: %s", e)

    return {
        "tools_updated_in_place": updated_tools,
        "default_tools_rebuilt": recreated_default_tools,
    }


class HANASessionContextError(RuntimeError):
    """Raised when required HANA session context cannot be written."""

class HANAMLToolkit(BaseToolkit):
    """
    Toolkit for interacting with HANA SQL.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.
    used_tools : list, optional
        List of tools to use. If None or 'all', all tools are used. Default to None.

    Examples
    --------
    Assume cc is a connection to a SAP HANA instance:

    >>> from hana_ai.tools.toolkit import HANAMLToolkit
    >>> from hana_ai.agents.hanaml_agent_with_memory import HANAMLAgentWithMemory

    >>> tools = HANAMLToolkit(connection_context=cc, used_tools='all').get_tools()
    >>> chatbot = HANAMLAgentWithMemory(llm=llm, toos=tools, session_id='hana_ai_test', n_messages=10)
    """
    vectordb: Optional[HANAMLinVectorEngine] = None
    connection_context: ConnectionContext = None
    used_tools: Optional[list] = None
    default_tools: List[BaseTool] = None
    audit_enabled: bool = True
    audit_log_path: Optional[str] = None
    audit_service_name: str = "hana-ai-mcp-service"
    audit_environment: str = "local"
    trust_proxy_headers: bool = True
    trusted_proxy_networks: tuple[Any, ...] = ()
    mcp_session_metadata: dict[str, dict[str, Any]] = None
    # Registry of running MCP servers keyed by (host, port, transport)
    # Use a class-level global registry so multiple toolkit instances share state.
    _global_mcp_servers: ClassVar[dict] = {}
    mcp_servers: dict = None

    def __init__(
        self,
        connection_context,
        used_tools=None,
        return_direct=None,
        audit_enabled: bool = True,
        audit_log_path: Optional[str] = None,
        audit_service_name: str = "hana-ai-mcp-service",
        audit_environment: str = "local",
        trust_proxy_headers: Optional[bool] = None,
        trusted_proxy_cidrs: Optional[list[str] | str] = None,
    ):
        super().__init__(connection_context=connection_context)
        # Initialize server registry (shared across instances)
        self.mcp_servers = HANAMLToolkit._global_mcp_servers
        self.audit_enabled = bool(audit_enabled)
        if trust_proxy_headers is None:
            self.trust_proxy_headers = bool(_env_bool("MCP_TRUST_PROXY_HEADERS", default=True))
        else:
            self.trust_proxy_headers = bool(trust_proxy_headers)
        self.trusted_proxy_networks = self._parse_trusted_proxy_networks(trusted_proxy_cidrs)
        configured_audit_log_path = audit_log_path
        if configured_audit_log_path is None:
            configured_audit_log_path = os.environ.get("MCP_AUDIT_LOG_PATH")
        if configured_audit_log_path:
            self.audit_log_path = str(Path(configured_audit_log_path).expanduser())
        else:
            self.audit_log_path = None
        self.audit_service_name = audit_service_name
        self.audit_environment = audit_environment
        self.mcp_session_metadata: dict[str, dict[str, Any]] = {}
        self._audit_lock = threading.RLock()
        self.default_tools = [
            AccuracyMeasure(connection_context=self.connection_context),
            AdditiveModelForecastFitAndSave(connection_context=self.connection_context),
            AdditiveModelForecastLoadModelAndPredict(connection_context=self.connection_context),
            AutomaticTimeSeriesFitAndSave(connection_context=self.connection_context),
            AutomaticTimeSeriesLoadModelAndPredict(connection_context=self.connection_context),
            AutomaticTimeSeriesLoadModelAndScore(connection_context=self.connection_context),
            CAPArtifactsTool(connection_context=self.connection_context),
            DeleteModels(connection_context=self.connection_context),
            FetchDataTool(connection_context=self.connection_context),
            ImportCSVToTableTool(connection_context=self.connection_context),
            ForecastLinePlot(connection_context=self.connection_context),
            IntermittentForecast(connection_context=self.connection_context),
            ListModels(connection_context=self.connection_context),
            HDIArtifactsTool(connection_context=self.connection_context),
            SplitTableForForecastingTool(connection_context=self.connection_context),
            TimeSeriesDatasetReport(connection_context=self.connection_context),
            TimeSeriesCheck(connection_context=self.connection_context),
            TSOutlierDetection(connection_context=self.connection_context),
            ClassificationTool(connection_context=self.connection_context),
            RegressionTool(connection_context=self.connection_context),
            TSMakeFutureTableTool(connection_context=self.connection_context),
            SelectStatementToTableTool(connection_context=self.connection_context),
            MassiveAutomaticTimeSeriesFitAndSave(connection_context=self.connection_context),
            MassiveAutomaticTimeSeriesLoadModelAndPredict(connection_context=self.connection_context),
            MassiveAutomaticTimeSeriesLoadModelAndScore(connection_context=self.connection_context),
            MassiveAdditiveModelForecastFitAndSave(connection_context=self.connection_context),
            MassiveAdditiveModelForecastLoadModelAndPredict(connection_context=self.connection_context),
            MassiveTimeSeriesCheck(connection_context=self.connection_context),
            TSMakeFutureTableForMassiveForecastTool(connection_context=self.connection_context),
            MassiveTSOutlierDetection(connection_context=self.connection_context),
            PythonHanaMLExecTool(connection_context=self.connection_context),
        ]
        if isinstance(return_direct, dict):
            for tool in self.default_tools:
                if tool.name in return_direct:
                    tool.return_direct = return_direct[tool.name]
        if isinstance(return_direct, bool):
            for tool in self.default_tools:
                tool.return_direct = return_direct
        if used_tools is None or used_tools == "all":
            self.used_tools = self.default_tools
        else:
            if isinstance(used_tools, str):
                used_tools = [used_tools]
            self.used_tools = []
            for tool in self.default_tools:
                if tool.name in used_tools:
                    self.used_tools.append(tool)

    def add_custom_tool(self, tool: BaseTool):
        """
        Add a custom tool to the toolkit.

        Parameters
        ----------
        tool : BaseTool
            Custom tool to add.

            .. note::

                The tool must be a subclass of BaseTool. Please follow the guide to create the custom tools https://python.langchain.com/docs/how_to/custom_tools/.
        """
        self.used_tools.append(tool)

    def delete_tool(self, tool_name: str):
        """
        Delete a tool from the toolkit.

        Parameters
        ----------
        tool_name : str
            Name of the tool to delete.
        """
        for tool in self.used_tools:
            if tool.name == tool_name:
                self.used_tools.remove(tool)
                break

    def reset_tools(self, tools: Optional[List[BaseTool]] = None):
        """
        Reset the toolkit's tools.

        Parameters
        ----------
        tools : list of BaseTool or list of str, optional
            If provided, the toolkit will only contain these tools. When a list of
            strings is provided, tools will be matched by name from the default tools.
            If None, reset to default tools.
        """
        if tools is None:
            # Reset to the default tools list
            self.used_tools = self.default_tools
            return

        new_tools: List[BaseTool] = []
        for t in tools:
            if isinstance(t, BaseTool):
                new_tools.append(t)
            elif isinstance(t, str):
                # Match by name from default tools
                for dt in self.default_tools:
                    if getattr(dt, "name", None) == t:
                        new_tools.append(dt)
                        break
            # Ignore invalid entries silently

        self.used_tools = new_tools

    def set_bas(self, bas=True):
        """
        Set the BAS mode for all tools in the toolkit.
        """
        for tool in self.used_tools:
            if hasattr(tool, "bas"):
                tool.bas = bas
        # remove the GetCodeTemplateFromVectorDB tool if it is in the used_tools
        for tool in self.used_tools:
            if isinstance(tool, CAPArtifactsTool):
                self.used_tools.remove(tool)
                break
        self.used_tools.append(CAPArtifactsForBASTool(connection_context=self.connection_context))
        return self

    def set_vectordb(self, vectordb):
        """
        Set the vector database.

        Parameters
        ----------
        vectordb : HANAMLinVectorEngine
            Vector database.
        """
        self.vectordb = vectordb

    def is_port_available(self, port: int) -> bool:
        """检查端口是否可用"""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(('127.0.0.1', port))
                return True
            except OSError:
                return False

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _safe_identifier(self, value: str) -> str:
        return (value or "").replace('"', '""')

    def _safe_sql_literal(self, value: Any) -> str:
        return str(value).replace("'", "''")

    def _json_default(self, value: Any):
        return str(value)

    def _hash_payload(self, payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=self._json_default)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _redact_audit_payload(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            return {
                key: (
                    "***"
                    if _is_sensitive_key(str(key)) and value is not None
                    else self._redact_audit_payload(value)
                )
                for key, value in payload.items()
            }
        if isinstance(payload, (list, tuple)):
            return [self._redact_audit_payload(item) for item in payload]
        if isinstance(payload, str) and len(payload) > 5000:
            return payload[:5000]
        return payload

    def _collect_stdio_process_metadata(self, transport: str, server_host: str, server_port: Optional[int]) -> dict[str, Any]:
        process_name = os.path.basename(sys.argv[0]) if sys.argv else None
        return {
            "transport": transport,
            "client_ip": None,
            "client_port": None,
            "server_host": server_host,
            "server_port": server_port,
            "user_agent": None,
            "process_id": os.getpid(),
            "parent_process_id": os.getppid(),
            "process_name": process_name,
        }

    def _parse_trusted_proxy_networks(self, configured_cidrs: Optional[list[str] | str]) -> tuple[Any, ...]:
        values = ["127.0.0.0/8", "::1/128"]
        if configured_cidrs is None:
            configured_cidrs = os.environ.get("MCP_TRUSTED_PROXY_CIDRS", "")

        if isinstance(configured_cidrs, str):
            values.extend(
                item.strip()
                for item in configured_cidrs.replace(";", ",").split(",")
                if item.strip()
            )
        elif configured_cidrs:
            values.extend(str(item).strip() for item in configured_cidrs if str(item).strip())

        networks = []
        seen = set()
        for value in values:
            try:
                if "/" in value:
                    network = ipaddress.ip_network(value, strict=False)
                else:
                    address = ipaddress.ip_address(value)
                    prefix = 128 if address.version == 6 else 32
                    network = ipaddress.ip_network(f"{address}/{prefix}", strict=False)
            except ValueError:
                logging.warning("Ignoring invalid trusted proxy CIDR: %s", value)
                continue

            key = str(network)
            if key not in seen:
                seen.add(key)
                networks.append(network)

        return tuple(networks)

    def _normalize_ip_address(self, value: Optional[str]) -> Optional[str]:
        token = (value or "").strip().strip('"').strip("'")
        if not token:
            return None
        if token.lower() == "unknown" or token.startswith("_"):
            return None

        if token.startswith("["):
            closing_bracket = token.find("]")
            if closing_bracket != -1:
                token = token[1:closing_bracket]
            else:
                token = token[1:]

        try:
            return str(ipaddress.ip_address(token))
        except ValueError:
            pass

        if token.count(":") == 1:
            host, port = token.rsplit(":", 1)
            if port.isdigit():
                try:
                    return str(ipaddress.ip_address(host))
                except ValueError:
                    pass

        return None

    def _parse_forwarded_header(self, header_value: Optional[str]) -> list[str]:
        addresses: list[str] = []
        for entry in str(header_value or "").split(","):
            for directive in entry.split(";"):
                if "=" not in directive:
                    continue
                key, raw_value = directive.split("=", 1)
                if key.strip().lower() != "for":
                    continue
                normalized = self._normalize_ip_address(raw_value)
                if normalized:
                    addresses.append(normalized)
        return addresses

    def _parse_x_forwarded_for_header(self, header_value: Optional[str]) -> list[str]:
        addresses: list[str] = []
        for token in str(header_value or "").split(","):
            normalized = self._normalize_ip_address(token)
            if normalized:
                addresses.append(normalized)
        return addresses

    def _is_trusted_proxy_ip(self, ip_value: Optional[str]) -> bool:
        normalized = self._normalize_ip_address(ip_value)
        if not normalized:
            return False
        address = ipaddress.ip_address(normalized)
        return any(address in network for network in self.trusted_proxy_networks)

    def _resolve_client_ip(self, headers: dict[str, str], peer_ip: Optional[str]) -> Optional[str]:
        normalized_peer_ip = self._normalize_ip_address(peer_ip)
        if not normalized_peer_ip:
            return peer_ip
        if not self.trust_proxy_headers or not self._is_trusted_proxy_ip(normalized_peer_ip):
            return normalized_peer_ip

        forwarded_chain = self._parse_forwarded_header(headers.get("forwarded"))
        if not forwarded_chain:
            forwarded_chain = self._parse_x_forwarded_for_header(headers.get("x-forwarded-for"))
        if not forwarded_chain:
            real_ip = self._normalize_ip_address(headers.get("x-real-ip"))
            return real_ip or normalized_peer_ip

        full_chain = [*forwarded_chain, normalized_peer_ip]
        for candidate in reversed(full_chain):
            if self._is_trusted_proxy_ip(candidate):
                continue
            return candidate

        return forwarded_chain[0]

    def _collect_transport_metadata(self, transport: str, server_host: str, server_port: Optional[int]) -> dict[str, Any]:
        if transport not in {"http", "sse"} or get_http_request is None:
            return self._collect_stdio_process_metadata(transport, server_host, server_port)

        metadata = {
            "transport": transport,
            "client_ip": None,
            "client_port": None,
            "server_host": server_host,
            "server_port": server_port,
            "user_agent": None,
            "process_id": os.getpid(),
            "parent_process_id": os.getppid(),
            "process_name": os.path.basename(sys.argv[0]) if sys.argv else None,
        }

        try:
            request = get_http_request()
            headers = {key.lower(): value for key, value in request.headers.items()}
            client = getattr(request, "client", None)
            if client is not None:
                peer_ip = getattr(client, "host", None)
                metadata["client_ip"] = self._resolve_client_ip(headers, peer_ip)
                metadata["client_port"] = getattr(client, "port", None)
            metadata["user_agent"] = headers.get("user-agent")
            metadata["server_host"] = getattr(request.url, "hostname", None) or server_host
            metadata["server_port"] = getattr(request.url, "port", None) or server_port
        except Exception:
            pass

        return metadata

    def _read_value(self, source: Any, key: str, default: Any = None) -> Any:
        if source is None:
            return default
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    def _resolve_context_session_id(self, ctx: Any, transport: str) -> Optional[str]:
        if ctx is None:
            return f"stdio-{os.getpid()}" if transport == "stdio" else None

        request_context = getattr(ctx, "request_context", None)
        request = getattr(request_context, "request", None)
        if request is not None:
            try:
                header_value = request.headers.get("mcp-session-id")
                if header_value:
                    return header_value
            except Exception:
                pass

        try:
            session = getattr(ctx, "session", None)
        except Exception:
            session = None

        for attr_name in ("_fastmcp_id", "id", "session_id"):
            value = getattr(session, attr_name, None) if session is not None else None
            if value:
                return str(value)

        session_id_attr = inspect.getattr_static(ctx, "session_id", None)
        if session_id_attr is not None:
            try:
                session_id = getattr(ctx, "session_id")
                if session_id:
                    return str(session_id)
            except Exception:
                pass

        return f"stdio-{os.getpid()}" if transport == "stdio" else None

    def _extract_initialize_metadata(
        self,
        message: Any,
        transport: str,
        server_host: str,
        server_port: Optional[int],
    ) -> dict[str, Any]:
        metadata = self._collect_transport_metadata(transport, server_host, server_port)
        metadata["trust_level"] = "advisory"

        params = self._read_value(message, "params", {})
        client_info = self._read_value(params, "clientInfo", {})
        beta_meta = self._read_value(params, "metadata", {}) or self._read_value(params, "meta", {}) or {}

        headers = {}
        if transport in {"http", "sse"} and get_http_request is not None:
            try:
                request = get_http_request()
                headers = {key.lower(): value for key, value in request.headers.items()}
            except Exception:
                headers = {}

        metadata.update({
            "client_declared_name": self._read_value(client_info, "name") or headers.get("x-mcp-client-name"),
            "client_declared_version": self._read_value(client_info, "version") or headers.get("x-mcp-client-version"),
            "client_declared_id": (
                self._read_value(beta_meta, "client_id")
                or self._read_value(beta_meta, "clientId")
                or headers.get("x-mcp-client-id")
            ),
            "client_declared_agent_name": (
                self._read_value(beta_meta, "agent_name")
                or self._read_value(beta_meta, "agentName")
                or headers.get("x-ai-agent-name")
            ),
            "client_declared_model_name": (
                self._read_value(beta_meta, "model_name")
                or self._read_value(beta_meta, "modelName")
                or headers.get("x-ai-model-name")
            ),
            "client_declared_model_version": (
                self._read_value(beta_meta, "model_version")
                or self._read_value(beta_meta, "modelVersion")
                or headers.get("x-ai-model-version")
            ),
        })

        return metadata

    def _extract_initialize_metadata_from_session(
        self,
        ctx: Any,
        transport: str,
        server_host: str,
        server_port: Optional[int],
    ) -> dict[str, Any]:
        """Build the same shape as ``_extract_initialize_metadata`` but read identity
        from the legacy ``ServerSession.client_params`` (a parsed Pydantic model)
        instead of a raw ``InitializeRequest`` message. Used by the stdio/SSE
        bootstrap path where no middleware-level message is available.
        """
        metadata = self._collect_transport_metadata(transport, server_host, server_port)
        metadata["trust_level"] = "advisory"

        client_params = None
        try:
            session = getattr(ctx, "session", None) if ctx is not None else None
            client_params = getattr(session, "client_params", None) if session is not None else None
        except Exception:
            client_params = None

        client_info = self._read_value(client_params, "clientInfo", None) if client_params is not None else None
        # Some MCP server implementations expose the parsed params with a top-level
        # clientInfo attribute; others only carry name/version directly. Fall back
        # to reading them off client_params itself.
        if client_info is None and client_params is not None:
            client_info = client_params

        beta_meta = (
            self._read_value(client_params, "metadata", {}) if client_params is not None else {}
        ) or (
            self._read_value(client_params, "meta", {}) if client_params is not None else {}
        ) or {}

        headers = {}
        if transport in {"http", "sse"} and get_http_request is not None:
            try:
                request = get_http_request()
                headers = {key.lower(): value for key, value in request.headers.items()}
            except Exception:
                headers = {}

        metadata.update({
            "client_declared_name": self._read_value(client_info, "name") or headers.get("x-mcp-client-name"),
            "client_declared_version": self._read_value(client_info, "version") or headers.get("x-mcp-client-version"),
            "client_declared_id": (
                self._read_value(beta_meta, "client_id")
                or self._read_value(beta_meta, "clientId")
                or headers.get("x-mcp-client-id")
            ),
            "client_declared_agent_name": (
                self._read_value(beta_meta, "agent_name")
                or self._read_value(beta_meta, "agentName")
                or headers.get("x-ai-agent-name")
            ),
            "client_declared_model_name": (
                self._read_value(beta_meta, "model_name")
                or self._read_value(beta_meta, "modelName")
                or headers.get("x-ai-model-name")
            ),
            "client_declared_model_version": (
                self._read_value(beta_meta, "model_version")
                or self._read_value(beta_meta, "modelVersion")
                or headers.get("x-ai-model-version")
            ),
        })

        return metadata

    def _apply_audit_session_started(
        self,
        session_id: Optional[str],
        transport: str,
        server_host: str,
        server_port: Optional[int],
        *,
        ctx: Any = None,
        message: Any = None,
    ) -> dict[str, Any]:
        """Shared session-bootstrap path used by every transport.

        - HTTP path passes ``message=context.message`` from the FastMCP middleware.
        - stdio / SSE pass ``ctx`` from the per-tool execution wrapper, and the
          helper reads ``ServerSession.client_params`` to recover the same
          ``clientInfo`` / extension fields that HTTP gets from the message.

        The helper:

        1. Builds the same metadata dict as ``_extract_initialize_metadata``.
        2. Stores it under ``session_id`` so subsequent tool calls see it.
        3. Emits the ``mcp.session.started`` audit event.
        4. Best-effort projects the started event into HANA Session Variables
           via ``_set_hana_session_variables`` so the bootstrap is observable
           on the connection even before the first tool runs SQL.

        Returns the metadata dict (mainly for callers that want to log it).
        """
        if message is not None:
            metadata = self._extract_initialize_metadata(
                message,
                transport=transport,
                server_host=server_host,
                server_port=server_port,
            )
        else:
            metadata = self._extract_initialize_metadata_from_session(
                ctx,
                transport=transport,
                server_host=server_host,
                server_port=server_port,
            )

        if not session_id:
            return metadata

        metadata["mcp_session_id"] = session_id
        self._store_session_metadata(session_id, metadata)

        started_event = self._build_audit_event(
            "mcp.session.started",
            session_id,
            payload=metadata,
        )
        self._emit_audit_event(started_event)

        # Best-effort: project the started event into HANA Session Variables
        # so the bootstrap is observable on the active connection. Mirrors
        # _update_hana_session_event(strict=False) semantics — never break the
        # caller (tool-call wrapper) on driver failures.
        try:
            self._set_hana_session_variables(
                session_id,
                invocation_id=None,
                hana_correlation_id=None,
                tool_name="",
                event=started_event,
            )
        except Exception as exc:
            logging.warning(
                "Failed to project mcp.session.started into HANA session variables: %s",
                exc,
            )

        return metadata

    def _extract_request_identity_metadata(
        self,
        transport: str,
        server_host: str,
        server_port: Optional[int],
    ) -> dict[str, Any]:
        metadata = self._collect_transport_metadata(transport, server_host, server_port)
        metadata["trust_level"] = "advisory"

        headers = {}
        if transport in {"http", "sse"} and get_http_request is not None:
            try:
                request = get_http_request()
                headers = {key.lower(): value for key, value in request.headers.items()}
            except Exception:
                headers = {}

        metadata.update({
            "client_declared_name": headers.get("x-mcp-client-name"),
            "client_declared_version": headers.get("x-mcp-client-version"),
            "client_declared_id": headers.get("x-mcp-client-id"),
            "client_declared_agent_name": headers.get("x-ai-agent-name"),
            "client_declared_model_name": headers.get("x-ai-model-name"),
            "client_declared_model_version": headers.get("x-ai-model-version"),
        })
        return metadata

    def _store_session_metadata(self, session_id: str, metadata: dict[str, Any]) -> None:
        if not session_id:
            return
        with self._audit_lock:
            existing = self.mcp_session_metadata.get(session_id, {}).copy()
            if not existing:
                existing["created_at"] = self._utc_now_iso()
            existing.update({key: value for key, value in metadata.items() if value is not None})
            existing["mcp_session_id"] = session_id
            existing["last_seen_at"] = self._utc_now_iso()
            self.mcp_session_metadata[session_id] = existing

    def _get_session_metadata(self, session_id: Optional[str]) -> dict[str, Any]:
        if not session_id:
            return {}
        with self._audit_lock:
            return dict(self.mcp_session_metadata.get(session_id, {}))

    def _generate_invocation_id(self) -> str:
        return f"inv-{uuid4().hex}"

    def _generate_hana_correlation_id(self) -> str:
        return f"hana-corr-{uuid4().hex}"

    def _build_audit_event(
        self,
        event_type: str,
        session_id: Optional[str],
        payload: dict[str, Any],
        *,
        invocation_id: Optional[str] = None,
        hana_correlation_id: Optional[str] = None,
    ) -> dict[str, Any]:
        session_metadata = self._get_session_metadata(session_id)
        # Attach the HANA-authenticated identity of the current connection.
        # Best-effort — an empty dict is fine, the session block just leaves
        # those keys unset. Uses the same cached probe as the SESSION_CONTEXT
        # projection so multiple events share a single SELECT.
        hana_identity = self._fetch_hana_identity()
        return {
            "event_id": f"evt-{uuid4().hex}",
            "event_type": event_type,
            "event_version": "1.0",
            "occurred_at": self._utc_now_iso(),
            "service_name": self.audit_service_name,
            "environment": self.audit_environment,
            "principal": {},
            "session": {
                "mcp_session_id": session_id,
                "transport": session_metadata.get("transport"),
                "client_ip": session_metadata.get("client_ip"),
                "client_port": session_metadata.get("client_port"),
                "server_host": session_metadata.get("server_host"),
                "server_port": session_metadata.get("server_port"),
                "user_agent": session_metadata.get("user_agent"),
                "process_id": session_metadata.get("process_id"),
                "parent_process_id": session_metadata.get("parent_process_id"),
                "process_name": session_metadata.get("process_name"),
                "client_declared_name": session_metadata.get("client_declared_name"),
                "client_declared_version": session_metadata.get("client_declared_version"),
                "client_declared_id": session_metadata.get("client_declared_id"),
                "client_declared_agent_name": session_metadata.get("client_declared_agent_name"),
                "client_declared_model_name": session_metadata.get("client_declared_model_name"),
                "client_declared_model_version": session_metadata.get("client_declared_model_version"),
                "trust_level": session_metadata.get("trust_level"),
                # HANA-authenticated identity — sourced from CURRENT_USER /
                # SESSION_USER / CURRENT_CONNECTION plus setclientinfo channels.
                "hana_db_user": hana_identity.get("HANA_DB_USER"),
                "hana_db_session_user": hana_identity.get("HANA_DB_SESSION_USER"),
                "hana_connection_id": hana_identity.get("HANA_CONNECTION_ID"),
                "hana_application_user": hana_identity.get("HANA_APPLICATION_USER"),
                "hana_application": hana_identity.get("HANA_APPLICATION"),
                "hana_client_host": hana_identity.get("HANA_CLIENT_HOST"),
            },
            "correlation": {
                "invocation_id": invocation_id,
                "hana_correlation_id": hana_correlation_id,
            },
            "payload": self._redact_audit_payload(payload),
        }

    def _serialize_audit_event(self, event: dict[str, Any]) -> str:
        return json.dumps(event, sort_keys=True, ensure_ascii=True, default=self._json_default)

    def _emit_server_audit_log(self, event: dict[str, Any]) -> None:
        logging.info("MCP_AUDIT %s", self._serialize_audit_event(event))

    def _write_disk_audit_record(self, event: dict[str, Any]) -> bool:
        if not self.audit_log_path:
            return False
        try:
            log_path = Path(self.audit_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_lock:
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(self._serialize_audit_event(event))
                    log_file.write("\n")
            return True
        except Exception as exc:
            logging.warning("Failed to write disk audit record: %s", exc)
            return False

    def _emit_audit_event(self, event: dict[str, Any]) -> None:
        self._emit_server_audit_log(event)
        if not self.audit_enabled:
            return
        self._write_disk_audit_record(event)

    def _extract_audit_result_payload(self, result: Any) -> Any:
        structured_content = getattr(result, "structured_content", None)
        if structured_content is not None:
            return structured_content

        content = getattr(result, "content", None)
        if content is not None:
            if isinstance(content, list):
                normalized_content = []
                for item in content:
                    if hasattr(item, "text"):
                        normalized_content.append(getattr(item, "text"))
                    elif hasattr(item, "model_dump"):
                        try:
                            normalized_content.append(item.model_dump(exclude_none=True))
                        except Exception:
                            normalized_content.append(item)
                    else:
                        normalized_content.append(item)
                if len(normalized_content) == 1:
                    return normalized_content[0]
                return normalized_content
            return content

        return result

    def _build_hana_attribution_values(
        self,
        session_id: Optional[str],
        invocation_id: str,
        hana_correlation_id: str,
        tool_name: str,
    ) -> dict[str, str]:
        session_metadata = self._get_session_metadata(session_id)
        raw_values = {
            "MCP_SESSION_ID": session_id,
            "MCP_CLIENT_NAME": session_metadata.get("client_declared_name"),
            "MCP_CLIENT_ID": session_metadata.get("client_declared_id"),
            "AI_AGENT_NAME": session_metadata.get("client_declared_agent_name"),
            "AI_MODEL_NAME": session_metadata.get("client_declared_model_name"),
            "INVOCATION_ID": invocation_id,
            "HANA_CORRELATION_ID": hana_correlation_id,
            "TOOL_NAME": tool_name,
        }
        return {
            key: str(value)[:512]
            for key, value in raw_values.items()
            if value is not None
        }

    def _normalize_hana_session_variable_value(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value)[:512]

    def _fetch_hana_identity(self) -> dict[str, str]:
        """Read the authenticated HANA identity of the toolkit's connection.

        Returns a mapping of session-context keys (``HANA_DB_USER``,
        ``HANA_DB_SESSION_USER``, ``HANA_CONNECTION_ID``,
        ``HANA_APPLICATION_USER``, ``HANA_APPLICATION``, ``HANA_CLIENT_HOST``)
        to their string values. Complements ``CLIENT_DECLARED_*`` in the audit
        snapshot — those are what the MCP client *says* it is; these are what
        HANA authenticated.

        Cached on ``self`` keyed by the underlying ``dbapi.Connection`` object
        so the SELECT runs at most once per connection reuse. Returns an empty
        dict on driver failure — the caller MUST NOT let audit projection
        break tool execution.
        """
        if self.connection_context is None:
            return {}
        connection = getattr(self.connection_context, "connection", None)
        if connection is None or not hasattr(connection, "cursor"):
            return {}

        cache = getattr(self, "_hana_identity_cache", None)
        if cache is None:
            cache = {}
            self._hana_identity_cache = cache
        cache_key = id(connection)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        query = (
            "SELECT CURRENT_USER, "
            "SESSION_USER, "
            "CURRENT_CONNECTION, "
            "SESSION_CONTEXT('APPLICATIONUSER'), "
            "SESSION_CONTEXT('APPLICATION'), "
            "SESSION_CONTEXT('CLIENT_HOST') "
            "FROM DUMMY"
        )
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(query)
            row = cursor.fetchone() or ()
        except Exception as exc:  # noqa: BLE001 — audit projection is best-effort
            logging.warning("Failed to read HANA identity for audit snapshot: %s", exc)
            row = ()
        finally:
            if cursor is not None:
                close_method = getattr(cursor, "close", None)
                if callable(close_method):
                    try:
                        close_method()
                    except Exception:
                        pass

        def _cell(idx: int) -> str:
            if idx >= len(row) or row[idx] is None:
                return ""
            return str(row[idx])

        identity = {
            "HANA_DB_USER": _cell(0),
            "HANA_DB_SESSION_USER": _cell(1),
            "HANA_CONNECTION_ID": _cell(2),
            "HANA_APPLICATION_USER": _cell(3),
            "HANA_APPLICATION": _cell(4),
            "HANA_CLIENT_HOST": _cell(5),
        }
        cache[cache_key] = identity
        return identity

    def _build_hana_session_event_values(self, event: dict[str, Any]) -> dict[str, str]:
        session = event.get("session", {}) or {}
        correlation = event.get("correlation", {}) or {}
        payload = event.get("payload", {}) or {}

        # Merge the HANA-authenticated identity into the session block so it
        # rides along with every audit event (started, tool.succeeded, ...).
        # Session block wins if the caller already populated the field.
        identity = self._fetch_hana_identity()

        raw_values = {
            "EVENT_TYPE": event.get("event_type"),
            "OCCURRED_AT": event.get("occurred_at"),
            "MCP_SESSION_ID": session.get("mcp_session_id"),
            "CLIENT_IP": session.get("client_ip"),
            "CLIENT_DECLARED_NAME": session.get("client_declared_name"),
            "CLIENT_DECLARED_AGENT_NAME": session.get("client_declared_agent_name"),
            "CLIENT_DECLARED_MODEL_NAME": session.get("client_declared_model_name"),
            "HANA_DB_USER": session.get("hana_db_user") or identity.get("HANA_DB_USER"),
            "HANA_DB_SESSION_USER": session.get("hana_db_session_user") or identity.get("HANA_DB_SESSION_USER"),
            "HANA_CONNECTION_ID": session.get("hana_connection_id") or identity.get("HANA_CONNECTION_ID"),
            "HANA_APPLICATION_USER": session.get("hana_application_user") or identity.get("HANA_APPLICATION_USER"),
            "HANA_APPLICATION": session.get("hana_application") or identity.get("HANA_APPLICATION"),
            "HANA_CLIENT_HOST": session.get("hana_client_host") or identity.get("HANA_CLIENT_HOST"),
            "TOOL_NAME": payload.get("tool_name"),
            "TARGET_TABLES": payload.get("target_tables"),
            "TOOL_ARGS_JSON": payload.get("tool_args_json"),
            "RESPONSE_SIZE": payload.get("response_size"),
            "MODEL_STORAGE_NAME": payload.get("model_storage_name"),
            "MODEL_STORAGE_VERSION": payload.get("model_storage_version"),
            "STATUS": payload.get("status"),
            "DURATION_MS": payload.get("duration_ms"),
            "HANA_CORRELATION_ID": correlation.get("hana_correlation_id"),
            "INVOCATION_ID": correlation.get("invocation_id"),
            "MCP_CLIENT_NAME": session.get("client_declared_name"),
            "MCP_CLIENT_ID": session.get("client_declared_id"),
            "AI_AGENT_NAME": session.get("client_declared_agent_name"),
            "AI_MODEL_NAME": session.get("client_declared_model_name"),
        }
        return {
            key: self._normalize_hana_session_variable_value(value)
            for key, value in raw_values.items()
        }

    def _write_hana_session_variables(self, values: dict[str, Any]) -> None:
        if self.connection_context is None:
            raise HANASessionContextError(
                "Required HANA session variables could not be written because connection_context is unavailable."
            )

        connection = getattr(self.connection_context, "connection", None)
        if connection is None or not hasattr(connection, "cursor"):
            raise HANASessionContextError(
                "Required HANA session variables could not be written because the active connection does not expose a cursor."
            )

        if not values:
            raise HANASessionContextError(
                "Required HANA session variables could not be written because no attribution values were available."
            )

        cursor = None
        try:
            cursor = connection.cursor()
            for key, value in values.items():
                if any(not (char.isalnum() or char == "_") for char in key):
                    raise HANASessionContextError(f"Invalid HANA session variable name: {key}")
                cursor.execute(
                    f"SET '{key}' = '{self._safe_sql_literal(self._normalize_hana_session_variable_value(value))}'"
                )
        except HANASessionContextError:
            raise
        except Exception as exc:
            raise HANASessionContextError(
                f"Failed to write required HANA session variables: {exc}"
            ) from exc
        finally:
            if cursor is not None:
                close_method = getattr(cursor, "close", None)
                if callable(close_method):
                    try:
                        close_method()
                    except Exception:
                        pass

    def _set_hana_session_variables(
        self,
        session_id: Optional[str],
        invocation_id: str,
        hana_correlation_id: str,
        tool_name: str,
        event: Optional[dict[str, Any]] = None,
    ) -> None:
        if event is not None:
            values = self._build_hana_session_event_values(event)
        else:
            values = self._build_hana_attribution_values(
                session_id,
                invocation_id,
                hana_correlation_id,
                tool_name,
            )
        self._write_hana_session_variables(values)

    def _set_hana_client_info(
        self,
        session_id: Optional[str],
        invocation_id: Optional[str],
        hana_correlation_id: Optional[str],
        tool_name: str,
        event: Optional[dict[str, Any]] = None,
    ) -> bool:
        if self.connection_context is None:
            return False

        connection = getattr(self.connection_context, "connection", None)
        if connection is None or not hasattr(connection, "setclientinfo"):
            return False

        session_metadata = self._get_session_metadata(session_id)
        # The MCP client-declared name is the end-user identity (or the agent
        # identity if no end-user is passed). Surface it on the driver's
        # APPLICATIONUSER channel so HANA's SYS.AUDIT_LOG.APPLICATION_USER_NAME
        # captures it alongside USER_NAME (the DB technical user). Auditors
        # can then correlate DB-authenticated USER_NAME with the end-user the
        # MCP session was acting on behalf of.
        declared_end_user = (
            session_metadata.get("client_declared_id")
            or session_metadata.get("client_declared_name")
        )
        values = {
            "APPLICATION": session_metadata.get("client_declared_name") or self.audit_service_name,
            "APPLICATIONVERSION": session_metadata.get("client_declared_version") or self.audit_environment,
            "APPLICATIONUSER": declared_end_user,
            "APPLICATIONCOMPONENT": tool_name,
            "APPLICATIONCOMPONENTTYPE": "HANA AI Toolkit MCP Server",
            **self._build_hana_attribution_values(
                session_id,
                invocation_id,
                hana_correlation_id,
                tool_name,
            ),
        }
        if event is not None:
            values.update(self._build_hana_session_event_values(event))

        try:
            for key, value in values.items():
                if value is not None:
                    connection.setclientinfo(key, str(value)[:512])
            return True
        except Exception as exc:
            logging.warning("Failed to set HANA client info: %s", exc)
            return False

    def _ensure_hana_execution_context(
        self,
        session_id: Optional[str],
        invocation_id: str,
        hana_correlation_id: str,
        tool_name: str,
        started_event: Optional[dict[str, Any]] = None,
    ) -> None:
        self._set_hana_session_variables(
            session_id,
            invocation_id,
            hana_correlation_id,
            tool_name,
            event=started_event,
        )
        self._set_hana_client_info(
            session_id,
            invocation_id,
            hana_correlation_id,
            tool_name,
            event=started_event,
        )

    def _update_hana_session_event(self, event: dict[str, Any], *, strict: bool = False) -> None:
        try:
            session = event.get("session", {}) or {}
            correlation = event.get("correlation", {}) or {}
            payload = event.get("payload", {}) or {}
            self._write_hana_session_variables(self._build_hana_session_event_values(event))
            self._set_hana_client_info(
                session.get("mcp_session_id"),
                correlation.get("invocation_id"),
                correlation.get("hana_correlation_id"),
                payload.get("tool_name") or "",
                event=event,
            )
        except Exception as exc:
            if strict:
                raise
            logging.warning("Failed to update HANA session event snapshot: %s", exc)

    def _summarize_tool_result(self, result: Any) -> dict[str, Any]:
        audit_result = self._extract_audit_result_payload(result)
        summary = {"result_type": type(audit_result).__name__}
        if isinstance(audit_result, dict):
            summary["keys"] = sorted(str(key) for key in audit_result.keys())[:20]
            if "error" in audit_result:
                summary["contains_error"] = True
        elif isinstance(audit_result, (list, tuple)):
            summary["item_count"] = len(audit_result)
        elif isinstance(audit_result, str):
            summary["text_length"] = len(audit_result)
        return summary

    def _extract_model_storage_metadata(self, result: Any) -> dict[str, Any]:
        result = self._extract_audit_result_payload(result)
        candidate = None

        if isinstance(result, dict):
            candidate = result
        elif isinstance(result, str):
            try:
                parsed = json.loads(result)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                candidate = parsed

        if not isinstance(candidate, dict):
            return {}

        metadata = {}
        if candidate.get("model_storage_name") is not None:
            metadata["model_storage_name"] = candidate.get("model_storage_name")
        if candidate.get("model_storage_version") is not None:
            metadata["model_storage_version"] = candidate.get("model_storage_version")
        return metadata

    def _normalize_response_size(self, value: Any) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
        return None

    def _count_markdown_table_rows(self, value: str) -> Optional[int]:
        table_lines = [
            line.strip()
            for line in str(value or "").splitlines()
            if line.strip().startswith("|") and line.strip().endswith("|")
        ]
        if len(table_lines) < 2:
            return None

        separator = table_lines[1].replace("|", "").replace(":", "").replace("-", "").strip()
        if separator:
            return None
        return max(len(table_lines) - 2, 0)

    def _extract_response_size(self, tool_name: str, arguments: Any, result: Any) -> Optional[int]:
        if result is None:
            return None

        audit_result = self._extract_audit_result_payload(result)
        candidate = audit_result

        if isinstance(candidate, str):
            stripped = candidate.strip()
            if stripped:
                try:
                    candidate = json.loads(stripped)
                except Exception:
                    markdown_row_count = self._count_markdown_table_rows(stripped)
                    if markdown_row_count is not None:
                        return markdown_row_count

        if isinstance(candidate, dict):
            for key in ("response_size", "rows_imported", "row_count", "rows_processed"):
                normalized = self._normalize_response_size(candidate.get(key))
                if normalized is not None:
                    return normalized

        if hasattr(candidate, "shape"):
            try:
                return int(candidate.shape[0])
            except Exception:
                pass

        if isinstance(candidate, (list, tuple, set)):
            return len(candidate)

        if tool_name == "fetch_data" and isinstance(arguments, dict):
            for key in ("top_n", "last_n"):
                normalized = self._normalize_response_size(arguments.get(key))
                if normalized is not None:
                    return normalized

        return None

    def _split_identifier_parts(self, value: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        in_quotes = False
        index = 0

        while index < len(value):
            char = value[index]
            if char == '"':
                current.append(char)
                if in_quotes and index + 1 < len(value) and value[index + 1] == '"':
                    current.append(value[index + 1])
                    index += 1
                else:
                    in_quotes = not in_quotes
            elif char == "." and not in_quotes:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
            else:
                current.append(char)
            index += 1

        part = "".join(current).strip()
        if part:
            parts.append(part)
        return parts

    def _normalize_audit_table_name(self, value: Any) -> Optional[str]:
        text = str(value or "").strip()
        if not text:
            return None

        parts = self._split_identifier_parts(text)
        table_name = parts[-1] if parts else text
        table_name = table_name.strip()
        if table_name.startswith('"') and table_name.endswith('"') and len(table_name) >= 2:
            table_name = table_name[1:-1].replace('""', '"')

        table_name = table_name.strip()
        if not table_name or table_name.startswith("#"):
            return None
        return table_name

    def _split_sql_segments(self, value: str, delimiter: str = ",") -> list[str]:
        segments: list[str] = []
        current: list[str] = []
        in_quotes = False
        depth = 0
        index = 0

        while index < len(value):
            char = value[index]
            if char == '"':
                current.append(char)
                if in_quotes and index + 1 < len(value) and value[index + 1] == '"':
                    current.append(value[index + 1])
                    index += 1
                else:
                    in_quotes = not in_quotes
            elif not in_quotes:
                if char == "(":
                    depth += 1
                elif char == ")" and depth > 0:
                    depth -= 1
                elif char == delimiter and depth == 0:
                    segment = "".join(current).strip()
                    if segment:
                        segments.append(segment)
                    current = []
                    index += 1
                    continue
                current.append(char)
            else:
                current.append(char)
            index += 1

        segment = "".join(current).strip()
        if segment:
            segments.append(segment)
        return segments

    def _extract_table_reference_name(self, value: Any) -> Optional[str]:
        text = str(value or "").strip()
        if not text or text.startswith("("):
            return None

        identifier_pattern = re.compile(
            r'^\s*((?:"(?:[^"]|"")*"|[#A-Za-z_][\w$#]*)(?:\s*\.\s*(?:"(?:[^"]|"")*"|[#A-Za-z_][\w$#]*))*)'
        )
        match = identifier_pattern.match(text)
        if not match:
            return None
        return self._normalize_audit_table_name(match.group(1))

    def _extract_target_tables_from_select_statement(self, select_statement: Any) -> list[str]:
        sql_text = str(select_statement or "").strip()
        if not sql_text:
            return []

        cleaned_sql = re.sub(r"/\*.*?\*/", " ", sql_text, flags=re.S)
        cleaned_sql = re.sub(r"--[^\n]*", " ", cleaned_sql)

        tables: list[str] = []

        def add_table(candidate: Optional[str]) -> None:
            if candidate and candidate not in tables:
                tables.append(candidate)

        clause_terminator = r"(?=\bwhere\b|\bgroup\b|\border\b|\bhaving\b|\blimit\b|\bunion\b|\bexcept\b|\bintersect\b|\bminus\b|\bqualify\b|$)"
        for match in re.finditer(r"\bfrom\b(.*?)" + clause_terminator, cleaned_sql, flags=re.I | re.S):
            for segment in self._split_sql_segments(match.group(1)):
                add_table(self._extract_table_reference_name(segment))
                join_terminator = r"(?=\bon\b|\busing\b|\bjoin\b|\bwhere\b|\bgroup\b|\border\b|\bhaving\b|\blimit\b|\bunion\b|\bexcept\b|\bintersect\b|\bminus\b|\bqualify\b|$)"
                for join_match in re.finditer(r"\bjoin\b\s+(.*?)" + join_terminator, segment, flags=re.I | re.S):
                    add_table(self._extract_table_reference_name(join_match.group(1)))

        return tables

    def _collect_target_tables(self, arguments: dict[str, Any]) -> list[str]:
        target_tables: list[str] = []

        def add_table(candidate: Optional[str]) -> None:
            if candidate and candidate not in target_tables:
                target_tables.append(candidate)

        raw_table_name = arguments.get("table_name")
        if isinstance(raw_table_name, str):
            for segment in self._split_sql_segments(raw_table_name):
                add_table(self._normalize_audit_table_name(segment))
        elif isinstance(raw_table_name, (list, tuple, set)):
            for item in raw_table_name:
                add_table(self._normalize_audit_table_name(item))

        for candidate in self._extract_target_tables_from_select_statement(arguments.get("select_statement")):
            add_table(candidate)

        return target_tables

    def _shrink_tool_args_for_session(self, value: Any, *, depth: int = 0) -> Any:
        if depth >= 4:
            return str(type(value).__name__)

        if isinstance(value, dict):
            shrunk: dict[str, Any] = {}
            items = list(value.items())
            for index, (key, item_value) in enumerate(items[:20]):
                shrunk[str(key)] = self._shrink_tool_args_for_session(item_value, depth=depth + 1)
            if len(items) > 20:
                shrunk["__truncated_keys__"] = len(items) - 20
            return shrunk

        if isinstance(value, (list, tuple)):
            shrunk_items = [
                self._shrink_tool_args_for_session(item, depth=depth + 1)
                for item in list(value)[:20]
            ]
            if len(value) > 20:
                shrunk_items.append({"__truncated_items__": len(value) - 20})
            return shrunk_items

        if isinstance(value, str):
            if len(value) <= 120:
                return value
            return value[:117] + "..."

        return value

    def _build_tool_args_json(self, redacted_args: Any) -> str:
        preview = self._shrink_tool_args_for_session(redacted_args)
        serialized = json.dumps(
            preview,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            default=self._json_default,
        )
        if len(serialized) <= 512:
            return serialized

        fallback: dict[str, Any] = {
            "truncated": True,
            "input_hash": self._hash_payload(redacted_args),
            "type": type(redacted_args).__name__,
        }
        if isinstance(redacted_args, dict):
            fallback["keys"] = sorted(str(key) for key in redacted_args.keys())[:20]

        serialized = json.dumps(
            fallback,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            default=self._json_default,
        )
        if len(serialized) <= 512:
            return serialized

        return json.dumps(
            {
                "truncated": True,
                "input_hash": self._hash_payload(redacted_args),
            },
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )

    def _extract_tool_audit_metadata(
        self,
        tool_name: str,
        arguments: Any,
        *,
        redacted_args: Any = None,
        result: Any = None,
    ) -> dict[str, Any]:
        if redacted_args is None:
            redacted_args = self._redact_audit_payload(arguments or {})

        metadata = {
            "tool_args_json": self._build_tool_args_json(redacted_args),
        }

        if tool_name == "fetch_data" and isinstance(arguments, dict):
            target_tables = self._collect_target_tables(arguments)
            if target_tables:
                metadata["target_tables"] = ",".join(target_tables)

        response_size = self._extract_response_size(tool_name, arguments, result)
        if response_size is not None:
            metadata["response_size"] = response_size

        return metadata

    def launch_mcp_server(
        self,
        server_name: str = "HANATools",
        host: str = "127.0.0.1",
        transport: str = "stdio",
        port: int = 8001,
        auth_token: Optional[str] = None,
        max_retries: int = 5
    ):
        """
        Launch the MCP server with the specified configuration.
        This method initializes the MCP server, registers all tools, and starts the server in a background thread.
        If the specified port is occupied, it will try the next port up to `max_retries` times.

        Parameters
        ----------
        server_name : str
            Name of the server. Default is "HANATools".
        host : str
            Host address for the server.
        transport : {"stdio", "sse", "http"}
            Transport protocol to use. Default is "stdio". Can be "sse" for Server-Sent Events.
        port : int
            Network port to use for server transports that require a port (SSE/HTTP). Default is 8001. Ignored for stdio.
        auth_token : str, optional
            Authentication token for the server. If provided, the server will require this token for access.
        max_retries : int
            Maximum number of retries to find an available port. Default is 5.
        """
        attempts = 0
        original_port = port

        while attempts < max_retries:
            # 初始化MCP配置
            server_settings = {
                "name": server_name,
                "host": host
            }

            # 更新端口设置
            if transport == "sse":
                # 检查端口可用性
                if not self.is_port_available(port):
                    logging.warning("⚠️  Port %s occupied, trying next port", port)
                    port += 1
                    attempts += 1
                    time.sleep(0.2)
                    continue

                server_settings.update({
                    "port": port,
                    "sse_path": '/sse'
                })

            # 创建MCP实例（stdio/sse 使用 mcp.server.fastmcp；http 使用 fastmcp）
            if transport == "http":
                if FastMCPHTTP is None:
                    logging.error("HTTP transport requested but 'fastmcp' package is unavailable.")
                    raise RuntimeError("HTTP transport not supported (fastmcp missing)")
                # HTTP transport relies on the wrapped function signature for schema inference.
                # Register each tool exactly once to avoid FastMCP duplicate-tool warnings.
                mcp = FastMCPHTTP(server_settings.get("name", "HANATools"), host=server_settings.get("host", "127.0.0.1"), port=port, streamable_http_path="/mcp", json_response=True)
                # 检查端口可用性
                if not self.is_port_available(port):
                    logging.warning("⚠️  Port %s occupied, trying next port", port)
                    port += 1
                    attempts += 1
                    time.sleep(0.2)
                    continue
            else:
                mcp = FastMCP(**server_settings)

            if transport == "http" and Middleware is not None and hasattr(mcp, "add_middleware"):
                toolkit = self
                current_host = server_settings.get("host", "127.0.0.1")
                current_port = port

                class MCPAuditMiddleware(Middleware):
                    async def on_initialize(self, context, call_next):
                        result = await call_next(context)
                        session_id = toolkit._resolve_context_session_id(
                            getattr(context, "fastmcp_context", None),
                            transport=transport,
                        )
                        toolkit._apply_audit_session_started(
                            session_id,
                            transport=transport,
                            server_host=current_host,
                            server_port=current_port,
                            message=context.message,
                        )
                        return result

                    async def on_call_tool(self, context, call_next):
                        session_id = toolkit._resolve_context_session_id(
                            getattr(context, "fastmcp_context", None),
                            transport=transport,
                        )
                        if session_id:
                            metadata = toolkit._extract_request_identity_metadata(
                                transport,
                                current_host,
                                current_port,
                            )
                            toolkit._store_session_metadata(session_id, metadata)
                        return await call_next(context)

                mcp.add_middleware(MCPAuditMiddleware())

                if hasattr(mcp, "_call_tool_middleware"):
                    original_call_tool_middleware = mcp._call_tool_middleware

                    async def audited_call_tool_middleware(key: str, arguments: dict[str, Any]) -> Any:
                        ctx = get_fastmcp_context() if get_fastmcp_context is not None else None
                        session_id = toolkit._resolve_context_session_id(ctx, transport)
                        current_metadata = toolkit._get_session_metadata(session_id)
                        if not current_metadata and session_id:
                            # Fallback bootstrap if on_initialize did not fire
                            # (e.g. legacy clients that skip the initialize step).
                            toolkit._apply_audit_session_started(
                                session_id,
                                transport=transport,
                                server_host=current_host,
                                server_port=current_port,
                                ctx=ctx,
                            )

                        invocation_id = toolkit._generate_invocation_id()
                        hana_correlation_id = toolkit._generate_hana_correlation_id()
                        redacted_args = toolkit._redact_audit_payload(arguments or {})
                        tool_audit_metadata = toolkit._extract_tool_audit_metadata(
                            key,
                            arguments or {},
                            redacted_args=redacted_args,
                        )
                        started_payload = {
                            "tool_name": key,
                            "status": "started",
                            "input_redacted": redacted_args,
                            "input_hash": toolkit._hash_payload(redacted_args),
                            **tool_audit_metadata,
                        }
                        started_event = toolkit._build_audit_event(
                            "mcp.tool.invocation.started",
                            session_id,
                            payload=started_payload,
                            invocation_id=invocation_id,
                            hana_correlation_id=hana_correlation_id,
                        )

                        started_at = time.time()
                        try:
                            toolkit._ensure_hana_execution_context(
                                session_id,
                                invocation_id,
                                hana_correlation_id,
                                key,
                                started_event=started_event,
                            )
                            toolkit._emit_audit_event(started_event)
                            result = await original_call_tool_middleware(key, arguments)
                        except Exception as exc:
                            duration_ms = int((time.time() - started_at) * 1000)
                            failure_payload = {
                                "tool_name": key,
                                "status": "failure",
                                "duration_ms": duration_ms,
                                "error_code": type(exc).__name__,
                                "error_class": type(exc).__name__,
                                "error_message_redacted": str(exc)[:5000],
                                "input_hash": started_payload["input_hash"],
                                **tool_audit_metadata,
                            }
                            failure_event = toolkit._build_audit_event(
                                "mcp.tool.invocation.failed",
                                session_id,
                                payload=failure_payload,
                                invocation_id=invocation_id,
                                hana_correlation_id=hana_correlation_id,
                            )
                            toolkit._update_hana_session_event(failure_event)
                            toolkit._emit_audit_event(failure_event)
                            raise

                        duration_ms = int((time.time() - started_at) * 1000)
                        audit_result = toolkit._extract_audit_result_payload(result)
                        model_storage_metadata = toolkit._extract_model_storage_metadata(result)
                        success_tool_audit_metadata = toolkit._extract_tool_audit_metadata(
                            key,
                            arguments or {},
                            redacted_args=redacted_args,
                            result=result,
                        )
                        success_payload = {
                            "tool_name": key,
                            "status": "success",
                            "duration_ms": duration_ms,
                            "result_summary": toolkit._summarize_tool_result(result),
                            "input_hash": started_payload["input_hash"],
                            "output_hash": toolkit._hash_payload(toolkit._redact_audit_payload(audit_result)),
                            **success_tool_audit_metadata,
                            **model_storage_metadata,
                        }
                        success_event = toolkit._build_audit_event(
                            "mcp.tool.invocation.succeeded",
                            session_id,
                            payload=success_payload,
                            invocation_id=invocation_id,
                            hana_correlation_id=hana_correlation_id,
                        )
                        toolkit._update_hana_session_event(success_event)
                        toolkit._emit_audit_event(success_event)
                        return result

                    mcp._call_tool_middleware = audited_call_tool_middleware

            # --- Admin tool: update connection context at runtime ---
            # This is intentionally registered before business tools, and is transport-agnostic.
            # NOTE: For stdio transport, any stray stdout breaks the protocol; we only use logging.
            @mcp.tool()
            def admin_update_connection_context(
                address: Annotated[str, TxtDoc("HANA host/address") if TxtDoc is not None else str],
                port: Annotated[int, TxtDoc("HANA port") if TxtDoc is not None else int] = 443,
                user: Annotated[str, TxtDoc("HANA user") if TxtDoc is not None else str] = "",
                password: Annotated[str, TxtDoc("HANA password") if TxtDoc is not None else str] = "",
                encrypt: Annotated[Optional[bool], TxtDoc("Use TLS") if TxtDoc is not None else Optional[bool]] = None,
                ssl_validate_certificate: Annotated[Optional[bool], TxtDoc("Validate TLS certificate") if TxtDoc is not None else Optional[bool]] = None,
                test_connection: Annotated[bool, TxtDoc("If true, open a test connection before switching") if TxtDoc is not None else bool] = False,
            ):
                """Update the toolkit's HANA ConnectionContext without restarting the MCP server."""
                new_params: dict[str, Any] = {
                    "address": address,
                    "port": port,
                    "user": user,
                    "password": password,
                }
                if encrypt is not None:
                    new_params["encrypt"] = bool(encrypt)
                if ssl_validate_certificate is not None:
                    new_params["sslValidateCertificate"] = bool(ssl_validate_certificate)

                # Optionally validate credentials/route before mutating live tools.
                if test_connection:
                    try:
                        test_cc = ConnectionContext(**new_params)
                        # Best-effort ping: open/close if supported.
                        close_meth = getattr(test_cc, "close", None)
                        if callable(close_meth):
                            close_meth()
                    except Exception as e:
                        logging.error("Connection test failed: %s", e)
                        return {"ok": False, "error": str(e)}

                # Swap context
                try:
                    self.connection_context = ConnectionContext(**new_params)
                except Exception as e:
                    logging.error("Failed to build ConnectionContext: %s", e)
                    return {"ok": False, "error": str(e)}

                # Propagate to tools. Many tools store connection_context on construction.
                refresh_stats = _refresh_tools_for_new_context(self)
                logging.warning(
                    "✅ Updated ConnectionContext for toolkit; tools updated=%s, defaults rebuilt=%s",
                    refresh_stats.get("tools_updated_in_place"),
                    refresh_stats.get("default_tools_rebuilt"),
                )
                return {
                    "ok": True,
                    "connection": _redact_dict(new_params),
                    **refresh_stats,
                }

            @mcp.tool()
            def admin_reload_connection_context_from_env(
                test_connection: Annotated[bool, TxtDoc("If true, open a test connection before switching") if TxtDoc is not None else bool] = False,
            ):
                """Reload HANA ConnectionContext from server environment variables (HANA_*) without restarting the MCP server."""
                try:
                    params = _build_cc_params_from_env()
                except Exception as e:
                    logging.error("Failed to read HANA_* env vars: %s", e)
                    return {"ok": False, "error": str(e)}

                if test_connection:
                    try:
                        test_cc = ConnectionContext(**params)
                        close_meth = getattr(test_cc, "close", None)
                        if callable(close_meth):
                            close_meth()
                    except Exception as e:
                        logging.error("Connection test failed: %s", e)
                        return {"ok": False, "error": str(e)}

                try:
                    self.connection_context = ConnectionContext(**params)
                except Exception as e:
                    logging.error("Failed to build ConnectionContext from env: %s", e)
                    return {"ok": False, "error": str(e)}

                refresh_stats = _refresh_tools_for_new_context(self)
                logging.warning(
                    "✅ Reloaded ConnectionContext from env; tools updated=%s, defaults rebuilt=%s",
                    refresh_stats.get("tools_updated_in_place"),
                    refresh_stats.get("default_tools_rebuilt"),
                )
                return {
                    "ok": True,
                    "connection": _redact_dict(params),
                    **refresh_stats,
                }

            # 获取并注册所有工具
            tools = self.get_tools()
            registered_tools = []
            for tool in tools:
                context_annotation = FastMCPContext if transport == "http" else LegacyMCPContext

                # 为 FastMCP 构建带真实参数签名与描述的包装器（方案A）
                # 1) 基础包装执行体（接收命名参数）
                def _exec_wrapper(wrapped_tool):
                    def _inner(ctx=None, **kwargs):
                        if transport == "http":
                            return wrapped_tool._run(**kwargs)

                        session_id = self._resolve_context_session_id(ctx, transport)
                        current_metadata = self._get_session_metadata(session_id)
                        if not current_metadata and session_id:
                            # Mirrors HTTP's MCPAuditMiddleware.on_initialize for
                            # transports (stdio / sse) that legacy
                            # mcp.server.fastmcp.FastMCP cannot host a Middleware
                            # against. Reads clientInfo from ServerSession.client_params
                            # via ctx, then writes mcp.session.started to the audit
                            # log AND to HANA Session Variables.
                            self._apply_audit_session_started(
                                session_id,
                                transport=transport,
                                server_host=host,
                                server_port=port,
                                ctx=ctx,
                            )

                        invocation_id = self._generate_invocation_id()
                        hana_correlation_id = self._generate_hana_correlation_id()
                        redacted_args = self._redact_audit_payload(kwargs)
                        tool_audit_metadata = self._extract_tool_audit_metadata(
                            wrapped_tool.name,
                            kwargs,
                            redacted_args=redacted_args,
                        )
                        started_payload = {
                            "tool_name": wrapped_tool.name,
                            "status": "started",
                            "input_redacted": redacted_args,
                            "input_hash": self._hash_payload(redacted_args),
                            **tool_audit_metadata,
                        }
                        started_event = self._build_audit_event(
                            "mcp.tool.invocation.started",
                            session_id,
                            payload=started_payload,
                            invocation_id=invocation_id,
                            hana_correlation_id=hana_correlation_id,
                        )
                        started_at = time.time()
                        try:
                            self._ensure_hana_execution_context(
                                session_id,
                                invocation_id,
                                hana_correlation_id,
                                wrapped_tool.name,
                                started_event=started_event,
                            )
                            self._emit_audit_event(started_event)
                            result = wrapped_tool._run(**kwargs)
                        except Exception as e:
                            duration_ms = int((time.time() - started_at) * 1000)
                            failure_payload = {
                                "tool_name": wrapped_tool.name,
                                "status": "failure",
                                "duration_ms": duration_ms,
                                "error_code": type(e).__name__,
                                "error_class": type(e).__name__,
                                "error_message_redacted": str(e)[:5000],
                                "input_hash": started_payload["input_hash"],
                                **tool_audit_metadata,
                            }
                            failure_event = self._build_audit_event(
                                "mcp.tool.invocation.failed",
                                session_id,
                                payload=failure_payload,
                                invocation_id=invocation_id,
                                hana_correlation_id=hana_correlation_id,
                            )
                            self._update_hana_session_event(failure_event)
                            self._emit_audit_event(failure_event)
                            logging.error("Tool %s failed: %s", wrapped_tool.name, str(e))
                            return {
                                "error": str(e),
                                "tool": wrapped_tool.name,
                                "invocation_id": invocation_id,
                                "hana_correlation_id": hana_correlation_id,
                            }
                        duration_ms = int((time.time() - started_at) * 1000)
                        model_storage_metadata = self._extract_model_storage_metadata(result)
                        success_tool_audit_metadata = self._extract_tool_audit_metadata(
                            wrapped_tool.name,
                            kwargs,
                            redacted_args=redacted_args,
                            result=result,
                        )
                        success_payload = {
                            "tool_name": wrapped_tool.name,
                            "status": "success",
                            "duration_ms": duration_ms,
                            "result_summary": self._summarize_tool_result(result),
                            "input_hash": started_payload["input_hash"],
                            "output_hash": self._hash_payload(self._redact_audit_payload(result)),
                            **success_tool_audit_metadata,
                            **model_storage_metadata,
                        }
                        success_event = self._build_audit_event(
                            "mcp.tool.invocation.succeeded",
                            session_id,
                            payload=success_payload,
                            invocation_id=invocation_id,
                            hana_correlation_id=hana_correlation_id,
                        )
                        self._update_hana_session_event(success_event)
                        self._emit_audit_event(success_event)
                        return result
                    return _inner

                tool_wrapper = _exec_wrapper(tool)
                tool_wrapper.__name__ = tool.name
                tool_wrapper.__doc__ = tool.description

                # 2) 从 Pydantic args_schema 派生参数签名与注解（包含描述）
                parameters = []
                annotations: dict[str, Any] = {}
                required_fields = []

                if hasattr(tool, 'args_schema') and tool.args_schema:
                    schema_model = tool.args_schema
                    # 获取 required 列表，兼容 v1/v2
                    required_fields = []
                    try:
                        if hasattr(schema_model, 'model_json_schema'):
                            # pydantic v2
                            json_schema = schema_model.model_json_schema()
                            required_fields = json_schema.get('required', []) or []
                        elif hasattr(schema_model, 'schema'):
                            # pydantic v1
                            json_schema = schema_model.schema()
                            required_fields = json_schema.get('required', []) or []
                    except Exception:  # 容错
                        required_fields = []

                    # 字段列表与类型/描述/默认
                    if hasattr(schema_model, 'model_fields'):
                        # pydantic v2
                        fields_iter = schema_model.model_fields.items()
                        for fname, finfo in fields_iter:
                            ftype = getattr(finfo, 'annotation', Any)
                            fdesc = getattr(finfo, 'description', None)
                            # 使用 Annotated 注入描述，若无描述则不包裹
                            if fdesc and PydField is not None:
                                annotated_type = Annotated[ftype, PydField(description=fdesc)]
                            elif fdesc and TxtDoc is not None:
                                annotated_type = Annotated[ftype, TxtDoc(fdesc)]
                            else:
                                annotated_type = ftype

                            annotations[fname] = annotated_type

                            # 默认值处理：若必填，则无默认；否则使用字段默认（可为 None）
                            default_exists = hasattr(finfo, 'default')
                            if fname in required_fields:
                                param = inspect.Parameter(
                                    fname,
                                    kind=inspect.Parameter.KEYWORD_ONLY,
                                    default=inspect._empty
                                )
                            else:
                                default_value = getattr(finfo, 'default', None) if default_exists else None
                                param = inspect.Parameter(
                                    fname,
                                    kind=inspect.Parameter.KEYWORD_ONLY,
                                    default=default_value
                                )
                            parameters.append(param)
                    elif hasattr(schema_model, '__fields__'):
                        # pydantic v1
                        fields_iter = schema_model.__fields__.items()
                        for fname, mfield in fields_iter:
                            ftype = mfield.outer_type_ if hasattr(mfield, 'outer_type_') else mfield.type_ if hasattr(mfield, 'type_') else Any
                            fdesc = None
                            try:
                                # v1: 描述在 field_info.description
                                fdesc = getattr(mfield.field_info, 'description', None)
                            except Exception:
                                fdesc = None

                            if fdesc and PydField is not None:
                                annotated_type = Annotated[ftype, PydField(description=fdesc)]
                            elif fdesc and TxtDoc is not None:
                                annotated_type = Annotated[ftype, TxtDoc(fdesc)]
                            else:
                                annotated_type = ftype

                            annotations[fname] = annotated_type

                            # 必填判断：优先使用 required 列表；否则使用 mfield.required
                            is_required = fname in required_fields
                            if not is_required:
                                try:
                                    is_required = bool(getattr(mfield, 'required', False))
                                except Exception:
                                    is_required = False

                            if is_required:
                                param = inspect.Parameter(
                                    fname,
                                    kind=inspect.Parameter.KEYWORD_ONLY,
                                    default=inspect._empty
                                )
                            else:
                                default_value = None
                                try:
                                    default_value = mfield.default if hasattr(mfield, 'default') else None
                                except Exception:
                                    default_value = None
                                param = inspect.Parameter(
                                    fname,
                                    kind=inspect.Parameter.KEYWORD_ONLY,
                                    default=default_value
                                )
                            parameters.append(param)

                # 应用签名与注解到包装器
                if context_annotation is not None:
                    annotations["ctx"] = context_annotation
                    parameters.append(
                        inspect.Parameter(
                            "ctx",
                            kind=inspect.Parameter.KEYWORD_ONLY,
                            default=None,
                        )
                    )
                if parameters:
                    sig = inspect.Signature(parameters=parameters)
                    try:
                        tool_wrapper.__signature__ = sig
                    except Exception:
                        pass
                if annotations:
                    tool_wrapper.__annotations__ = annotations

                # 3) 注册到 MCP（所有传输均注册执行体；非 HTTP 额外覆盖 schema）
                mcp.tool()(tool_wrapper)
                if transport != "http":
                    # stdio/sse：覆盖其参数 schema 为显式 Pydantic JSON Schema（方案C）
                    try:
                        explicit_schema = None
                        if hasattr(tool, 'args_schema') and tool.args_schema:
                            if hasattr(tool.args_schema, 'model_json_schema'):
                                explicit_schema = tool.args_schema.model_json_schema(by_alias=True)
                            elif hasattr(tool.args_schema, 'schema'):
                                explicit_schema = tool.args_schema.schema(by_alias=True)
                        if explicit_schema:
                            # 获取内部 Tool 并覆盖 parameters（list_tools 将返回此 schema）
                            info = getattr(mcp, '_tool_manager', None)
                            if info is not None:
                                internal_tool = info.get_tool(tool.name)
                                if internal_tool is not None:
                                    internal_tool.parameters = explicit_schema
                                    logging.debug("🧩 Overrode schema for %s", tool.name)
                    except Exception as e:
                        logging.warning("Failed to override schema for %s: %s", tool.name, e)
                registered_tools.append(tool.name)
                try:
                    param_list = list(getattr(tool_wrapper, "__signature__", inspect.Signature()).parameters.keys())
                except Exception:
                    param_list = []
                logging.info("✅ Registered tool: %s", tool.name)
                logging.debug("🔎 Params for %s: %s", tool.name, ", ".join(param_list))

            # 安全配置
            server_args = {"transport": transport}
            if transport == "stdio" and not hasattr(sys.stdout, 'buffer'):
                logging.warning("⚠️  Unsupported stdio, switching to SSE")
                transport = "sse"
                port = original_port  # 重置端口重试
                attempts = 0         # 重置尝试次数
                continue

            if auth_token:
                server_args["auth_token"] = auth_token
                logging.info("🔐 Authentication enabled")

            # 启动服务器线程
            def run_server(mcp_instance, server_args):
                try:
                    logging.info("🚀 Starting MCP server on port %s...", port)
                    if server_args.get("transport") == "http":
                        # fastmcp prints a server banner and may perform a PyPI version check.
                        # In locked-down envs (or misconfigured SSL_CERT_FILE), that check can crash
                        # the server before it starts listening. Disable both explicitly.
                        try:
                            import fastmcp
                            fastmcp.settings.check_for_updates = "off"
                            fastmcp.settings.show_cli_banner = False
                        except Exception:
                            pass
                        # fastmcp HTTP 运行参数
                        # 使用标准路径 /mcp，并启用 JSON 响应
                        mcp_instance.run(
                            transport="http",
                            host=server_settings.get("host", "127.0.0.1"),
                            port=port,
                            path="/mcp",
                            json_response=True,
                        )
                    else:
                        mcp_instance.run(**server_args)
                except Exception as e:
                    logging.exception("Server crashed: %s", str(e))
                    # 这里不再自动重启，由外部监控

            logging.info("Starting MCP server in background thread...")
            server_thread = threading.Thread(
                target=run_server,
                args=(mcp, server_args),
                name=f"MCP-Server-Port-{port}",
                daemon=True
            )
            server_thread.start()
            logging.info("🚀 MCP server started on port %s with tools: %s", port, registered_tools)
            # Record server instance and thread for later shutdown
            try:
                key = (server_settings.get("host", "127.0.0.1"), port, transport)
                HANAMLToolkit._global_mcp_servers[key] = {
                    "instance": mcp,
                    "thread": server_thread,
                    "name": server_settings.get("name", server_name),
                    "host": server_settings.get("host", "127.0.0.1"),
                    "port": port,
                    "transport": transport,
                }
                logging.debug("🗂️ Registered MCP server in registry: %s", key)
            except Exception as e:
                logging.warning("Failed to register server in registry: %s", e)
            return  # 成功启动，退出函数

        # 所有尝试失败
        logging.error("❌ Failed to start server after %s attempts", max_retries)
        raise RuntimeError(f"Could not find available port in range {original_port}-{original_port + max_retries}")

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        if self.vectordb is not None:
            get_code = GetCodeTemplateFromVectorDB()
            get_code.set_vectordb(self.vectordb)
            return self.used_tools + [get_code]
        return self.used_tools

    def stop_mcp_server(
        self,
        host: str = "127.0.0.1",
        port: int = 8001,
        transport: str = "sse",
        force: bool = False,
        timeout: float = 5.0,
    ) -> bool:
        """
        停止指定地址与端口的 MCP 服务。

        参数
        ------
        host : str
            MCP 服务的主机地址。
        port : int
            MCP 服务的端口（stdio 传输也使用此键进行注册标识）。
        transport : {"stdio", "sse", "http"}
            传输类型，需要与启动时一致以匹配注册记录。
        force : bool
            若正常关闭失败，是否尝试强制关闭（尽力而为，可能无法完全保证）。
        timeout : float
            等待服务器线程退出的最长秒数。

        返回
        ------
        bool
            若成功触发关闭并线程在超时前结束，返回 True；否则返回 False。
        """
        key = (host, port, transport)
        info = HANAMLToolkit._global_mcp_servers.get(key)
        if not info:
            logging.warning("No MCP server found for %s", key)
            return False

        mcp_instance = info.get("instance")
        server_thread: threading.Thread = info.get("thread")

        # Try graceful shutdown via common method names
        stopped_gracefully = False
        for meth_name in ("shutdown", "stop", "close"):
            try:
                meth = getattr(mcp_instance, meth_name, None)
                if callable(meth):
                    logging.info("Attempting graceful '%s' on MCP server %s", meth_name, key)
                    try:
                        meth()
                        stopped_gracefully = True
                        break
                    except Exception as e:
                        logging.warning("'%s' failed for %s: %s", meth_name, key, e)
            except Exception:
                pass

        # Wait for thread exit
        if server_thread and server_thread.is_alive():
            try:
                server_thread.join(timeout)
            except Exception:
                pass

        # If still alive and force requested, attempt best-effort termination hooks
        if server_thread and server_thread.is_alive() and force:
            logging.info("Server thread still alive after graceful attempt; trying forceful shutdown for %s", key)
            # Best-effort: signal known event attributes if present
            for attr in ("shutdown_event", "stop_event"):
                try:
                    ev = getattr(mcp_instance, attr, None)
                    if ev:
                        try:
                            ev.set()
                        except Exception:
                            pass
                except Exception:
                    pass
            for attr in ("should_exit", "force_exit"):
                for candidate in (mcp_instance, getattr(mcp_instance, "_mcp_server", None)):
                    try:
                        if candidate is not None and hasattr(candidate, attr):
                            setattr(candidate, attr, True)
                    except Exception:
                        pass
            try:
                server_thread.join(timeout)
            except Exception:
                pass

        alive = server_thread.is_alive() if server_thread else False
        success = stopped_gracefully and not alive

        # fastmcp HTTP/SSE may spawn a separate server process (e.g., uvicorn) that outlives
        # this thread. In that case, best-effort shutdown may not be observable here.
        # If force=True, treat registry cleanup as success to prevent leaking state.
        if force and (alive or not stopped_gracefully):
            success = True

        # 仅在服务已停止（或本就不在运行）时移除注册记录。
        # force=True 时也会清理注册记录（best-effort 关闭可能无法同步观察）。
        if success or (not alive):
            try:
                HANAMLToolkit._global_mcp_servers.pop(key, None)
            except Exception:
                pass
            if success:
                logging.info("✅ MCP server stopped: %s", key)
            else:
                logging.info("ℹ️ MCP server already stopped: %s", key)
        else:
            logging.warning("⚠️ MCP server may still be running: %s", key)
        return success

    def stop_all_mcp_servers(self, force: bool = False, timeout: float = 5.0) -> int:
        """
        关闭全部已注册 MCP 服务。

        参数
        ------
        force : bool
            若正常关闭失败，是否尝试强制关闭。
        timeout : float
            每个服务等待线程退出的最长秒数。

        返回
        ------
        int
            成功关闭的服务数量。
        """
        keys = list(HANAMLToolkit._global_mcp_servers.keys())
        success_count = 0
        for host, port, transport in keys:
            if self.stop_mcp_server(host=host, port=port, transport=transport, force=force, timeout=timeout):
                success_count += 1
        logging.info("Stopped %s MCP servers", success_count)
        return success_count
