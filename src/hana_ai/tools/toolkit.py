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
import re
from contextlib import closing
import logging
import threading
import time
from typing import Optional, List, Annotated, Any, ClassVar
import inspect
from urllib.parse import urlparse, parse_qs
try:
    from pydantic import Field as PydField
except Exception:
    PydField = None
try:
    from typing_extensions import Doc as TxtDoc  # PEP 727 style doc metadata
except Exception:
    TxtDoc = None
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mcp"])
    from mcp.server.fastmcp import FastMCP

# For HTTP transport support via fastmcp (separate package)
try:
    from fastmcp import FastMCP as FastMCPHTTP
    from fastmcp.tools import Tool as HTTPTool
except ImportError:
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastmcp"])
        from fastmcp import FastMCP as FastMCPHTTP
        from fastmcp.tools import Tool as HTTPTool
    except Exception:
        FastMCPHTTP = None

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


def _str_to_bool(val: Optional[str]) -> Optional[bool]:
    if val is None:
        return None
    v = str(val).strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return None


def _read_kv_file(file_path: str) -> dict[str, str]:
    """Read a simple env-style key/value file.

    Supports lines like:
      VCAP_SERVICES='...json...'
      HDI_DEPLOY_OPTIONS='...json...'
      export KEY=value

    Important: For VCAP_SERVICES we must NOT unescape backslashes (e.g. \n),
    because they are part of the JSON and must remain escaped for json.loads().
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    out: dict[str, str] = {}
    for line in raw_lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export "):].lstrip()
        if "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        # Strip optional trailing ';'
        if value.endswith(";"):
            value = value[:-1].rstrip()
        # Strip surrounding quotes without interpreting escapes.
        if (len(value) >= 2) and ((value[0] == value[-1]) and value[0] in ("'", '"')):
            value = value[1:-1]
        out[key] = value
    return out


def _build_cc_params_from_vcap_services_json(vcap_services: dict[str, Any]) -> dict[str, Any]:
    """Build ConnectionContext params from a VCAP_SERVICES dict."""
    if not isinstance(vcap_services, dict):
        raise ValueError("VCAP_SERVICES must be a JSON object")

    hana_services = vcap_services.get("hana")
    if not hana_services or not isinstance(hana_services, list):
        raise ValueError("VCAP_SERVICES does not contain a 'hana' service list")

    svc0 = hana_services[0] if hana_services else None
    if not isinstance(svc0, dict):
        raise ValueError("VCAP_SERVICES['hana'][0] is invalid")

    creds = svc0.get("credentials") or {}
    if not isinstance(creds, dict):
        raise ValueError("VCAP_SERVICES['hana'][0].credentials is invalid")

    address = creds.get("host") or creds.get("address")
    port_raw = creds.get("port", 443)
    user = creds.get("user")
    password = creds.get("password")

    # Best-effort TLS flags from JDBC URL query: encrypt / validateCertificate
    encrypt = None
    ssl_validate = None
    jdbc_url = creds.get("url")
    if isinstance(jdbc_url, str) and jdbc_url:
        try:
            # url might be like: jdbc:sap://host:443?encrypt=true&validateCertificate=true
            q = urlparse(jdbc_url).query
            qs = parse_qs(q)
            encrypt = _str_to_bool((qs.get("encrypt") or [None])[0])
            ssl_validate = _str_to_bool((qs.get("validateCertificate") or [None])[0])
        except Exception:
            pass

    missing = [
        k
        for k, v in {"address": address, "user": user, "password": password}.items()
        if not v
    ]
    if missing:
        raise ValueError(f"Missing required credential fields in VCAP_SERVICES: {', '.join(missing)}")

    try:
        port = int(port_raw)
    except Exception as e:
        raise ValueError(f"Invalid port in VCAP_SERVICES credentials: {port_raw}") from e

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


def _build_cc_params_from_vcap_services_file(file_path: str) -> dict[str, Any]:
    """Load ConnectionContext params from a file containing VCAP_SERVICES='...json...'."""
    if not file_path or not isinstance(file_path, str):
        raise ValueError("file_path is required")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    kv = _read_kv_file(file_path)
    vcap_raw = kv.get("VCAP_SERVICES")
    if not vcap_raw:
        raise ValueError("VCAP_SERVICES not found in file")

    try:
        vcap_services = json.loads(vcap_raw)
    except Exception as e:
        # Try a small salvage: remove a leading 'VCAP_SERVICES=' prefix if user pasted whole line.
        m = re.match(r"^VCAP_SERVICES=(.+)$", vcap_raw.strip())
        if m:
            try:
                candidate = m.group(1).strip()
                if (len(candidate) >= 2) and ((candidate[0] == candidate[-1]) and candidate[0] in ("'", '"')):
                    candidate = candidate[1:-1]
                vcap_services = json.loads(candidate)
            except Exception as e2:
                raise ValueError(f"Invalid VCAP_SERVICES JSON in file: {e2}") from e2
        else:
            raise ValueError(f"Invalid VCAP_SERVICES JSON in file: {e}") from e

    return _build_cc_params_from_vcap_services_json(vcap_services)


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


def _connection_context_brief(cc: Any) -> dict[str, Any]:
    """Best-effort extract of key ConnectionContext fields (non-sensitive)."""
    if cc is None:
        return {}
    out: dict[str, Any] = {}
    for key, attrs in {
        "address": ("address", "_address", "host", "_host"),
        "port": ("port", "_port"),
        "user": ("user", "_user"),
        "encrypt": ("encrypt", "_encrypt"),
        "sslValidateCertificate": ("sslValidateCertificate", "_sslValidateCertificate"),
        "userkey": ("userkey", "_userkey"),
    }.items():
        for attr in attrs:
            try:
                v = getattr(cc, attr, None)
            except Exception:
                v = None
            if v is not None:
                out[key] = v
                break
    return out

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
    # Registry of running MCP servers keyed by (host, port, transport)
    # Use a class-level global registry so multiple toolkit instances share state.
    _global_mcp_servers: ClassVar[dict] = {}
    mcp_servers: dict = None

    def __init__(self, connection_context, used_tools=None, return_direct=None):
        super().__init__(connection_context=connection_context)
        # Initialize server registry (shared across instances)
        self.mcp_servers = HANAMLToolkit._global_mcp_servers
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
                if FastMCPHTTP is None or HTTPTool is None:
                    logging.error("HTTP transport requested but 'fastmcp' package is unavailable.")
                    raise RuntimeError("HTTP transport not supported (fastmcp missing)")
                # 为 HTTP 预构建 Tool 列表（方案C：显式 inputSchema）
                pre_tools = []
                for tool in self.get_tools():
                    if hasattr(tool, 'args_schema') and tool.args_schema:
                        try:
                            schema = None
                            if hasattr(tool.args_schema, 'model_json_schema'):
                                schema = tool.args_schema.model_json_schema(by_alias=True)
                            elif hasattr(tool.args_schema, 'schema'):
                                schema = tool.args_schema.schema(by_alias=True)
                            if schema is None:
                                continue
                            http_tool = HTTPTool(
                                name=tool.name,
                                title=getattr(tool, 'name', None),
                                description=getattr(tool, 'description', '') or tool.name,
                                parameters=schema,
                            )
                            pre_tools.append(http_tool)
                        except Exception as e:
                            logging.warning("Failed to build explicit schema for %s: %s", tool.name, e)
                # fastmcp 的构造函数以 name 作为位置参数，并支持 tools 列表
                mcp = FastMCPHTTP(server_settings.get("name", "HANATools"), tools=pre_tools, host=server_settings.get("host", "127.0.0.1"), port=port, streamable_http_path="/mcp", json_response=True)
                # 检查端口可用性
                if not self.is_port_available(port):
                    logging.warning("⚠️  Port %s occupied, trying next port", port)
                    port += 1
                    attempts += 1
                    time.sleep(0.2)
                    continue
            else:
                mcp = FastMCP(**server_settings)

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
            def admin_reload_connection_context_from_file(
                file_path: Annotated[str, TxtDoc("Path to a file containing VCAP_SERVICES='...json...' (required)") if TxtDoc is not None else str],
                test_connection: Annotated[bool, TxtDoc("If true, open a test connection before switching") if TxtDoc is not None else bool] = False,
            ):
                """Reload HANA ConnectionContext from a file (VCAP_SERVICES) without restarting the MCP server."""
                prev_cc = getattr(self, "connection_context", None)
                prev_brief = _connection_context_brief(prev_cc)
                prev_id = id(prev_cc) if prev_cc is not None else None
                try:
                    params = _build_cc_params_from_vcap_services_file(file_path)
                except Exception as e:
                    logging.error("Failed to read ConnectionContext from file '%s': %s", file_path, e)
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
                    logging.error("Failed to build ConnectionContext from file: %s", e)
                    return {"ok": False, "error": str(e)}

                refresh_stats = _refresh_tools_for_new_context(self)
                logging.warning(
                    "✅ Reloaded ConnectionContext from file; tools updated=%s, defaults rebuilt=%s",
                    refresh_stats.get("tools_updated_in_place"),
                    refresh_stats.get("default_tools_rebuilt"),
                )
                new_cc = getattr(self, "connection_context", None)
                new_id = id(new_cc) if new_cc is not None else None
                return {
                    "ok": True,
                    "switched": (prev_id is not None and new_id is not None and prev_id != new_id),
                    "previous_connection": _redact_dict(prev_brief),
                    "connection": _redact_dict(params),
                    **refresh_stats,
                }

            # 获取并注册所有工具
            tools = self.get_tools()
            registered_tools = []
            for tool in tools:
                # 为 FastMCP 构建带真实参数签名与描述的包装器（方案A）
                # 1) 基础包装执行体（接收命名参数）
                def _exec_wrapper(wrapped_tool):
                    def _inner(**kwargs):
                        try:
                            return wrapped_tool._run(**kwargs)
                        except Exception as e:
                            logging.error("Tool %s failed: %s", wrapped_tool.name, str(e))
                            return {"error": str(e), "tool": wrapped_tool.name}
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
            logging.warning("Server thread still alive after graceful attempt; trying forceful shutdown for %s", key)
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
