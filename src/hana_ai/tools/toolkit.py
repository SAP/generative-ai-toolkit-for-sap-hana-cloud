"""
Toolkit for interacting with hana-ml.

The following class is available:

    * :class `HANAMLToolkit`
"""
import sys
import socket
from contextlib import closing
import logging
import threading
import time
from typing import Optional, List, Annotated, Any
import inspect
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
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool

from hana_ai.tools.code_template_tools import GetCodeTemplateFromVectorDB
from hana_ai.tools.hana_ml_tools.fetch_tools import FetchDataTool
from hana_ai.tools.hana_ml_tools.model_storage_tools import DeleteModels, ListModels
from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine
from hana_ai.tools.hana_ml_tools.additive_model_forecast_tools import AdditiveModelForecastFitAndSave, AdditiveModelForecastLoadModelAndPredict
from hana_ai.tools.hana_ml_tools.cap_artifacts_tools import CAPArtifactsForBASTool, CAPArtifactsTool
from hana_ai.tools.hana_ml_tools.intermittent_forecast_tools import IntermittentForecast
from hana_ai.tools.hana_ml_tools.ts_visualizer_tools import ForecastLinePlot, TimeSeriesDatasetReport
from hana_ai.tools.hana_ml_tools.automatic_timeseries_tools import AutomaticTimeSeriesFitAndSave, AutomaticTimeSeriesLoadModelAndPredict, AutomaticTimeSeriesLoadModelAndScore
from hana_ai.tools.hana_ml_tools.ts_check_tools import TimeSeriesCheck, MassiveTimeSeriesCheck
from hana_ai.tools.hana_ml_tools.ts_outlier_detection_tools import TSOutlierDetection
from hana_ai.tools.hana_ml_tools.ts_accuracy_measure_tools import AccuracyMeasure
from hana_ai.tools.hana_ml_tools.hdi_artifacts_tools import HDIArtifactsTool
from hana_ai.tools.hana_ml_tools.unsupported_tools import ClassificationTool, RegressionTool
from hana_ai.tools.hana_ml_tools.ts_make_predict_table import TSMakeFutureTableTool, TSMakeFutureTableForMassiveForecastTool
from hana_ai.tools.hana_ml_tools.select_statement_to_table_tools import SelectStatementToTableTool
from hana_ai.tools.hana_ml_tools.massive_automatic_timeseries_tools import MassiveAutomaticTimeSeriesFitAndSave, MassiveAutomaticTimeSeriesLoadModelAndPredict, MassiveAutomaticTimeSeriesLoadModelAndScore
from hana_ai.tools.hana_ml_tools.massive_ts_outlier_detection_tools import MassiveTSOutlierDetection

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
    mcp_servers: dict = None

    def __init__(self, connection_context, used_tools=None, return_direct=None):
        super().__init__(connection_context=connection_context)
        # Initialize server registry
        self.mcp_servers = {}
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
            ForecastLinePlot(connection_context=self.connection_context),
            IntermittentForecast(connection_context=self.connection_context),
            ListModels(connection_context=self.connection_context),
            HDIArtifactsTool(connection_context=self.connection_context),
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
            MassiveTimeSeriesCheck(connection_context=self.connection_context),
            TSMakeFutureTableForMassiveForecastTool(connection_context=self.connection_context),
            MassiveTSOutlierDetection(connection_context=self.connection_context)
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
                    logging.error("Server crashed: %s", str(e))
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
                self.mcp_servers[key] = {
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
        info = self.mcp_servers.get(key)
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

        # Remove from registry regardless to avoid stale records
        try:
            self.mcp_servers.pop(key, None)
        except Exception:
            pass

        if success:
            logging.info("✅ MCP server stopped: %s", key)
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
        keys = list(self.mcp_servers.keys())
        success_count = 0
        for host, port, transport in keys:
            if self.stop_mcp_server(host=host, port=port, transport=transport, force=force, timeout=timeout):
                success_count += 1
        logging.info("Stopped %s MCP servers", success_count)
        return success_count
