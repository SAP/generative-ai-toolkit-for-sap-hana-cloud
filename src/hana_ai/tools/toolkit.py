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
from typing import Optional, List
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mcp"])
    from mcp.server.fastmcp import FastMCP

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
from hana_ai.tools.hana_ml_tools.automatic_timeseries_tools import AutomaticTimeSeriesFitAndSave, AutomaticTimeseriesLoadModelAndPredict, AutomaticTimeseriesLoadModelAndScore
from hana_ai.tools.hana_ml_tools.ts_check_tools import TimeSeriesCheck
from hana_ai.tools.hana_ml_tools.ts_outlier_detection_tools import TSOutlierDetection
from hana_ai.tools.hana_ml_tools.ts_accuracy_measure_tools import AccuracyMeasure
from hana_ai.tools.hana_ml_tools.unsupported_tools import ClassificationTool, RegressionTool

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

    def __init__(self, connection_context, used_tools=None, return_direct=None):
        super().__init__(connection_context=connection_context)
        self.default_tools = [
            AccuracyMeasure(connection_context=self.connection_context),
            AdditiveModelForecastFitAndSave(connection_context=self.connection_context),
            AdditiveModelForecastLoadModelAndPredict(connection_context=self.connection_context),
            AutomaticTimeSeriesFitAndSave(connection_context=self.connection_context),
            AutomaticTimeseriesLoadModelAndPredict(connection_context=self.connection_context),
            AutomaticTimeseriesLoadModelAndScore(connection_context=self.connection_context),
            CAPArtifactsTool(connection_context=self.connection_context),
            FetchDataTool(connection_context=self.connection_context),
            ForecastLinePlot(connection_context=self.connection_context),
            IntermittentForecast(connection_context=self.connection_context),
            ListModels(connection_context=self.connection_context),
            DeleteModels(connection_context=self.connection_context),
            TimeSeriesDatasetReport(connection_context=self.connection_context),
            TimeSeriesCheck(connection_context=self.connection_context),
            TSOutlierDetection(connection_context=self.connection_context),
            ClassificationTool(connection_context=self.connection_context),
            RegressionTool(connection_context=self.connection_context)
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
        version: str = "1.0",
        host: str = "127.0.0.1",
        transport: str = "stdio",
        sse_port: int = 8001,
        auth_token: Optional[str] = None,
        max_retries: int = 5
    ):
        """
        启动MCP服务器并注册所有工具
        
        参数:
        - server_name: MCP服务名称
        - version: 服务版本号
        - transport: 传输协议 (stdio/sse/http)
        - sse_port: SSE协议使用的端口
        - auth_token: 认证令牌
        - max_retries: 最大端口重试次数
        """
        attempts = 0
        original_port = sse_port
        port = sse_port

        while attempts < max_retries:
            # 初始化MCP配置
            server_settings = {
                "name": server_name,
                "version": version,
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

            # 创建MCP实例
            mcp = FastMCP(**server_settings)

            # 获取并注册所有工具
            tools = self.get_tools()
            registered_tools = []
            for tool in tools:
                # 使用默认参数绑定当前工具
                def create_tool_wrapper(wrapped_tool=tool):
                    def tool_wrapper(**kwargs):
                        try:
                            return wrapped_tool._run(**kwargs)
                        except Exception as e:
                            logging.error("Tool %s failed: %s", wrapped_tool.name, str(e))
                            return {"error": str(e), "tool": wrapped_tool.name}
                    return tool_wrapper

                tool_wrapper = create_tool_wrapper()
                tool_wrapper.__name__ = tool.name
                tool_wrapper.__doc__ = tool.description

                # 设置正确的参数类型注解
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    if hasattr(tool.args_schema, 'model_fields'):
                        # Pydantic v2 的访问方式
                        annotations = {
                            k: v.annotation
                            for k, v in tool.args_schema.model_fields.items()
                        }
                    else:
                        # Pydantic v1 的访问方式
                        annotations = tool.args_schema.__annotations__

                    tool_wrapper.__annotations__ = annotations

                mcp.tool()(tool_wrapper)
                registered_tools.append(tool.name)
                logging.info("✅ Registered tool: %s", tool.name)

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
