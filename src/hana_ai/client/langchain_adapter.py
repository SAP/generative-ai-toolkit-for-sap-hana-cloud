"""
LangChain工具适配器，将MCP工具转换为LangChain Tools
"""
import asyncio
import inspect
from typing import Dict, Any, List, Optional, Callable, Type, Union
from functools import wraps, partial
import json
import os

from langchain.tools import BaseTool, Tool, StructuredTool
from langchain_core.tools import BaseTool as CoreBaseTool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field, create_model

from .mcp_client import MCPClient, MCPCallResult, MCPTransport, get_mcp_client, call_mcp_tool


class MCPToolSchema(BaseModel):
    """MCP工具输入schema基类"""
    # 动态生成，这里只是基类


def create_schema_from_mcp_tool(tool_info: Dict[str, Any]) -> Type[BaseModel]:
    """从MCP工具信息创建Pydantic模型"""
    
    fields = {}
    properties = tool_info.get("inputSchema", {}).get("properties", {})
    required = tool_info.get("inputSchema", {}).get("required", [])
    
    for prop_name, prop_info in properties.items():
        prop_type = prop_info.get("type", "string")
        
        # 映射JSON Schema类型到Python类型
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        python_type = type_mapping.get(prop_type, str)
        
        # 创建字段
        field = Field(
            ... if prop_name in required else None,
            description=prop_info.get("description", "")
        )
        
        fields[prop_name] = (python_type, field)
    
    # 动态创建模型
    model_name = f"{tool_info.get('name', 'MCPTool').title()}Input"
    return create_model(model_name, **fields)


class MCPLangChainTool(BaseTool):
    """LangChain工具适配器，包装MCP工具"""
    
    mcp_tool_name: str
    mcp_client: Optional[MCPClient] = None
    transport: Union[str, MCPTransport] = MCPTransport.HTTP
    client_kwargs: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # 如果提供了MCP工具信息，使用它
        if "mcp_tool_info" in data:
            self._setup_from_tool_info(data["mcp_tool_info"])
    
    def _setup_from_tool_info(self, tool_info: Dict[str, Any]) -> None:
        """从MCP工具信息设置工具属性"""
        self.name = tool_info.get("name", self.name)
        self.description = tool_info.get("description", self.description)
        
        # 创建args_schema
        if "inputSchema" in tool_info:
            self.args_schema = create_schema_from_mcp_tool(tool_info)
    
    async def _get_client(self) -> MCPClient:
        """获取MCP客户端"""
        if self.mcp_client is None:
            self.mcp_client = await get_mcp_client(
                transport=self.transport,
                **self.client_kwargs
            )
        return self.mcp_client
    
    def _run(
        self,
        *args,
        **kwargs
    ) -> str:
        """同步运行工具（包装异步调用）"""
        return asyncio.run(self._arun(*args, **kwargs))
    
    async def _arun(
        self,
        *args,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """异步运行工具"""
        try:
            # 获取客户端
            client = await self._get_client()
            
            # 调用工具
            result = await client.call_tool(
                tool_name=self.mcp_tool_name,
                arguments=kwargs,
                session_id=self.session_id
            )
            
            if result.success:
                return str(result.data)
            else:
                return f"工具调用失败: {result.error}"
                
        except Exception as e:
            return f"工具执行错误: {str(e)}"
    
    class Config:
        """Pydantic配置"""
        arbitrary_types_allowed = True


class MCPToolAdapter:
    """MCP工具适配器，管理多个工具"""
    
    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        transport: Union[str, MCPTransport] = MCPTransport.HTTP,
        base_url: Optional[str] = None,
        timeout: int = 30,
        session_id: Optional[str] = None
    ):
        self.transport = transport
        # 默认从环境变量读取，回退到本机 /mcp；不带尾斜杠
        if base_url is None:
            base_url = os.getenv("MCP_BASE_URL", "http://127.0.0.1:9000/mcp")
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session_id = session_id
        
        # 客户端实例
        self._mcp_client = mcp_client
        
        # 工具缓存
        self._tools: Dict[str, MCPLangChainTool] = {}
        self._tool_definitions: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> None:
        """初始化适配器"""
        if self._mcp_client is None:
            from .mcp_client import MCPClientFactory
            self._mcp_client = MCPClientFactory.create_client(
                transport=self.transport,
                base_url=self.base_url,
                server_name="hana-ml-tools",
                timeout=self.timeout
            )
        
        await self._mcp_client.initialize()
        
        # 获取工具列表
        await self._load_tools()
    
    async def _load_tools(self) -> None:
        """从MCP服务器加载工具"""
        if self._mcp_client is None:
            raise RuntimeError("MCP客户端未初始化")
        
        # 获取工具定义
        mcp_tools = await self._mcp_client.list_tools()
        
        for mcp_tool in mcp_tools:
            tool_dict = {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "inputSchema": mcp_tool.inputSchema,
                "metadata": mcp_tool.metadata or {}
            }
            
            self._tool_definitions[mcp_tool.name] = tool_dict
            
            # 创建LangChain工具
            langchain_tool = MCPLangChainTool(
                name=mcp_tool.name,
                description=mcp_tool.description,
                mcp_tool_name=mcp_tool.name,
                mcp_tool_info=tool_dict,
                mcp_client=self._mcp_client,
                transport=self.transport,
                client_kwargs={
                    "base_url": self.base_url,
                    "timeout": self.timeout
                },
                session_id=self.session_id
            )
            
            self._tools[mcp_tool.name] = langchain_tool
    
    def get_tool(self, tool_name: str) -> Optional[MCPLangChainTool]:
        """获取指定工具"""
        return self._tools.get(tool_name)
    
    def get_tools(self) -> List[MCPLangChainTool]:
        """获取所有工具"""
        return list(self._tools.values())
    
    def get_tool_definitions(self) -> Dict[str, Dict[str, Any]]:
        """获取所有工具定义"""
        return self._tool_definitions.copy()
    
    def as_langchain_tools(self) -> List[BaseTool]:
        """转换为LangChain工具列表"""
        return list(self._tools.values())
    
    async def close(self) -> None:
        """关闭适配器"""
        if self._mcp_client:
            await self._mcp_client.close()
            self._mcp_client = None


def create_mcp_tool(
    tool_name: str,
    description: Optional[str] = None,
    mcp_client: Optional[MCPClient] = None,
    **kwargs
) -> MCPLangChainTool:
    """创建单个MCP工具的LangChain包装器"""
    
    # 如果没有提供描述，使用默认值
    if description is None:
        description_map = {
            "set_hana_connection": "Set HANA database connection parameters",
            "discovery_agent": "Use HANA discovery agent to analyze and explore data",
            "data_agent": "Use HANA data agent to query and manipulate data"
        }
        description = description_map.get(tool_name, f"MCP tool: {tool_name}")
    
    return MCPLangChainTool(
        name=tool_name,
        description=description,
        mcp_tool_name=tool_name,
        mcp_client=mcp_client,
        **kwargs
    )


# 预定义的HANA MCP工具
def create_hana_mcp_tools(
    base_url: Optional[str] = None,
    timeout: int = 30,
    session_id: Optional[str] = None
) -> List[BaseTool]:
    """创建HANA MCP工具的LangChain包装器
    
    参数:
        base_url: MCP服务器URL
        timeout: 超时时间（秒）
        session_id: 会话ID（用于多会话场景）
    
    返回:
        List[BaseTool]: LangChain工具列表
    """
    
    # 定义工具配置
    tool_configs = [
        {
            "name": "set_hana_connection",
            "description": "Set HANA database connection parameters. Required before using other tools.",
            "args_schema": create_model("SetHANAConnectionInput", 
                host=Field(..., description="HANA database host"),
                port=Field(..., description="HANA database port"),
                user=Field(..., description="HANA database username"),
                password=Field(..., description="HANA database password")
            )
        },
        {
            "name": "discovery_agent",
            "description": "Use HANA discovery agent to explore and analyze database schema, tables, and data patterns.",
            "args_schema": create_model("DiscoveryAgentInput",
                query=Field(..., description="Natural language query about data discovery")
            )
        },
        {
            "name": "data_agent",
            "description": "Use HANA data agent to execute data queries, transformations, and analysis.",
            "args_schema": create_model("DataAgentInput",
                query=Field(..., description="Natural language query for data operations")
            )
        }
    ]
    
    # 统一 base_url 来源
    if base_url is None:
        base_url = os.getenv("MCP_BASE_URL", "http://127.0.0.1:9000/mcp")
    base_url = base_url.rstrip('/')

    tools = []
    
    for config in tool_configs:
        tool = MCPLangChainTool(
            name=config["name"],
            description=config["description"],
            mcp_tool_name=config["name"],
            args_schema=config.get("args_schema"),
            transport=MCPTransport.HTTP,
            client_kwargs={
                "base_url": base_url,
                "timeout": timeout
            },
            session_id=session_id
        )
        tools.append(tool)
    
    return tools


# 便捷函数
async def get_hana_mcp_tools(
    base_url: Optional[str] = None,
    auto_discover: bool = True
) -> List[BaseTool]:
    """获取HANA MCP工具
    
    参数:
        base_url: MCP服务器URL
        auto_discover: 是否自动从服务器发现工具
    
    返回:
        List[BaseTool]: LangChain工具列表
    """
    
    if base_url is None:
        base_url = os.getenv("MCP_BASE_URL", "http://127.0.0.1:9000/mcp")
    base_url = base_url.rstrip('/')

    if auto_discover:
        # 自动从服务器发现工具
        adapter = MCPToolAdapter(
            transport=MCPTransport.HTTP,
            base_url=base_url
        )
        
        try:
            await adapter.initialize()
            return adapter.as_langchain_tools()
        except Exception as e:
            print(f"自动发现工具失败，使用预定义工具: {e}")
            # 回退到预定义工具
            return create_hana_mcp_tools(base_url=base_url)
    else:
        # 使用预定义工具
        return create_hana_mcp_tools(base_url=base_url)