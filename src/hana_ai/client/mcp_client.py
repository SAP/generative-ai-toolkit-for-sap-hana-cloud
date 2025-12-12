"""
MCP client for connecting to HANA ML MCP server
"""
import asyncio
import json
import inspect
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import httpx
from contextlib import asynccontextmanager


class MCPTransport(Enum):
    """MCP transport protocol"""
    HTTP = "http"
    SSE = "sse"
    STDIO = "stdio"


@dataclass
class MCPTool:
    """MCP tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MCPCallResult:
    """MCP call result"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MCPClient:
    """Base MCP client class"""
    
    def __init__(self, server_name: str = "hana-ml-tools"):
        self.server_name = server_name
        self.tools: Dict[str, MCPTool] = {}
        self.session_id: Optional[str] = None
        
    async def initialize(self) -> None:
        """Initialize client"""
        raise NotImplementedError
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPCallResult:
        """Call MCP tool"""
        raise NotImplementedError
        
    async def list_tools(self) -> List[MCPTool]:
        """List all available tools"""
        raise NotImplementedError
        
    async def close(self) -> None:
        """Close client connection"""
        pass


class HTTPMCPClient(MCPClient):
    """MCP client using HTTP transport"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/mcp",
        server_name: str = "hana-ml-tools",
        timeout: int = 30
    ):
        super().__init__(server_name)
        # Normalize base_url to ensure it ends with /mcp and has no trailing slash
        normalized = base_url.rstrip('/')
        if not normalized.endswith('/mcp'):
            normalized = normalized + '/mcp'
        self.base_url = normalized.rstrip('/')
        self.timeout = timeout
        self._client: Optional[aiohttp.ClientSession] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._session_id: Optional[str] = None
        
    async def initialize(self) -> None:
        """初始化HTTP客户端"""
        # aiohttp is only for interface compatibility, httpx is the main client
        if self._client is None:
            self._client = aiohttp.ClientSession(
                base_url=f"{self.base_url.rstrip('/')}/",
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )

        if self._http_client is None:
            default_headers = {
                "accept": "application/json",
                "content-type": "application/json",
                # Session id is assigned by server during initialization
                "mcp-protocol-version": "2024-11-05",
            }
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                trust_env=False,
                follow_redirects=True,
                headers=default_headers,
            )

        # First handshake with server to get assigned session id
        try:
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "hana-ai-client", "version": "0.1"},
                },
            }
            init_resp = await self._http_client.post("", json=init_payload)
            sid = init_resp.headers.get("mcp-session-id")
            if sid:
                self._session_id = sid
                self._http_client.headers["mcp-session-id"] = sid
        except Exception as e:
            # Do not interrupt initialization, later calls will provide clearer errors
            pass

        # Fetch tool list
        await self._refresh_tools()
        
    async def _refresh_tools(self) -> None:
        """Refresh available tool list (via MCP JSON-RPC: tools/list)"""
        try:
            # JSON-RPC request
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                # Also put session in params for server session validation compatibility
                "params": (
                    {"session": {"id": self._session_id}} if getattr(self, "_session_id", None) else {}
                ),
            }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "mcp-protocol-version": "2024-11-05",
            }
            if hasattr(self, "_session_id") and self._session_id:
                headers["mcp-session-id"] = self._session_id
            # Note: POST to /mcp (no trailing slash) to match server route
            response = await self._http_client.post("", json=payload, headers=headers)
            if response.status_code == 200:
                resp = response.json()
                result = resp.get("result") or {}
                tools_data = result.get("tools") or []
                self.tools.clear()

                for tool_data in tools_data:
                    tool = MCPTool(
                        name=tool_data.get("name"),
                        description=tool_data.get("description", ""),
                        inputSchema=tool_data.get("inputSchema", {}),
                        metadata=tool_data.get("metadata", {}),
                    )
                    self.tools[tool.name] = tool
            else:
                print(f"警告: 获取工具列表失败，HTTP {response.status_code}")
        except Exception as e:
            print(f"Warning: Failed to fetch tool list: {e}")
            self._use_default_tools()
    
    def _use_default_tools(self) -> None:
        """Use default tool definitions (for development/testing)"""
        self.tools = {
            "set_hana_connection": MCPTool(
                name="set_hana_connection",
                description="Set HANA connection parameters in the context.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "host": {"type": "string", "description": "The HANA database host."},
                        "port": {"type": "integer", "description": "The HANA database port."},
                        "user": {"type": "string", "description": "The HANA database user."},
                        "password": {"type": "string", "description": "The HANA database password."}
                    },
                    "required": ["host", "port", "user", "password"]
                }
            ),
            "discovery_agent": MCPTool(
                name="discovery_agent",
                description="Use the HANA discovery agent tool to run a query.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query to execute."}
                    },
                    "required": ["query"]
                }
            ),
            "data_agent": MCPTool(
                name="data_agent",
                description="Use the HANA data agent tool to run a query.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query to execute."}
                    },
                    "required": ["query"]
                }
            )
        }
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> MCPCallResult:
        """Call MCP tool"""
        if self._http_client is None:
            await self.initialize()
            
        # If tool list is empty or does not contain the tool, still try direct JSON-RPC call
        # for compatibility with servers that do not expose tool list or client cache is stale.
        
        # Prepare request data
        payload = {
            "arguments": arguments
        }
        
        # Add session ID
        if session_id:
            payload["session"] = {"id": session_id}
        
        try:
            # Use MCP JSON-RPC: tools/call
            # Prepare JSON-RPC payload, include session info to ensure server recognizes session
            effective_session_id = session_id or self._session_id
            rpc_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                    **({"session": {"id": effective_session_id}} if effective_session_id else {}),
                },
            }
            headers = {}
            if effective_session_id:
                headers["mcp-session-id"] = effective_session_id

            # Ensure JSON response
            headers = {
                **headers,
                "accept": "application/json",
                "content-type": "application/json",
                "mcp-protocol-version": "2024-11-05",
            }
            response = await self._http_client.post("", json=rpc_payload, headers=headers)

            if response.status_code != 200:
                return MCPCallResult(
                    success=False,
                    data=None,
                    error=f"HTTP {response.status_code}: {response.text}",
                )

            resp = response.json()
            if "error" in resp:
                # JSON-RPC 层的错误
                err = resp.get("error", {})
                return MCPCallResult(success=False, data=None, error=str(err))

            result_data = resp.get("result", {})
            # Parse MCP tool result, extract text content
            content = result_data.get("content", [])
            if content and isinstance(content, list):
                text_content = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
                data = "\n".join([t for t in text_content if t])
            else:
                data = str(result_data)

            return MCPCallResult(success=True, data=data, metadata={"status_code": response.status_code})

        except Exception as e:
            return MCPCallResult(success=False, data=None, error=f"Tool call failed: {str(e)}")
    
    async def list_tools(self) -> List[MCPTool]:
        """List all available tools"""
        if not self.tools:
            await self._refresh_tools()
        return list(self.tools.values())
    
    async def close(self) -> None:
        """Close client connection"""
        if self._client:
            await self._client.close()
            self._client = None
        
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


class StdioMCPClient(MCPClient):
    """MCP client using Stdio transport (for Claude Desktop, etc.)"""
    
    def __init__(
        self,
        command: str = "python",
        args: List[str] = None,
        server_name: str = "hana-ml-tools"
    ):
        super().__init__(server_name)
        self.command = command
        self.args = args or []
        self._process = None
        
    async def initialize(self) -> None:
        """Initialize Stdio client"""
        # Stdio communication needs to be implemented
        # Only basic structure provided here
        pass
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPCallResult:
        """Call MCP tool (Stdio transport)"""
        # Implement stdio communication logic
        # Usually involves starting subprocess and JSON-RPC communication
        raise NotImplementedError("Stdio客户端待实现")
    
    async def list_tools(self) -> List[MCPTool]:
        """List all available tools"""
        # Implement stdio tool list fetching
        raise NotImplementedError("Stdio客户端待实现")


class MCPClientFactory:
    """MCP client factory"""
    
    @staticmethod
    def create_client(
        transport: Union[str, MCPTransport] = MCPTransport.HTTP,
        **kwargs
    ) -> MCPClient:
        """Create MCP client instance"""
        
        if isinstance(transport, str):
            transport = MCPTransport(transport.lower())
        
        if transport == MCPTransport.HTTP:
            # Default to /mcp path; if not present, auto-append
            base_url = kwargs.get("base_url", "http://localhost:8000/mcp")
            bu = base_url.rstrip('/')
            if not bu.endswith('/mcp'):
                base_url = bu + '/mcp'
            server_name = kwargs.get("server_name", "hana-ml-tools")
            timeout = kwargs.get("timeout", 30)
            
            return HTTPMCPClient(
                base_url=base_url,
                server_name=server_name,
                timeout=timeout
            )
            
        elif transport == MCPTransport.STDIO:
            command = kwargs.get("command", "python")
            args = kwargs.get("args", [])
            server_name = kwargs.get("server_name", "hana-ml-tools")
            
            return StdioMCPClient(
                command=command,
                args=args,
                server_name=server_name
            )
            
        else:
            raise ValueError(f"不支持的传输协议: {transport}")


 # Convenient global client instance
_global_client: Optional[MCPClient] = None


async def get_mcp_client(
    transport: Union[str, MCPTransport] = MCPTransport.HTTP,
    **kwargs
) -> MCPClient:
    """Get MCP client (singleton)"""
    global _global_client
    if _global_client is None:
        _global_client = MCPClientFactory.create_client(transport, **kwargs)
        await _global_client.initialize()
    return _global_client


async def call_mcp_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    transport: Union[str, MCPTransport] = MCPTransport.HTTP,
    session_id: Optional[str] = None,
    **client_kwargs
) -> MCPCallResult:
    """Convenience function: call MCP tool"""
    client = await get_mcp_client(transport, **client_kwargs)
    return await client.call_tool(tool_name, arguments, session_id=session_id)

@asynccontextmanager
async def mcp_client_context(
    transport: Union[str, MCPTransport] = MCPTransport.HTTP,
    **kwargs
):
    """Async context manager for MCP client"""
    client = MCPClientFactory.create_client(transport, **kwargs)
    await client.initialize()
    try:
        yield client
    finally:
        await client.close()