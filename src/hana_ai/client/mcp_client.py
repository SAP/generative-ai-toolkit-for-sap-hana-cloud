"""
MCP client for connecting to HANA ML MCP server
"""

# pylint: disable=global-statement

import asyncio
import json
import ipaddress
import os
import shutil
from urllib.parse import urlsplit
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Union

import httpx



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
        timeout: int = 30,
        extra_headers: Optional[Dict[str, str]] = None,
        allow_insecure_http: bool = False,
    ):
        super().__init__(server_name)
        # Normalize base_url to ensure it ends with /mcp and has no trailing slash
        normalized = _validate_http_base_url(
            base_url,
            allow_insecure_http=allow_insecure_http,
        ).rstrip('/')
        if not normalized.endswith('/mcp'):
            normalized = normalized + '/mcp'
        self.base_url = normalized.rstrip('/')
        self.timeout = timeout
        self._extra_headers = {str(key): str(value) for key, value in (extra_headers or {}).items()}
        self._http_client: Optional[httpx.AsyncClient] = None
        self._session_id: Optional[str] = None

    async def initialize(self) -> None:
        """初始化HTTP客户端"""
        if self._http_client is None:
            default_headers = {
                "accept": "application/json",
                "content-type": "application/json",
                # Session id is assigned by server during initialization
                "mcp-protocol-version": "2024-11-05",
            }
            default_headers.update(self._extra_headers)
            self._http_client = httpx.AsyncClient(
                timeout=self.timeout,
                trust_env=False,
                # Do not auto-follow redirects — a malicious/MitM'd server could 307 the
                # client to an attacker origin and leak the mcp-session-id (and same-origin, the
                # Authorization header). Handle redirects explicitly if ever needed.
                follow_redirects=False,
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
            init_resp = await self._http_client.post(self.base_url, json=init_payload)
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
            headers.update(self._extra_headers)
            if hasattr(self, "_session_id") and self._session_id:
                headers["mcp-session-id"] = self._session_id
            # Use the exact configured URL so redirect handling remains disabled.
            response = await self._http_client.post(self.base_url, json=payload, headers=headers)
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
            "admin_update_connection_context": MCPTool(
                name="admin_update_connection_context",
                description="Update the toolkit's HANA ConnectionContext without restarting the MCP server.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "address": {"type": "string", "description": "The HANA database host/address."},
                        "port": {"type": "integer", "description": "The HANA database port."},
                        "user": {"type": "string", "description": "The HANA database user."},
                        "password": {"type": "string", "description": "The HANA database password."},
                        "encrypt": {"type": "boolean", "description": "Use TLS."},
                        "ssl_validate_certificate": {"type": "boolean", "description": "Validate TLS certificate."},
                        "test_connection": {"type": "boolean", "description": "Test new connection before switching."}
                    },
                    "required": ["address", "port", "user", "password"]
                }
            ),
            "object_discovery": MCPTool(
                name="object_discovery",
                description="Use the HANA object discovery tool to run a query.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query to execute."}
                    },
                    "required": ["query"]
                }
            ),
            "data_retrieval": MCPTool(
                name="data_retrieval",
                description="Use the HANA data retrieval tool to run a query.",
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
            headers.update(self._extra_headers)
            if effective_session_id:
                headers["mcp-session-id"] = effective_session_id

            # Ensure JSON response
            headers = {
                **headers,
                "accept": "application/json",
                "content-type": "application/json",
                "mcp-protocol-version": "2024-11-05",
            }
            response = await self._http_client.post(
                self.base_url, json=rpc_payload, headers=headers
            )

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
            return MCPCallResult(
                success=False, data=None,
                error=f"Tool call failed ({type(e).__name__}): {e}",
            )

    async def list_tools(self) -> List[MCPTool]:
        """List all available tools"""
        if not self.tools:
            await self._refresh_tools()
        return list(self.tools.values())

    async def close(self) -> None:
        """Close client connection"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


def _validate_http_base_url(base_url: str, *, allow_insecure_http: bool = False) -> str:
    """Require TLS outside loopback while preserving the bundled local HTTP flow."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("MCP base_url must be a non-empty absolute URL")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("MCP base_url must use http:// or https:// with an explicit host")
    if parsed.username or parsed.password:
        raise ValueError("MCP credentials must be supplied as headers, not embedded in base_url")
    host = parsed.hostname.lower().rstrip(".")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if parsed.scheme == "http" and not loopback and allow_insecure_http is not True:
        raise ValueError(
            "plaintext MCP HTTP is allowed only on loopback; use https:// or set "
            "allow_insecure_http=True explicitly for a trusted isolated network"
        )
    return base_url


_STDIO_INLINE_CODE_FLAGS = {
    "-c", "-e", "--eval", "--evaluate", "-command", "-encodedcommand",
}
_STDIO_SHELL_NAMES = {
    "sh", "bash", "dash", "ash", "zsh", "ksh", "csh", "tcsh", "fish",
    "cmd", "command", "powershell", "pwsh",
}


def _is_stdio_inline_code_arg(arg: str) -> bool:
    """Distinguish /c shell switches from /code/server.py-style paths."""
    lowered = arg.lower()
    if any(lowered.startswith(flag) for flag in _STDIO_INLINE_CODE_FLAGS):
        return True
    if lowered.startswith("/c"):
        # Keep /c, /cCOMMAND and /c COMMAND blocked. A multi-component POSIX
        # path is not a cmd switch for a non-shell executable. Exact host
        # profile matching still applies; shell executables are always denied.
        return "/" not in arg[2:] or arg[2].isspace() or arg[2] == "\\"
    return False


def _reject_stdio_shell(command: str) -> None:
    """Reject shell interpreters and implicitly shell-executed Windows scripts."""
    name = command.replace("\\", "/").rsplit("/", 1)[-1].lower()
    stem, extension = os.path.splitext(name)
    if name in _STDIO_SHELL_NAMES or (
        extension in {".exe", ".com"} and stem in _STDIO_SHELL_NAMES
    ) or extension in {".bat", ".cmd", ".ps1"}:
        raise ValueError("shell interpreters and batch scripts are not allowed for stdio")


def _resolve_stdio_executable(command: str) -> str:
    resolved = command if os.path.isabs(command) else shutil.which(command)
    if not resolved or not os.path.isfile(resolved):
        raise ValueError(f"stdio command not found: {command!r}")
    return os.path.realpath(resolved)


def _load_stdio_profiles() -> List[Dict[str, Any]]:
    """Load host-controlled executable and argument prefixes for stdio servers."""
    raw = os.environ.get("HANA_MCP_STDIO_ALLOWED_PROFILES", "")
    if not raw:
        raise ValueError(
            "stdio process spawning is disabled; set HANA_MCP_STDIO_ALLOWED_PROFILES "
            "to a JSON list of trusted command/args_prefix profiles"
        )
    try:
        profiles = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("HANA_MCP_STDIO_ALLOWED_PROFILES must be valid JSON") from exc
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("HANA_MCP_STDIO_ALLOWED_PROFILES must be a non-empty JSON list")
    return profiles


def _validate_stdio_command(command: str, args: List[str]) -> str:
    """Match an untrusted stdio command against a host-controlled server profile.

    An executable-name allowlist is insufficient because a Python or package runner can
    still execute attacker-selected code. A profile therefore fixes the resolved executable
    and a non-empty argument list. Additional operational arguments are rejected unless the
    host profile explicitly sets ``allow_additional_args: true``.
    """
    if not command or not isinstance(command, str):
        raise ValueError("stdio command must be a non-empty string")
    _reject_stdio_shell(command)
    args = list(args or [])
    if not all(isinstance(arg, str) and "\x00" not in arg for arg in args):
        raise ValueError("stdio args must be strings without NUL bytes")
    if any(_is_stdio_inline_code_arg(arg) for arg in args):
        raise ValueError("inline code or shell execution flags are not allowed in stdio args")

    resolved = _resolve_stdio_executable(command)
    _reject_stdio_shell(resolved)
    normalized = os.path.normcase(resolved)
    for profile in _load_stdio_profiles():
        if not isinstance(profile, dict):
            continue
        allowed_command = profile.get("command")
        prefix = profile.get("args_prefix")
        if not isinstance(allowed_command, str) or not isinstance(prefix, list) or not prefix:
            continue
        if not all(isinstance(arg, str) for arg in prefix):
            continue
        try:
            allowed_resolved = os.path.normcase(_resolve_stdio_executable(allowed_command))
        except ValueError:
            continue
        allow_additional_args = profile.get("allow_additional_args") is True
        args_match = args == prefix or (
            allow_additional_args and args[:len(prefix)] == prefix
        )
        if normalized == allowed_resolved and args_match:
            return resolved
    raise ValueError(
        "stdio command and args do not match any trusted "
        "HANA_MCP_STDIO_ALLOWED_PROFILES entry"
    )


class StdioMCPClient(MCPClient):
    """MCP client using Stdio transport (for Claude Desktop, etc.)"""

    def __init__(
        self,
        command: str = "python",
        args: List[str] = None,
        server_name: str = "hana-ml-tools",
        timeout: float = 30,
    ):
        super().__init__(server_name)
        self.command = command
        self.args = args or []
        self.timeout = timeout
        self._process = None
        self._next_id = 1
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._session_id: Optional[str] = None
        self._stderr_tail: List[str] = []

    async def _ensure_process(self) -> None:
        if self._process is not None:
            return
        # The stdio command/args come from untrusted config; validate and
        # resolve to an absolute path before spawning (no shells, no inline code,
        # allowlisted binaries only).
        resolved_command = _validate_stdio_command(self.command, self.args)
        self._process = await asyncio.create_subprocess_exec(
            resolved_command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # An all-tools MCP schema response can exceed asyncio's 64 KiB
            # default line limit because stdio JSON-RPC uses one line/message.
            limit=10 * 1024 * 1024,
        )

        async def _read_stdout():
            assert self._process is not None
            assert self._process.stdout is not None
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8", errors="ignore").strip())
                except Exception:
                    continue
                if isinstance(msg, dict) and "id" in msg:
                    mid = msg.get("id")
                    fut = self._pending.pop(mid, None)
                    if fut is not None and not fut.done():
                        fut.set_result(msg)

        self._reader_task = asyncio.create_task(_read_stdout())

        async def _read_stderr():
            assert self._process is not None
            assert self._process.stderr is not None
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                txt = line.decode("utf-8", errors="ignore").rstrip()
                if txt:
                    self._stderr_tail.append(txt)
                    # keep last 200 lines
                    if len(self._stderr_tail) > 200:
                        self._stderr_tail = self._stderr_tail[-200:]

        self._stderr_task = asyncio.create_task(_read_stderr())

    async def _rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        await self._ensure_process()
        assert self._process is not None
        assert self._process.stdin is not None

        rpc_id = self._next_id
        self._next_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[rpc_id] = fut

        data = (json.dumps(payload) + "\n").encode("utf-8")
        self._process.stdin.write(data)
        await self._process.stdin.drain()

        try:
            resp = await asyncio.wait_for(fut, timeout=self.timeout)
            return resp
        except asyncio.TimeoutError as e:
            tail = "\n".join(self._stderr_tail[-50:])
            raise asyncio.TimeoutError(f"STDIO RPC timeout for method={method}. Server stderr tail:\n{tail}") from e

    async def _notify(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send an MCP notification, which intentionally has no response id."""
        await self._ensure_process()
        assert self._process is not None
        assert self._process.stdin is not None
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await self._process.stdin.drain()

    async def initialize(self) -> None:
        """Initialize Stdio client"""
        resp = await self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "hana-ai-stdio-client", "version": "0.1"},
            },
        )
        if "error" in resp:
            raise RuntimeError(str(resp.get("error")))
        result = resp.get("result") or {}
        # some servers may return session info
        sid = None
        try:
            sid = (result.get("session") or {}).get("id")
        except Exception:
            sid = None
        self._session_id = sid
        await self._notify("notifications/initialized", {})
        await self._refresh_tools()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPCallResult:
        """Call MCP tool (Stdio transport)"""
        try:
            params: Dict[str, Any] = {
                "name": tool_name,
                "arguments": arguments or {},
            }
            if self._session_id:
                params["session"] = {"id": self._session_id}

            resp = await self._rpc("tools/call", params)
            if "error" in resp:
                return MCPCallResult(success=False, data=None, error=str(resp.get("error")))

            result_data = resp.get("result", {})
            content = result_data.get("content", [])
            if content and isinstance(content, list):
                text_content = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                data = "\n".join([t for t in text_content if t])
            else:
                data = result_data
            return MCPCallResult(success=True, data=data)
        except Exception as e:
            return MCPCallResult(
                success=False, data=None,
                error=f"Tool call failed ({type(e).__name__}): {e}",
            )

    async def list_tools(self) -> List[MCPTool]:
        """List all available tools"""
        if not self.tools:
            await self._refresh_tools()
        return list(self.tools.values())

    async def _refresh_tools(self) -> None:
        resp = await self._rpc(
            "tools/list",
            ({"session": {"id": self._session_id}} if self._session_id else {}),
        )
        if "error" in resp:
            # fallback: keep defaults
            self._use_default_tools_stdio()
            return

        result = resp.get("result") or {}
        tools_data = result.get("tools") or []
        self.tools.clear()
        for tool_data in tools_data:
            tool = MCPTool(
                name=tool_data.get("name"),
                description=tool_data.get("description", ""),
                inputSchema=tool_data.get("inputSchema", {}) or {},
                metadata=tool_data.get("metadata", {}),
            )
            self.tools[tool.name] = tool

    def _use_default_tools_stdio(self) -> None:
        """Use default tool definitions (same as HTTP fallback).

        This wrapper exists to satisfy static analyzers (pylint) and to avoid
        depending on private base-class behavior.
        """
        # Keep this local to avoid relying on methods that exist only on HTTPMCPClient.
        self.tools = {
            "admin_update_connection_context": MCPTool(
                name="admin_update_connection_context",
                description="Update the toolkit's HANA ConnectionContext without restarting the MCP server.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "address": {"type": "string", "description": "The HANA database host/address."},
                        "port": {"type": "integer", "description": "The HANA database port."},
                        "user": {"type": "string", "description": "The HANA database user."},
                        "password": {"type": "string", "description": "The HANA database password."},
                        "encrypt": {"type": "boolean", "description": "Use TLS."},
                        "ssl_validate_certificate": {"type": "boolean", "description": "Validate TLS certificate."},
                        "test_connection": {"type": "boolean", "description": "Test new connection before switching."},
                    },
                    "required": ["address", "port", "user", "password"],
                },
            ),
            "object_discovery": MCPTool(
                name="object_discovery",
                description="Use the HANA object discovery tool to run a query.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query to execute."}
                    },
                    "required": ["query"],
                },
            ),
            "data_retrieval": MCPTool(
                name="data_retrieval",
                description="Use the HANA data retrieval tool to run a query.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query to execute."}
                    },
                    "required": ["query"],
                },
            ),
        }

    async def close(self) -> None:
        if self._process is not None:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except Exception:
                pass
            self._process = None

        if self._reader_task is not None:
            try:
                self._reader_task.cancel()
            except Exception:
                pass
            self._reader_task = None

        if self._stderr_task is not None:
            try:
                self._stderr_task.cancel()
            except Exception:
                pass
            self._stderr_task = None


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
                timeout=timeout,
                extra_headers=kwargs.get("extra_headers"),
                allow_insecure_http=kwargs.get("allow_insecure_http", False),
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
