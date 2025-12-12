"""
hana_ai.tools.mcp_tools
"""

import asyncio

from hana_ml import ConnectionContext
from fastmcp import FastMCP, Context

from hana_ai.tools.hana_ml_tools.graph_tools import DiscoveryAgentTool, DataAgentTool
import json

mcp = FastMCP("HANA ML Tools")

@mcp.tool()
async def set_hana_connection(
    host: str,
    port: int,
    user: str,
    password: str,
    context: Context
) -> str:
    """
    Set HANA connection parameters in the context.

    Parameters
    ----------
    host : str
        The HANA database host.
    port : int
        The HANA database port.
    user : str
        The HANA database user.
    password : str
        The HANA database password.
    """
    connection_context = {
        "host": host,
        "port": port,
        "user": user,
        "password": password
    }

    context.set_state("hana_connection", connection_context)
    return "HANA connection set successfully."

def get_discovery_agent_tool(context: Context):
    """Get or create the HANA discovery agent tool instance for the current session"""
    if context.get_state("discovery_agent") is None:
        # Get connection info from context (must be set by another tool first)
        conn_info = context.get_state("hana_connection")
        if not conn_info:
            raise ValueError("Please set HANA connection first.")

        # Create tool instance
        connection_context = ConnectionContext(
            host=conn_info["host"],
            port=conn_info["port"],
            user=conn_info["user"],
            password=conn_info["password"]
        )
        context.set_state("discovery_agent", DiscoveryAgentTool(
            connection_context=connection_context))
    return context.get_state("discovery_agent")

def get_data_agent_tool(context: Context):
    """Get or create the HANA data agent tool instance for the current session"""
    if context.get_state("data_agent") is None:
        # 从context中获取连接信息（需要先通过其他tool设置）
        conn_info = context.get_state("hana_connection")
        if not conn_info:
            raise ValueError("Please set HANA connection first.")

        # 创建工具实例
        connection_context = ConnectionContext(
            host=conn_info["host"],
            port=conn_info["port"],
            user=conn_info["user"],
            password=conn_info["password"]
        )
        context.set_state("data_agent", DataAgentTool(
            connection_context=connection_context))
    return context.get_state("data_agent")

 # Wrap _run method as FastMCP tool
@mcp.tool()
async def discovery_agent(query: str, context: Context) -> str:
    """
    Use the HANA discovery agent tool to run a query.
    """
    tool = get_discovery_agent_tool(context)

    result = await asyncio.to_thread(tool._run, query)

    return result

@mcp.tool()
async def data_agent(query: str, context: Context) -> str:
    """
    Use the HANA data agent tool to run a query.
    """
    tool = get_data_agent_tool(context)

    result = await asyncio.to_thread(tool._run, query)

    return result

 # Debug tool: view current session's connection info and created tools
@mcp.tool()
async def debug_session(context: Context) -> str:
    info = {
        "hana_connection": context.get_state("hana_connection"),
        "has_discovery_agent": context.get_state("discovery_agent") is not None,
        "has_data_agent": context.get_state("data_agent") is not None,
    }
    return json.dumps(info, ensure_ascii=False)
