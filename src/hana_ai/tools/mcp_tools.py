"""
hana_ai.tools.mcp_tools
"""

import asyncio
import json
from typing import Annotated
from fastmcp import FastMCP, Context
from hana_ml import ConnectionContext
from hana_ai.tools.hana_ml_tools.graph_tools import DiscoveryAgentTool, DataAgentTool


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
@mcp.tool(
        name="discovery_agent",
        description="Tool for discovering HANA objects via knowledge graph."
)
async def discovery_agent(query: Annotated[str, "The query to discover HANA objects via knowledge graph."],
                          context: Context,
                          remote_source_name: Annotated[str, "The name of the remote source to connect to AI Core. Default is 'HANA_DISCOVERY_AGENT_CREDENTIALS'."] = "HANA_DISCOVERY_AGENT_CREDENTIALS",
                          rag_schema_name: Annotated[str, "The schema name where RAG tables are stored. Default is 'SYSTEM'."] = "SYSTEM",
                          rag_table_name: Annotated[str, "The table name where RAG data is stored. Default is 'RAG'."] = "RAG",
                          graph_name: Annotated[str, "he name of the knowledge graph to use. Default is 'HANA_OBJECTS'."] = "HANA_OBJECTS",
                          ) -> str:
    """
    Use the HANA discovery agent tool to run a query.
    """
    tool = get_discovery_agent_tool(context)
    additional_config = {
        "remoteSourceName": remote_source_name,
        "ragSchemaName": rag_schema_name,
        "ragTableName": rag_table_name,
        "graphName": graph_name
    }
    result = await asyncio.to_thread(tool._run, query, additional_config)

    return result

@mcp.tool(
        name="data_agent",
        description="Tool for querying HANA data via knowledge graph."
)
async def data_agent(query: Annotated[str, "The query to discover HANA objects via knowledge graph."],
                     context: Context,
                     remote_source_name: Annotated[str, "The name of the remote source to connect to AI Core. Default is 'HANA_DISCOVERY_AGENT_CREDENTIALS'."] = "HANA_DISCOVERY_AGENT_CREDENTIALS",
                     rag_schema_name: Annotated[str, "The schema name where RAG tables are stored. Default is 'SYSTEM'."] = "SYSTEM",
                     rag_table_name: Annotated[str, "The table name where RAG data is stored. Default is 'RAG'."] = "RAG",
                     graph_name: Annotated[str, "he name of the knowledge graph to use. Default is 'HANA_OBJECTS'."] = "HANA_OBJECTS",) -> str:
    """
    Use the HANA data agent tool to run a query.
    """
    tool = get_data_agent_tool(context)
    additional_config = {
        "remoteSourceName": remote_source_name,
        "ragSchemaName": rag_schema_name,
        "ragTableName": rag_table_name,
        "graphName": graph_name
    }
    result = await asyncio.to_thread(tool._run, query, additional_config)

    return result

 # Debug tool: view current session's connection info and created tools
@mcp.tool()
async def debug_session(context: Context) -> str:
    """
    View current session's HANA connection info and created tools.
    """
    info = {
        "hana_connection": context.get_state("hana_connection"),
        "has_discovery_agent": context.get_state("discovery_agent") is not None,
        "has_data_agent": context.get_state("data_agent") is not None,
    }
    return json.dumps(info, ensure_ascii=False)
