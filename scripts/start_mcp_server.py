#!/usr/bin/env python3
"""
Start HANA ML MCP server
"""
import asyncio
import sys
import os
from pathlib import Path

 # Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def start_mcp_server(transport: str = "stdio"):
    """Start MCP server"""
    from hana_ai.tools.mcp_tools import mcp
    
    print(f"🚀 Starting HANA ML MCP server ({transport.upper()} transport)...")
    print("📋 Available tools:")
    print("  - set_hana_connection: Set HANA DB connection")
    print("  - discovery_agent: HANA data discovery agent")
    print("  - data_agent: HANA data operation agent")
    print("=" * 50)
    
    # Allow configuring port and host via environment variables
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    try:
        port = int(os.environ.get("MCP_PORT", "8000"))
    except ValueError:
        port = 8000

    if transport == "http":
        await mcp.run_async(
            transport="http",
            host=host,
            port=port,
            path="/mcp",
            json_response=True,
            stateless_http=False,
        )
    elif transport == "sse":
        await mcp.run_async(
            transport="sse",
            host=host,
            port=port,
            path="/mcp",
            json_response=True,
        )
    else:  # stdio
        await mcp.run_async(transport="stdio")


if __name__ == "__main__":
    import asyncio
    try:
        asyncio.get_running_loop().close()
    except Exception:
        pass
    import argparse
    
    parser = argparse.ArgumentParser(description="Start HANA ML MCP server")
    parser.add_argument(
        "transport",
        nargs="?",
        default="stdio",
        choices=["stdio", "http", "sse"],
        help="Transport protocol (stdio/http/sse), default: stdio"
    )
    
    args = parser.parse_args()
    
    try:
        try:
            asyncio.run(start_mcp_server(args.transport))
        except RuntimeError as e:
            if "already running" in str(e).lower():
                if args.transport == "stdio":
                    print("❌ Failed to start server: An asyncio event loop already exists, cannot start in stdio mode. Try http or sse mode, or run in an environment without an event loop.")
                else:
                    print(f"❌ Failed to start server: An asyncio event loop already exists, cannot start in {args.transport} mode. Please run in an environment without an event loop.")
                sys.exit(1)
            else:
                raise
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)