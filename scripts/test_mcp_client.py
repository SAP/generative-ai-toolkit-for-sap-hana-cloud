#!/usr/bin/env python3
"""
Test MCP client
"""
import asyncio
import sys
from pathlib import Path

 # Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_mcp_client():
    """Test MCP client"""
    from hana_ai.client.mcp_client import MCPClientFactory, MCPTransport
    
    print("🧪 Testing MCP client connection...")
    
    # 创建客户端
    import os
    base_url = os.getenv("MCP_BASE_URL", "http://127.0.0.1:9000/mcp")
    client = MCPClientFactory.create_client(
        transport=MCPTransport.HTTP,
        base_url=base_url,
        server_name="hana-ml-tools",
        timeout=10
    )
    
    try:
        # Initialize
        await client.initialize()
        print("✅ MCP客户端初始化成功")
        
        # List tools
        tools = await client.list_tools()
        print(f"📋 发现 {len(tools)} 个工具:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:50]}...")
        
        # Test tool call
        print("\n🔧 测试工具调用...")
        
        # Set connection
        result = await client.call_tool("set_hana_connection", {
            "host": "localhost",
            "port": 30015,
            "user": "SYSTEM",
            "password": "YourPassword123"
        })
        
        if result.success:
            print(f"✅ 连接设置: {result.data}")
        else:
            print(f"⚠️  连接测试失败: {result.error}")
            print("(这是预期的，因为我们没有真正的HANA服务器)")
        
        # Test discovery_agent
        result = await client.call_tool("discovery_agent", {
            "query": "test connection"
        })
        
        if result.success:
            print(f"✅ Discovery Agent: {result.data[:100]}...")
        else:
            print(f"⚠️  Discovery Agent失败: {result.error}")
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()


async def test_langchain_adapter():
    """Test LangChain adapter"""
    from hana_ai.client.langchain_adapter import MCPToolAdapter
    
    print("\n🔌 Testing LangChain adapter...")
    
    import os
    base_url = os.getenv("MCP_BASE_URL", "http://127.0.0.1:9000/mcp")
    adapter = MCPToolAdapter(
        transport="http",
        base_url=base_url
    )
    
    try:
        await adapter.initialize()
        print("✅ LangChain adapter initialized successfully")
        
        tools = adapter.get_tools()
        print(f"📦 Loaded {len(tools)} LangChain tools:")
        
        for tool in tools:
            print(f"  - {tool.name}")
            if hasattr(tool, 'args_schema'):
                print(f"    参数: {list(tool.args_schema.__fields__.keys())}")
        
        # Get tool definitions
        tool_defs = adapter.get_tool_definitions()
        print(f"\n📄 Tool definitions: {list(tool_defs.keys())}")
        
    except Exception as e:
        print(f"❌ Adapter test failed: {e}")
    finally:
        await adapter.close()


async def main():
    """Main test function"""
    print("=" * 60)
    print("HANA MCP客户端测试套件")
    print("=" * 60)
    
    # Make sure MCP server is running first
    print("⚠️  Please make sure MCP server is running: python scripts/start_mcp_server.py http")
    input("Press Enter to continue...")
    
    await test_mcp_client()
    await test_langchain_adapter()


if __name__ == "__main__":
    asyncio.run(main())