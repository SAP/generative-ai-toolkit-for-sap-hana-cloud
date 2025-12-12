#!/usr/bin/env python3
"""
使用MCP工具的示例
"""
import asyncio
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入工具
from hana_ai.client.mcp_client import MCPClientFactory, MCPTransport, get_mcp_client, mcp_client_context
from hana_ai.client.langchain_adapter import (
    MCPToolAdapter,
    get_hana_mcp_tools,
    create_hana_mcp_tools
)


async def example_direct_mcp_client():
    """示例1: 直接使用MCP客户端"""
    print("=" * 60)
    print("示例1: 直接使用MCP客户端")
    print("=" * 60)
    
    base_url = os.getenv("MCP_BASE_URL", "http://127.0.0.1:9000/mcp")
    async with mcp_client_context(
        transport=MCPTransport.HTTP,
        base_url=base_url
    ) as client:
        # 列出工具
        tools = await client.list_tools()
        print(f"可用工具: {[tool.name for tool in tools]}")
        
        # 设置HANA连接
        result = await client.call_tool("set_hana_connection", {
            "host": os.getenv("HANA_HOST", "localhost"),
            "port": int(os.getenv("HANA_PORT", 443)),
            "user": os.getenv("HANA_USER", "SYSTEM"),
            "password": os.getenv("HANA_PASSWORD", "")
        })
        
        if result.success:
            print(f"✅ 连接设置: {result.data}")
            
            # 使用discovery_agent
            discovery_result = await client.call_tool("discovery_agent", {
                "query": "Show me all tables in the current schema"
            })
            
            if discovery_result.success:
                print(f"🔍 发现结果: {discovery_result.data[:200]}...")
            else:
                print(f"❌ 发现失败: {discovery_result.error}")
                
        else:
            print(f"❌ 连接设置失败: {result.error}")


async def example_langchain_adapter():
    """示例2: 使用LangChain适配器"""
    print("\n" + "=" * 60)
    print("示例2: 使用LangChain适配器")
    print("=" * 60)
    
    # 创建适配器
    base_url = os.getenv("MCP_BASE_URL", "http://127.0.0.1:9000/mcp")
    adapter = MCPToolAdapter(
        transport=MCPTransport.HTTP,
        base_url=base_url
    )
    
    try:
        # 初始化适配器
        await adapter.initialize()
        
        # 获取所有工具
        tools = adapter.get_tools()
        print(f"加载了 {len(tools)} 个工具:")
        
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:50]}...")
        
        # 获取特定工具
        set_conn_tool = adapter.get_tool("set_hana_connection")
        if set_conn_tool:
            # 设置连接
            result = await set_conn_tool.arun(
                host=os.getenv("HANA_HOST", "localhost"),
                port=int(os.getenv("HANA_PORT", 443)),
                user=os.getenv("HANA_USER", "SYSTEM"),
                password=os.getenv("HANA_PASSWORD", "")
            )
            print(f"\n🔧 设置连接结果: {result}")
        
        # 使用discovery_agent
        discovery_tool = adapter.get_tool("discovery_agent")
        if discovery_tool:
            result = await discovery_tool.arun(
                query="List all tables and their row counts"
            )
            print(f"\n🔍 数据发现结果: {result[:200]}...")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
    finally:
        await adapter.close()


async def example_with_langchain_agent():
    """示例3: 在LangChain Agent中使用MCP工具"""
    print("\n" + "=" * 60)
    print("示例3: 在LangChain Agent中使用MCP工具")
    print("=" * 60)
    
    try:
        from langchain.agents import AgentExecutor, create_openai_tools_agent
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.agents import Tool
        
        # 获取MCP工具
        base_url = os.getenv("MCP_BASE_URL", "http://127.0.0.1:9000/mcp")
        mcp_tools = await get_hana_mcp_tools(
            base_url=base_url,
            auto_discover=True
        )
        
        # 转换为LangChain Tool格式
        langchain_tools = []
        for mcp_tool in mcp_tools:
            tool = Tool(
                name=mcp_tool.name,
                description=mcp_tool.description,
                func=mcp_tool._run,  # 使用同步方法
                args_schema=mcp_tool.args_schema
            )
            langchain_tools.append(tool)
        
        print(f"创建了 {len(langchain_tools)} 个LangChain工具")
        
        # 创建LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个HANA数据库专家，可以使用以下工具：
            
            可用工具：
            {tools}
            
            使用流程：
            1. 首先使用set_hana_connection设置数据库连接
            2. 然后使用discovery_agent探索数据库结构
            3. 最后使用data_agent查询具体数据
            
            请严格按照工具的参数要求调用工具。"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # 创建Agent
        agent = create_openai_tools_agent(llm, langchain_tools, prompt)
        
        # 创建执行器
        agent_executor = AgentExecutor(
            agent=agent,
            tools=langchain_tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # 运行Agent
        print("\n🤖 启动HANA数据库助手...")
        print("输入 'quit' 退出")
        print("-" * 40)
        
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if not user_input:
                    continue
                
                # 执行Agent
                result = await agent_executor.ainvoke({
                    "input": user_input,
                    "chat_history": []
                })
                
                print(f"\n🤖 助手: {result['output']}")
                
            except KeyboardInterrupt:
                print("\n\n⏹️ 中断")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
                
    except ImportError as e:
        print(f"❌ 需要安装LangChain: pip install langchain langchain-openai")
        print(f"错误详情: {e}")
    except Exception as e:
        print(f"❌ 创建Agent失败: {e}")


async def example_simple_chatbot():
    """示例4: 简化版聊天机器人（不依赖LangChain）"""
    print("\n" + "=" * 60)
    print("示例4: 简化版HANA聊天机器人")
    print("=" * 60)
    
    from hana_ai.client.mcp_client import call_mcp_tool
    
    print("HANA数据库聊天助手")
    print("可用命令:")
    print("  1. connect [host] [port] [user] [password] - 连接数据库")
    print("  2. discover [query] - 探索数据库")
    print("  3. query [query] - 查询数据")
    print("  4. help - 显示帮助")
    print("  5. exit - 退出")
    print("  6. debug - 显示当前会话状态")
    print("-" * 40)
    
    # 连接信息（会话由客户端在 initialize 时与服务器协商）
    connection_info = None
    base_url = os.getenv("MCP_BASE_URL", "http://127.0.0.1:9000/mcp")
    
    while True:
        try:
            user_input = input("\n🔧 输入命令: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', '退出']:
                print("👋 再见！")
                break
                
            if user_input.lower() == 'help':
                print("""
命令格式:
  connect [host] [port] [user] [password]  - 连接HANA数据库
  discover [问题]                          - 探索数据库结构和数据
  query [问题]                             - 查询和操作数据
  help                                     - 显示此帮助
  exit                                     - 退出程序
                """)
                continue
            
            parts = user_input.split(' ', 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if command == "connect":
                # 解析连接参数
                connect_parts = args.split()
                if len(connect_parts) < 4:
                    print("❌ 格式: connect [host] [port] [user] [password]")
                    continue
                    
                host, port_str, user, password = connect_parts[:4]
                
                try:
                    port = int(port_str)
                except ValueError:
                    print(f"❌ 端口必须是数字: {port_str}")
                    continue
                
                # 调用MCP工具设置连接
                result = await call_mcp_tool(
                    "set_hana_connection",
                    {
                        "host": host,
                        "port": port,
                        "user": user,
                        "password": password
                    },
                    transport=MCPTransport.HTTP,
                    base_url=base_url,
                )
                
                if result.success:
                    print(f"✅ {result.data}")
                    connection_info = {"host": host, "user": user}
                else:
                    print(f"❌ 连接失败: {result.error}")
                    
            elif command == "debug":
                # 调用调试工具，查看服务端会话状态
                result = await call_mcp_tool(
                    "debug_session",
                    {},
                    transport=MCPTransport.HTTP,
                    base_url=base_url,
                )
                if result.success:
                    print(f"🧭 会话状态:\n{result.data}")
                else:
                    print(f"❌ 调试失败: {result.error}")

            elif command == "discover":
                if not connection_info:
                    print("❌ 请先使用 'connect' 命令连接数据库")
                    continue
                    
                if not args:
                    print("❌ 请输入探索问题，例如: discover 显示所有表")
                    continue
                
                result = await call_mcp_tool(
                    "discovery_agent",
                    {"query": args},
                    transport=MCPTransport.HTTP,
                    base_url=base_url,
                )
                
                if result.success:
                    print(f"🔍 发现结果:\n{result.data}")
                else:
                    print(f"❌ 探索失败: {result.error}")
                    
            elif command == "query":
                if not connection_info:
                    print("❌ 请先使用 'connect' 命令连接数据库")
                    continue
                    
                if not args:
                    print("❌ 请输入查询问题，例如: query 查询用户表前10行")
                    continue
                
                result = await call_mcp_tool(
                    "data_agent",
                    {"query": args},
                    transport=MCPTransport.HTTP,
                    base_url=base_url,
                )
                
                if result.success:
                    print(f"📊 查询结果:\n{result.data}")
                else:
                    print(f"❌ 查询失败: {result.error}")
                    
            else:
                print(f"❌ 未知命令: {command}")
                print("输入 'help' 查看可用命令")
                
        except KeyboardInterrupt:
            print("\n\n⏹️ 中断")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


async def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
    else:
        print("请选择示例:")
        print("  1. 直接使用MCP客户端")
        print("  2. 使用LangChain适配器")
        print("  3. 在LangChain Agent中使用")
        print("  4. 简化版聊天机器人")
        example = input("选择 (1-4): ").strip()
    
    if example == "1":
        await example_direct_mcp_client()
    elif example == "2":
        await example_langchain_adapter()
    elif example == "3":
        await example_with_langchain_agent()
    elif example == "4":
        await example_simple_chatbot()
    else:
        print("无效选择")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())