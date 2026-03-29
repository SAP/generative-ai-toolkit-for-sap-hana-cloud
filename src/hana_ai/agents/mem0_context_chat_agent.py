"""Mem0 context-engineered chat agent.

This module is a thin wrapper around the implementation in hana_ai.iagents, so existing
examples/notebooks that import agents from hana_ai.agents can switch with minimal changes.

See: nutest/testscripts/demo/e2e_scenarios/mem0_agent.ipynb

Usage (mirrors Mem0HANARAGAgent):

- llm = init_llm(...)
- tools = HANAMLToolkit(cc, used_tools='all').get_tools()
- chatbot = Mem0ContextChatAgent(llm=llm, tools=tools, verbose=True)
- chatbot.clear_long_term_memory()
- chatbot.chat("...")
"""

from hana_ai.iagents.mem0_context_chat_agent import Mem0ContextChatAgent

__all__ = ["Mem0ContextChatAgent"]
