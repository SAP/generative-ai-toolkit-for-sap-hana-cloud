from calendar import c
from gen_ai_hub.proxy.langchain import init_llm
from hana_ml import dataframe
from hana_ai.agents.mem0_hana_agent import Mem0HANARAGAgent
from hana_ai.tools.toolkit import HANAMLToolkit

"""
Mem0 HANA Agent demo script.

This script mirrors the notebook usage pattern for HANAMLRAGAgent,
using Mem0HANARAGAgent with:
- auto ingestion classification
- entity extraction
- TTL controls (short vs long)
- tier-specific search commands
- export and expiration cleanup

Run:
    python nutest/testscripts/demo/e2e_scenarios/mem0_hana_agent_demo.py
"""

# Initialize tools and language model
import certifi

store = certifi.where()
connection_context = dataframe.ConnectionContext(userkey="RaysKey", sslValidateCertificate=True, encrypt=True, sslTrustStore=store)
tools = HANAMLToolkit(connection_context, used_tools='all').get_tools()
llm = init_llm('gpt-4.1', temperature=0.0, max_tokens=1800)

# Initialize Mem0 HANA chatbot
chatbot = Mem0HANARAGAgent(tools=tools, llm=llm, verbose=True)

# Optional: start with clean slate and set TTLs
print(chatbot.chat("!clear_long_term_memory"))
print(chatbot.chat("!set_ttl_short 86400"))   # 1 day short-term
print(chatbot.chat("!set_ttl_long 2592000"))  # 30 days long-term

HELP_TEXT = """
Commands:
  !clear_long_term_memory      - Clear all long-term memories
  !delete_expired              - Delete expired memories
  !export_memories             - Export memories (row count)
  !auto_ingest_on|off          - Toggle auto classification
  !auto_entity_on|off          - Toggle auto entity extraction
  !entity_assignment <mode>    - Set entity mode: manager|extract|merge
  !set_ttl_short <seconds>     - Set short-term TTL
  !set_ttl_long <seconds>      - Set long-term TTL
  !search_short <query>        - Search short-term memories
  !search_long <query>         - Search long-term memories
  !set_entity <id> <type>      - Set current entity partition
Type any other message to chat.
"""

print("Mem0 Agent ready. Type 'help' for commands, 'exit' to quit.")

# Conversation loop
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        if user_input.lower() == 'help':
            print(HELP_TEXT)
            continue

        response = chatbot.chat(user_input)
        print(f"Assistant: {response}")
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {str(e)}")
