from gen_ai_hub.proxy.langchain import init_llm
from hana_ml import dataframe
from hana_ai.agents.hanaml_rag_agent import HANAMLRAGAgent
from hana_ai.tools.toolkit import HANAMLToolkit

# Initialize tools and language model
connection_context = dataframe.ConnectionContext(userkey="RaysKey")
tools = HANAMLToolkit(connection_context, used_tools='all').get_tools()
llm = init_llm('gpt-4.1', temperature=0.0, max_tokens=1800)

# Initialize chatbot
chatbot = HANAMLRAGAgent(tools=tools, llm=llm)

# Conversation loop
print("Chat with AI Assistant (type 'exit' to quit):")
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
            
        response = chatbot.chat(user_input)
        print(f"Assistant: {response}")
        
        # Print memory status
        print("\nMemory Status:")
        print(f"Short-term entries: {len(chatbot.short_term_memory.chat_memory.messages)}")
        print(f"Long-term entries: {len(chatbot.long_term_store.messages)}")
        print(f"Vectorstore items: {chatbot.vectorstore.index.ntotal if hasattr(chatbot.vectorstore, 'index') else 'N/A'}\n")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {str(e)}")
