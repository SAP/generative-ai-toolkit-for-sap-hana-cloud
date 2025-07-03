from flask import Flask, request, jsonify
from gen_ai_hub.proxy.langchain import init_llm
from hana_ml import dataframe
from hana_ai.agents.hanaml_rag_agent import HANAMLRAGAgent
from hana_ai.tools.toolkit import HANAMLToolkit
import threading

# 创建 Flask 应用
app = Flask(__name__)

# 全局会话管理器
session_manager = {}
session_lock = threading.Lock()

def create_agent_session(session_id):
    """
    Create a new HANAMLRAGAgent session.

    Parameters
    ----------
    session_id : str
        Unique identifier for the session.
    """
    connection_context = dataframe.ConnectionContext(userkey="RaysKey")
    tools = HANAMLToolkit(connection_context, used_tools='all').get_tools()
    llm = init_llm('gpt-4.1', temperature=0.0, max_tokens=1800)
    return HANAMLRAGAgent(tools=tools, llm=llm,
                          vectorstore_path=f"chat_history_vectorstore_{session_id}",
                          long_term_db=f"sqlite:///chat_history_{session_id}.db")

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    Handle chat requests to the chatbot service.
    """
    data = request.get_json()
    message = data.get('message')
    session_id = data.get('session_id', 'default')
    
    if not message:
        return jsonify({"error": "Missing 'message' in request"}), 400
    
    with session_lock:
        if session_id not in session_manager:
            session_manager[session_id] = create_agent_session(session_id)
        chatbot = session_manager[session_id]
    
    try:
        response = chatbot.chat(message)
        memory_status = {
            "short_term_entries": len(chatbot.short_term_memory.chat_memory.messages),
            "long_term_entries": len(chatbot.long_term_store.messages),
            "vectorstore_items": chatbot.vectorstore.index.ntotal if hasattr(chatbot.vectorstore, 'index') else 'N/A'
        }
        
        return jsonify({
            "response": response,
            "memory_status": memory_status,
            "session_id": session_id
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """
    Delete a specific session by session_id.

    Parameter
    ---------
    session_id : str
        Unique identifier for the session to be deleted.
    """
    with session_lock:
        if session_id in session_manager:
            del session_manager[session_id]
            return jsonify({"status": f"Session {session_id} deleted"})
        return jsonify({"error": "Session not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4567, debug=True)

# # 创建新会话
# curl -X POST http://localhost:4567/chat -H "Content-Type: application/json" -d '{"session_id":"new_session", "message":"show me 10 records from SALES_REFUNDS table"}'

# # 删除会话
# curl -X DELETE http://localhost:4567/sessions/new_session
