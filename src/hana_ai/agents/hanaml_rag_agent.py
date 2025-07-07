"""
A chatbot that can remember the short term and long term chat history and use it to generate responses.

The following class is available:
    
        * :class `HANAMLRAGAgent`
"""

#pylint: disable=attribute-defined-outside-init, ununsed-argument, no-else-return

import re
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging
import pandas as pd
from sqlalchemy import delete
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage, AgentAction, AgentFinish
from langchain.load.dump import dumps
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_core.tools import BaseTool
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.hanavector import HanaDB

from sentence_transformers import CrossEncoder
from hana_ai.agents.utilities import _check_generated_cap_for_bas, _inspect_python_code
from hana_ai.vectorstore.embedding_service import GenAIHubEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FormatSafeAgentExecutor(AgentExecutor):
    """
    An AgentExecutor that safely formats the output of the agent to ensure
    that the observation is always a list of objects, even if it was originally
    a string. This is useful for ensuring compatibility with downstream
    components that expect a specific format.
    """
    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        # 调用原始逻辑获取下一步动作
        next_step = super()._take_next_step(
            name_to_tool_map, color_mapping, inputs, intermediate_steps, run_manager
        )

        # 仅处理AgentAction（工具调用结果）
        if isinstance(next_step, list):
            formatted_steps = []
            for action, observation in next_step:
                # 关键转换：将字符串observation转为对象数组
                if isinstance(observation, str):
                    formatted_obs = [{"type": "text", "text": observation}]
                    formatted_steps.append((action, formatted_obs))
                else:
                    formatted_steps.append((action, observation))
            return formatted_steps

        return next_step  # AgentFinish直接返回

class HANAMLRAGAgent:
    """
    A chatbot that integrates short-term and long-term memory systems using RAG (Retrieval-Augmented Generation).
    """
    def __init__(self,
                 tools: List[Tool],
                 llm: Any,
                 vectorstore_path: str = "chat_history_vectorstore", # FAISS vectorstore path
                 memory_window: int = 10,
                 long_term_db: str = "sqlite:///chat_history.db",
                 long_term_memory_limit: int = 1000,
                 skip_large_data_threshold: int = 10000,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 forget_percentage: float = 0.1,
                 max_iterations: int = 10,
                 cross_encoder: CrossEncoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'),
                 rerank_candidates: int = 20,
                 rerank_k: int = 3,
                 score_threshold: float = 0.5,
                 vector_store_type = "faiss", # 'faiss' or 'hanadb'
                 hana_connection_context = None, # Required if vector_store is 'hanadb'
                 hana_vector_table: str = "HANA_AI_CHAT_HISTORY",
                 drop_existing_hana_vector_table: bool = False,
                 verbose: bool = False):
        """
        Initialize the chatbot with integrated tools and memory systems.

        Parameters
        ----------
        tools : List[Tool]
            List of LangChain tools to be used by the agent.
        llm : Any
            Language model instance to be used for generating responses.
        vectorstore_path : str
            Path to store the vectorstore for long-term memory.
        memory_window : int
            Number of recent conversations to keep in short-term memory.

            Defaults to 10.
        long_term_db : str
            Connection string for long-term memory storage.

            Defaults to "sqlite:///chat_history.db".
        long_term_memory_limit : int
            Maximum number of long-term memory entries to retain.

            Defaults to 1000.
        skip_large_data_threshold : int
            Skip storing texts longer than this threshold.

            Defaults to 1000.
        chunk_size : int
            Text chunk size for embeddings.

            Defaults to 500.
        chunk_overlap : int
            Text chunk overlap for embeddings.

            Defaults to 50.
        forget_percentage : float
            Percentage of oldest memories to forget when long-term memory limit is reached.

            Defaults to 0.1 (10%).
        max_iterations : int
            Maximum number of iterations for agent execution.

            Defaults to 10.
        cross_encoder : CrossEncoder
            Cross-encoder model for reranking retrieved documents.

            Defaults to a pre-trained cross-encoder model 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
        rerank_candidates : int
            Number of candidate documents to retrieve for reranking.

            Defaults to 20.           
        rerank_k : int
            Number of documents to retrieve for reranking.

            Defaults to 3.
        score_threshold : float
            Similarity score threshold for retrieval.

            Defaults to 0.5.
        verbose : bool
            Whether to enable verbose logging.

            Defaults to False.
        """
        self.llm = llm
        self.tools = tools
        self.vectorstore_path = vectorstore_path
        self.long_term_db = long_term_db
        self.long_term_memory_limit = long_term_memory_limit
        self.memory_window = memory_window
        self.skip_large_data_threshold = skip_large_data_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.forget_percentage = forget_percentage
        self.max_iterations = max_iterations
        self.rerank_candidates = rerank_candidates
        self.rerank_k = rerank_k
        self.score_threshold = score_threshold
        self.verbose = verbose
        self.cross_encoder = cross_encoder
        self.vectorstore_type = vector_store_type
        self.hana_connection_context = hana_connection_context
        self.hana_vector_table = hana_vector_table
        self.drop_existing_hana_vector_table = drop_existing_hana_vector_table

        # Initialize memory systems
        self._initialize_memory()

        # Initialize agent with tool integration
        self._initialize_agent()

    def _initialize_memory(self):
        """Initialize short-term and long-term memory systems"""
        # Short-term memory (recent conversations)
        self.short_term_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=self.memory_window,
            return_messages=True
        )
        # Long-term memory storage
        self.long_term_store = SQLChatMessageHistory(
            connection_string=self.long_term_db,
            session_id="global_session"
        )
        # Initialize RAG vectorstore
        self._initialize_vectorstore()

    def delete_message_long_term_store(self, message_id) -> None:
        """Delete a specific message by its ID"""
        long_term_store = self.long_term_store
        try:
            long_term_store._create_table_if_not_exists()
            with long_term_store._make_sync_session() as session:
                stmt = delete(long_term_store.sql_model_class).where(
                    long_term_store.sql_model_class.id == message_id,
                    getattr(long_term_store.sql_model_class, long_term_store.session_id_field_name) == long_term_store.session_id
                )
                session.execute(stmt)
                session.commit()
        except Exception as e:
            logger.error("Failed to delete message with ID %s: %s", message_id, str(e))

    def _initialize_faiss_vectorstore(self):
        """Initialize or load FAISS vectorstore for long-term memory"""
        # Create embeddings service
        self.embeddings = GenAIHubEmbeddings()

        # Try loading existing vectorstore
        try:
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Loaded existing vectorstore from %s", self.vectorstore_path)
        except:
            # Initialize new vectorstore if loading fails
            self.vectorstore = FAISS.from_texts(
                texts=["Initialization text"],
                embedding=self.embeddings
            )
            self.vectorstore.save_local(self.vectorstore_path)
            logger.info("Created new vectorstore at %s", self.vectorstore_path)

    def _initialize_hanadb_vectorstore(self):
        """Initialize or load HANA DB vectorstore for long-term memory"""
        if not self.hana_connection_context:
            raise ValueError("HANA connection context is required for HANA DB vectorstore")

        # Create embeddings service
        self.embeddings = GenAIHubEmbeddings()

        # Initialize or load HANA vectorstore
        if self.drop_existing_hana_vector_table:
            self.hana_connection_context.drop_table(self.hana_vector_table)
        try:
            self.vectorstore = HanaDB(
                embedding=self.embeddings,
                connection=self.hana_connection_context.connection,
                table_name=self.hana_vector_table
            )
            logger.info("Initialized HANA DB vectorstore with table %s", self.hana_vector_table)
        except Exception as e:
            logger.error("Failed to initialize HANA DB vectorstore: %s", str(e))
            raise e

    def _initialize_vectorstore(self):
        if self.vectorstore_type.lower() == "faiss":
            self._initialize_faiss_vectorstore()
        else:
            self._initialize_hanadb_vectorstore()

    def _should_store(self, text: str) -> bool:
        """Determine if text should be stored in memory"""
        # Skip DataFrame conversions
        if isinstance(text, pd.DataFrame):
            return False

        # Skip large texts
        if isinstance(text, str) and len(text) > self.skip_large_data_threshold:
            return False

        # Skip Markdown tables
        if isinstance(text, str) and re.search(r'\|\s*.*\s*\|.*\n(\|\s*:?[-]+\s*:?\|\s*)+\n', text):
            return False

        return True

    def _update_long_term_memory(self, user_input: str, response: Any):
        """Update long-term memory with new conversation"""
        if not self._should_store(response):
            logger.debug("Skipping memory storage for large or special content")
            return

        response_str = str(response)
        current_time = datetime.now().isoformat()

        # Add messages with metadata
        self.long_term_store.add_messages([
            HumanMessage(content=[{"type": "text", "text": user_input}], metadata={"timestamp": current_time}),
            AIMessage(content=[{"type": "text", "text": str(response)}], metadata={"timestamp": current_time})
        ])

        # Create documents for vector store
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        documents = splitter.create_documents(
            texts=[f"User: {user_input}\nAssistant: {response_str}"],
            metadatas=[{"timestamp": current_time}]
        )

        # Update vector store
        self.vectorstore.add_documents(documents)
        if self.vectorstore_type.lower() == "faiss":
            self.vectorstore.save_local(self.vectorstore_path)

        # Clean up oldest memories if needed
        self._forget_old_memories()

    def _forget_old_memories(self):
        """Remove oldest memories when storage limit is exceeded"""
        if len(self.long_term_store.messages) > self.long_term_memory_limit:
            # Calculate number of oldest messages to remove
            num_to_remove = int(self.long_term_memory_limit * self.forget_percentage)
            # 1. Delete from SQL database
            try:
                # Get sorted list of messages by timestamp
                sorted_messages = sorted(
                    self.long_term_store.messages,
                    key=lambda msg: msg.metadata.get('timestamp', '1970-01-01')
                )
                # Delete oldest messages from SQL store
                for msg in sorted_messages[:num_to_remove]:
                    self.delete_message_long_term_store(msg.id)
                logger.info("Deleted %s oldest memories from SQL store", num_to_remove)
            except Exception as e:
                logger.error("Error deleting from SQL store: %s", str(e))
            # 2. Rebuild vectorstore from remaining messages
            try:
                # Get remaining messages after deletion
                remaining_messages = self.long_term_store.messages
                # Skip if no messages left
                if not remaining_messages:
                    if self.vectorstore_type.lower() == "faiss":
                        self.vectorstore = FAISS.from_texts(
                            [""],
                            embedding=self.embeddings
                        )
                    else:
                        self.hana_connection_context.drop_table(self.hana_vector_table)
                        self.vectorstore = HanaDB.from_texts(
                            texts=[""],
                            embedding=self.embeddings,
                            connection=self.hana_connection_context.connection,
                            table_name=self.hana_vector_table
                        )
                    return
                # Recreate documents from remaining messages
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                documents = []
                for i in range(0, len(remaining_messages), 2):
                    if i+1 >= len(remaining_messages):
                        break  # Skip incomplete pairs
                    user_msg = remaining_messages[i]
                    ai_msg = remaining_messages[i+1]
                    # Create combined document
                    text = f"User: {user_msg.content}\nAssistant: {ai_msg.content}"
                    metadata = {"timestamp": user_msg.metadata.get('timestamp', '')}
                    # Split and add to documents
                    docs = splitter.create_documents(
                        texts=[text],
                        metadatas=[metadata]
                    )
                    documents.extend(docs)
                # Rebuild vectorstore
                if self.vectorstore_type.lower() == "faiss":
                    self.vectorstore = FAISS.from_documents(
                        documents,
                        self.embeddings
                    )
                    self.vectorstore.save_local(self.vectorstore_path)
                else:
                    self.hana_connection_context.drop_table(self.hana_vector_table)
                    self.vectorstore = HanaDB.from_documents(
                        documents,
                        self.embeddings,
                        connection=self.hana_connection_context.connection,
                        table_name=self.hana_vector_table
                    )
                logger.info("Rebuilt vectorstore with %s documents", len(documents))
            except Exception as e:
                logger.error("Error rebuilding vectorstore: %s", str(e))
                # Fallback to empty vectorstore
                if self.vectorstore_type.lower() == "faiss":
                    self.vectorstore = FAISS.from_texts(
                        [""],
                        embedding=self.embeddings
                    )
                else:
                    self.hana_connection_context.drop_table(self.hana_vector_table)
                    self.vectorstore = HanaDB.from_texts(
                        texts=[""],
                        embedding=self.embeddings,
                        connection=self.hana_connection_context.connection,
                        table_name=self.hana_vector_table
                    )

    def _retrieve_relevant_memories(self, query: str) -> List[str]:
        """Retrieve relevant memories using RAG with reranking"""
        # Retrieve candidate documents
        if self.vectorstore_type.lower() == "faiss":
            candidate_docs = self.vectorstore.similarity_search_with_score(
                query=query,
                k=self.rerank_candidates,
                score_threshold=self.score_threshold
            )
        else:
            candidate_docs = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=self.rerank_candidates,
                score_threshold=self.score_threshold
            )

        if not candidate_docs:
            return []

        # Prioritize recent documents
        candidate_docs.sort(
            key=lambda x: x[0].metadata.get('timestamp', '1970-01-01'),
            reverse=True
        )

        # Rerank with cross-encoder
        doc_contents = [doc[0].page_content for doc in candidate_docs]
        rerank_scores = self.cross_encoder.predict([(query, content) for content in doc_contents])

        # Combine and select top documents
        combined = list(zip(doc_contents, rerank_scores))
        combined.sort(key=lambda x: x[1], reverse=True)

        return [content for content, _ in combined[:self.rerank_k]]

    def _build_context(self, user_input: str) -> SystemMessage:
        # 获取短时记忆并转换格式
        short_term_history = self.short_term_memory.load_memory_variables({})["chat_history"]
        formatted_history = []
        for msg in short_term_history:
            if isinstance(msg, (HumanMessage, AIMessage)):
                formatted_history.append({"type": "text", "text": f"{msg.type}: {msg.content}"})

        # 获取长时记忆
        long_term_context = self._retrieve_relevant_memories(user_input)
        long_term_str = "\n".join(long_term_context) if long_term_context else ""

        # 构建内容块
        context_content = [
            {"type": "text", "text": "CONVERSATION CONTEXT:"},
            {"type": "text", "text": "Recent Chat History:"},
            *formatted_history,
            {"type": "text", "text": f"Relevant Long-term Memories: {long_term_str}"},
            {"type": "text", "text": "GUIDELINES:\n1. Prioritize tool usage..."}
        ]
        return SystemMessage(content=context_content)

    def _initialize_agent(self):
        """Initialize agent with tool integration and context handling"""
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant with access to tools. Always use tools when appropriate."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create the agent with tool bindings
        self.agent = create_openai_functions_agent(self.llm, self.tools, prompt)

        # Create executor with memory integration
        self.executor = FormatSafeAgentExecutor(
            agent=self.agent,
            tools=self.tools,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            memory=self.short_term_memory,
            handle_parsing_errors=True
        )

    def _format_dataframe(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to Markdown table"""
        try:
            return df.to_markdown(index=False)
        except Exception as e:
            logger.error("DataFrame conversion failed: %s", str(e))
            return "Data output could not be formatted"

    def _build_long_term_context(self, user_input: str) -> str:
        long_term_context = self._retrieve_relevant_memories(user_input)
        return "\n".join(long_term_context) if long_term_context else ""

    def clear_long_term_memory(self):
        """
        Safely clear long-term memory by clearing the vectorstore and database.
        Avoids index errors in SQLChatMessageHistory implementation.
        """
        try:
            # 安全清空SQL存储（避免索引越界）
            self.long_term_store.clear()
            logger.debug("SQL long-term store cleared")
        except IndexError:
            # 捕获索引越界异常（空存储时可能发生）
            logger.warning("IndexError during clear (likely empty store)")
        except Exception as e:
            logger.error("Unexpected error clearing SQL store: %s", str(e))

        # 重建空向量库（无需前置检查）
        if self.vectorstore_type.lower() == "faiss":
            self.vectorstore = FAISS.from_texts([""], embedding=self.embeddings)  # 空文本占位
            self.vectorstore.save_local(self.vectorstore_path)
        else:
            self.hana_connection_context.drop_table(self.hana_vector_table)
            self.vectorstore = HanaDB.from_texts(
                texts=[""],
                embedding=self.embeddings,
                connection=self.hana_connection_context.connection,
                table_name=self.hana_vector_table
            )
        logger.info("Long-term memory cleared and vectorstore reset.")

    def clear_short_term_memory(self):
        """
        Clear short-term memory by resetting the conversation history.
        This is a placeholder for actual implementation.
        """
        self.short_term_memory.clear()
        logger.info("Short-term memory cleared.")

    def chat(self, user_input: str) -> str:
        """
        Main chat method to handle user input and return response.

        Parameters
        ----------
        user_input : str
            The input question or statement from the user.
        """
        if user_input.startswith("!clear_long_term_memory"):
            self.clear_long_term_memory()
            return "Long-term memory has been cleared."
        elif user_input.startswith("!clear_short_term_memory"):
            self.clear_short_term_memory()
            return "Short-term memory has been cleared."
        context_str = self._build_long_term_context(user_input)  # Returns string
        agent_input = {
            "input": [{
                "type": "text", 
                "text": f"Context:\n{context_str}\n\nQuestion: {user_input}"
            }]
        }
        response = self.executor.invoke(agent_input)
        self._update_long_term_memory(user_input, response['output'])
        return response['output']

def stateless_chat(
    query: str,
    tools: List[BaseTool],
    llm: Any,
    memory: List[str]
) -> str:
    """
    Stateless chat function that integrates tools and memory for RAG.

    Parameters
    ----------
    query : str
        The user query to process.
    tools : List[BaseTool]
        List of tools available for the agent to use.
    llm : Any
        The language model instance to generate responses.
    memory : List[str]
        List of long-term memory entries to integrate into the context.
    """

    # 1. 构建系统上下文 (整合长时记忆)
    def build_system_context(query: str, memory: List[str]) -> SystemMessage:
        """
        Build the system context message with long-term memory integration.
        """
        valid_memories = [m for m in memory if m.strip()]
        memory_block = "\n".join([f"- {m}" for m in valid_memories]) if valid_memories else "No long-term memory available."

        context_content = [
            {"type": "text", "text": "## System Instructions"},
            {"type": "text", "text": "You are a helpful assistant with access to tools. Always use tools when appropriate."},
            {"type": "text", "text": "## Recent Chat History"},
            {"type": "text", "text": memory_block},
            {"type": "text", "text": "## Operating Rules\n1. Respond directly to simple questions\n2. Use tools when calculation/query is needed"}
        ]
        return SystemMessage(content=context_content)

    # 2. 初始化提示模板
    system_message = build_system_context(query, memory)
    prompt_template = ChatPromptTemplate.from_messages([
        system_message,
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # 3. 创建带空记忆的代理
    agent = create_openai_functions_agent(llm, tools, prompt_template)
    agent_executor = FormatSafeAgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=10,
        handle_parsing_errors=True,
        memory=ConversationBufferWindowMemory(  # 空短期记忆
            memory_key="chat_history",
            k=0,
            return_messages=True
        ),
        return_intermediate_steps=True,
    )

    # 4. 执行查询并返回响应
    response = agent_executor.invoke({"input": query})
    intermediate_steps = response.get("intermediate_steps")
    response["intermediate_steps"] = dumps(intermediate_steps) if intermediate_steps else None
    response["inspect_script"] = _inspect_python_code(response["intermediate_steps"], tools)
    response["generated_cap_project"] = _check_generated_cap_for_bas(response["intermediate_steps"])
    return response
