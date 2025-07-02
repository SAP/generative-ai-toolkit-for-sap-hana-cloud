"""
A chatbot that can remember the short term and long term chat history and use it to generate responses.

The following class is available:
    
        * :class `HANAMLRAGAgent`
"""

import re
from typing import List, Any
from datetime import datetime
import logging
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.vectorstores import FAISS

from sentence_transformers import CrossEncoder
from hana_ai.vectorstore.embedding_service import GenAIHubEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HANAMLRAGAgent:
    """
    A chatbot that integrates short-term and long-term memory systems using RAG (Retrieval-Augmented Generation).
    """
    def __init__(self,
                 tools: List[Tool],
                 llm: Any,
                 vectorstore_path: str = "chat_history_vectorstore",
                 memory_window: int = 10,
                 long_term_db: str = "sqlite:///chat_history.db",
                 long_term_memory_limit: int = 1000,
                 skip_large_data_threshold: int = 1000,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 forget_percentage: float = 0.1,
                 max_iterations: int = 10,
                 cross_encoder: CrossEncoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'),
                 rerank_k: int = 10,
                 score_threshold: float = 0.5,
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
        rerank_k : int
            Number of documents to retrieve for reranking.

            Defaults to 10.
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
        self.rerank_k = rerank_k
        self.score_threshold = score_threshold
        self.verbose = verbose
        self.cross_encoder = cross_encoder

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

    def _initialize_vectorstore(self):
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
            HumanMessage(content=user_input, metadata={"timestamp": current_time}),
            AIMessage(content=response_str, metadata={"timestamp": current_time})
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
        self.vectorstore.save_local(self.vectorstore_path)

        # Clean up oldest memories if needed
        self._forget_old_memories()

    def _forget_old_memories(self):
        """Remove oldest memories when storage limit is exceeded"""
        if len(self.long_term_store.messages) > self.long_term_memory_limit:
            # Calculate number of oldest messages to remove
            num_to_remove = int(self.long_term_memory_limit * self.forget_percentage)

            # Implement actual deletion logic here
            # This is placeholder for actual database deletion
            logger.info("Should remove %d oldest memories", num_to_remove)
            # In practice: Delete oldest records from database

    def _retrieve_relevant_memories(self, query: str) -> List[str]:
        """Retrieve relevant memories using RAG with reranking"""
        # Retrieve candidate documents
        candidate_docs = self.vectorstore.similarity_search_with_score(
            query=query,
            k=self.rerank_k,
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

        return [content for content, _ in combined[:3]]

    def _build_context(self, user_input: str) -> ChatPromptTemplate:
        """Build combined context from short-term and long-term memory"""
        # Get short-term memory
        short_term_history = self.short_term_memory.load_memory_variables({})["chat_history"]
        short_term_str = "\n".join([
            f"{msg.type.capitalize()}: {msg.content}"
            for msg in short_term_history
        ]) if short_term_history else "No recent conversation history."

        # Get relevant long-term memories
        long_term_context = self._retrieve_relevant_memories(user_input)
        long_term_str = "\n".join(long_term_context) if long_term_context else "No relevant historical context found."

        # Create system message with context
        context_message = SystemMessage(
            content=f"CONVERSATION CONTEXT:\n"
                    f"Recent Chat History:\n{short_term_str}\n\n"
                    f"Relevant Long-term Memories:\n{long_term_str}\n\n"
                    "GUIDELINES:\n"
                    "1. Prioritize tool usage when appropriate\n"
                    "2. When tools return tabular data, format as Markdown tables\n"
                    "3. For large outputs, summarize key insights\n"
        )

        return context_message

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
        self.executor = AgentExecutor(
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
        self.vectorstore = FAISS.from_texts([""], embedding=self.embeddings)  # 空文本占位
        self.vectorstore.save_local(self.vectorstore_path)
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
        combined_input = f"Context:\n{context_str}\n\nQuestion: {user_input}"

        agent_input = {"input": combined_input}  # Single key
        response = self.executor.invoke(agent_input)
        return response['output']
