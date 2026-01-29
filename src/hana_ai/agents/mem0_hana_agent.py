"""
Mem0-powered HANA RAG Agent.

This agent fully uses the Mem0-like adapter (`Mem0HanaAdapter`) for long-term memory
on HANA, including optional cross-encoder reranking. It preserves short-term memory
behavior and tool-driven agent execution similar to `HANAMLRAGAgent`, but routes all
long-term memory operations (add/search/delete/clear) through the adapter.

Usage highlights:
- Enable reranking via `Mem0HanaAdapter` (using `PALCrossEncoder` when available).
- Store conversational summaries into HANA vector store with timestamp metadata.
- Retrieve relevant memories via adapter search with `top_k`, `threshold`, and `filters`.
- Optionally skip LLM agent initialization for unit tests (`auto_init_agent=False`).

Note: This class is designed to be drop-in alongside `HANAMLRAGAgent` while using
Mem0's style of memory flow adapted to HANA.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, AgentAction, AgentFinish
from langchain.embeddings.base import Embeddings
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_core.tools import BaseTool

from hana_ml.algorithms.pal.utility import check_pal_function_exist

from hana_ai.mem0.hana_mem0_adapter import SearchResult
from hana_ai.mem0.memory_manager import Mem0MemoryManager, IngestionRules
from hana_ai.mem0.memory_classifier import Mem0IngestionClassifier
from hana_ai.mem0.memory_entity_extractor import Mem0EntityExtractor
from hana_ai.agents.utilities import _get_user_info
from hana_ai.vectorstore.embedding_service import HANAVectorEmbeddings
from hana_ai.vectorstore.pal_cross_encoder import PALCrossEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FormatSafeAgentExecutor(AgentExecutor):
    """An AgentExecutor that formats observations safely."""
    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        next_step = super()._take_next_step(
            name_to_tool_map, color_mapping, inputs, intermediate_steps, run_manager
        )
        if isinstance(next_step, list):
            formatted_steps = []
            for action, observation in next_step:
                if isinstance(observation, str):
                    formatted_obs = [{"type": "text", "text": observation}]
                    formatted_steps.append((action, formatted_obs))
                else:
                    formatted_steps.append((action, observation))
            return formatted_steps
        return next_step


class Mem0HANARAGAgent:
    """
    A chatbot that uses Mem0-style long-term memory on HANA with reranking.
    """
    def __init__(
        self,
        tools: List[Tool],
        llm: Any,
        memory_window: int = 10,
        rerank_candidates: int = 20,
        rerank_k: int = 3,
        score_threshold: float = 0.5,
        hana_vector_table: Optional[str] = None,
        drop_existing_hana_vector_table: bool = False,
        verbose: bool = False,
        session_id: str = "global_session",
        _auto_init_agent: bool = True,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.memory_window = memory_window
        self.rerank_candidates = rerank_candidates
        self.rerank_k = rerank_k
        self.score_threshold = score_threshold
        self.verbose = verbose
        self.session_id = session_id
        self.drop_existing_hana_vector_table = drop_existing_hana_vector_table

        # Detect HANA connection_context from tools
        self.hana_connection_context = None
        for tool in self.tools:
            if hasattr(tool, 'connection_context'):
                self.hana_connection_context = tool.connection_context
                break

        # Prepare embedder only when building adapter locally
        self.embedding_service: Optional[Embeddings] = None

        # Table naming
        self.user = ''
        if hana_vector_table is None and self.hana_connection_context:
            self.user = _get_user_info(self.hana_connection_context)
            self.hana_vector_table = f"HANA_AI_CHAT_HISTORY_{self.user}"
        else:
            self.hana_vector_table = hana_vector_table or "HANA_AI_CHAT_HISTORY_DEFAULT"

        # Initialize memory manager (internal injection via _adapter for tests kept for BC but discouraged)
        if not self.hana_connection_context:
            raise ValueError("HANA connection context is required for Mem0HANARAGAgent")
        # Initialize embedding service and optional reranker
        self.embedding_service = HANAVectorEmbeddings(self.hana_connection_context)
        reranker = None
        try:
            if self.hana_connection_context and check_pal_function_exist(self.hana_connection_context, '%PAL_CROSSENCODER%', like=True):
                reranker = PALCrossEncoder(self.hana_connection_context)
        except Exception:
            reranker = None

        ingestion_rules = IngestionRules(
            enabled=True,
            min_length=1,
            max_length=None,
            allow_tags=None,
            deny_tags=None,
        )

        self.memory_manager = Mem0MemoryManager(
            connection_context=self.hana_connection_context,
            table_name=self.hana_vector_table,
            embedder=self.embedding_service,
            reranker=reranker,
            architecture="vector",
            default_ttl_seconds=None,
            short_term_ttl_seconds=None,
            partition_defaults={"agent_id": "mem0_hana_agent", "session_id": self.session_id},
            ingestion_rules=ingestion_rules,
            auto_classification_enabled=True,
            auto_entity_extraction_enabled=True,
            entity_assignment_mode="merge",
        )

        # Scope manager to current user entity
        if getattr(self, "user", None):
            self.memory_manager.set_entity(self.user, "user")
        try:
            classifier = Mem0IngestionClassifier(self.llm)
            self.memory_manager.set_classifier(classifier)
        except Exception:
            pass
        try:
            extractor = Mem0EntityExtractor(self.llm)
            self.memory_manager.set_entity_extractor(extractor)
        except Exception:
            pass

        # Optionally initialize LLM agent (internal flag for tests only)
        if _auto_init_agent:
            self._initialize_agent()

    def _initialize_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant with access to tools. Always use tools when appropriate."),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        self.agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        self.executor = FormatSafeAgentExecutor(
            agent=self.agent,
            tools=self.tools,
            max_iterations=20,
            verbose=self.verbose,
            handle_parsing_errors=True,
        )

    # ------------------------------------------------------------------
    # Long-term memory operations via adapter
    # ------------------------------------------------------------------
    def _update_long_term_memory(self, user_input: str, response: Any) -> None:
        self.memory_manager.add_interaction(user_input=user_input, assistant_output=str(response), tags=["chat", "conversation"], tier="long")

    def _retrieve_relevant_memories(self, query: str) -> List[str]:
        # Fetch candidates, rerank inside adapter
        results: List[SearchResult] = self.memory_manager.retrieve(
            query=query,
            top_k=self.rerank_candidates,
            threshold=self.score_threshold,
            tags=None,
            rerank=True,
        )
        # Take top-k after rerank
        results = results[:self.rerank_k]
        return [r.text for r in results]


    def clear_long_term_memory(self) -> None:
        """Clear all long-term memories from HANA."""
        try:
            self.memory_manager.clear_all()
            logger.info("Mem0 long-term memory cleared.")
        except Exception as e:
            logger.error("Failed to clear long-term memory: %s", e)


    # ------------------------------------------------------------------
    # Public chat API
    # ------------------------------------------------------------------
    def chat(self, user_input: str) -> str:
        """Process user input and return agent response, managing long-term memory."""
        if user_input.startswith("!clear_long_term_memory"):
            self.clear_long_term_memory()
            return "Long-term memory has been cleared."

        if user_input.startswith("!delete_expired"):
            deleted = self.memory_manager.delete_expired()
            return f"Expired memories deleted: {deleted}."

        if user_input.startswith("!export_memories"):
            try:
                rows = self.memory_manager.export()
                return f"Exported memories: {len(rows)} rows."
            except Exception as e:
                return f"Export failed: {e}"

        if user_input.strip() == "!auto_ingest_on":
            self.memory_manager.set_auto_classification_enabled(True)
            return "Auto ingestion classification enabled."

        if user_input.strip() == "!auto_ingest_off":
            self.memory_manager.set_auto_classification_enabled(False)
            return "Auto ingestion classification disabled."

        if user_input.strip() == "!auto_entity_on":
            self.memory_manager.set_auto_entity_extraction_enabled(True)
            return "Auto entity extraction enabled."

        if user_input.strip() == "!auto_entity_off":
            self.memory_manager.set_auto_entity_extraction_enabled(False)
            return "Auto entity extraction disabled."

        if user_input.startswith("!entity_assignment "):
            try:
                _, mode = user_input.split(maxsplit=1)
                self.memory_manager.set_entity_assignment_mode(mode)
                return f"Entity assignment mode set to: {mode}."
            except Exception:
                return "Usage: !entity_assignment <manager|extract|merge>"

        if user_input.startswith("!set_ttl_long "):
            try:
                _, sec = user_input.split(maxsplit=1)
                self.memory_manager.set_default_ttl_seconds(int(sec))
                return f"Default long-term TTL set to: {sec} seconds."
            except Exception:
                return "Usage: !set_ttl_long <seconds>"

        if user_input.startswith("!set_ttl_short "):
            try:
                _, sec = user_input.split(maxsplit=1)
                self.memory_manager.set_short_term_ttl_seconds(int(sec))
                return f"Default short-term TTL set to: {sec} seconds."
            except Exception:
                return "Usage: !set_ttl_short <seconds>"

        if user_input.startswith("!search_short "):
            try:
                _, q = user_input.split(maxsplit=1)
                results = self.memory_manager.retrieve_by_tier(query=q, tier="short", top_k=self.rerank_k, threshold=self.score_threshold, rerank=True)
                texts = [r.text for r in results]
                return f"Short-term hits ({len(texts)}):\n" + "\n".join(texts)
            except Exception as e:
                return f"search_short failed: {e}"

        if user_input.startswith("!search_long "):
            try:
                _, q = user_input.split(maxsplit=1)
                results = self.memory_manager.retrieve_by_tier(query=q, tier="long", top_k=self.rerank_k, threshold=self.score_threshold, rerank=True)
                texts = [r.text for r in results]
                return f"Long-term hits ({len(texts)}):\n" + "\n".join(texts)
            except Exception as e:
                return f"search_long failed: {e}"

        if user_input.startswith("!set_entity "):
            try:
                _, eid, etype = user_input.split(maxsplit=2)
                self.memory_manager.set_entity(eid, etype)
                return f"Entity set to id={eid}, type={etype}."
            except Exception:
                return "Usage: !set_entity <entity_id> <entity_type>"


        # Build context string via adapter memories
        long_term_context = self._retrieve_relevant_memories(user_input)
        context_str = "\n".join(long_term_context) if long_term_context else ""
        agent_input = {
            "input": [{
                "type": "text",
                "text": f"Context:\n{context_str}\n\nQuestion: {user_input}",
            }]
        }
        response = self.executor.invoke(agent_input)
        output = response["output"]
        self._update_long_term_memory(user_input, output)
        return output
