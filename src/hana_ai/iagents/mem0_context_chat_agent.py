"""Context-engineered Mem0 chat agent.

Goal
----
Provide a drop-in chat agent with the same usage scope as `Mem0HANARAGAgent`, while
improving long-horizon coherence and answer quality via context engineering:

- High recall: retrieve a larger candidate set from Mem0 long/short-term memory.
- High precision: synthesize a compact, query-relevant `Memory Brief` before injecting
  into the tool-using agent context.
- Structured note-taking: persist short turn summaries as short-term memories.

This module intentionally avoids copying large chunks of other agents; it composes and
extends existing Mem0 + tool-agent infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import json
import logging
import re
import sys

from hana_ml.algorithms.pal.utility import check_pal_function_exist

from hana_ai.langchain_compat import (
    ChatPromptTemplate,
    Embeddings,
    FormatSafeAgentExecutor,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessage,
    Tool,
    build_agent_executor,
)
from hana_ai.agents.utilities import _get_user_info
from hana_ai.mem0.hana_mem0_adapter import SearchResult
from hana_ai.mem0.memory_classifier import Mem0IngestionClassifier
from hana_ai.mem0.memory_entity_extractor import Mem0EntityExtractor
from hana_ai.mem0.memory_manager import IngestionRules, Mem0MemoryManager
from hana_ai.vectorstore.embedding_service import HANAVectorEmbeddings
from hana_ai.vectorstore.pal_cross_encoder import PALCrossEncoder

# Reuse proven parsing helpers from Mem0HANARAGAgent to keep behavior consistent.
from hana_ai.agents.mem0_hana_agent import (  # pylint: disable=import-error
    _extract_last_predicted_table_from_texts,
    _extract_table_facts_from_steps,
)

logger = logging.getLogger(__name__)


@dataclass
class ContextPack:
    """A concrete, testable representation of what we put into the model context.

    This mirrors the layering described in mem_paper/tech.tex.
    """

    task: str
    tool_guidance: str
    working_set: str
    session_summary: str
    memory_notes: str
    memory_brief: str

    # Budgeting (tech.tex: treat context as finite, allocate per layer)
    max_context_chars: int = 9000
    budget_task: int = 1200
    budget_working_set: int = 900
    budget_memory_notes: int = 2400
    budget_session_summary: int = 2000
    budget_memory_brief: int = 3200
    budget_tool_guidance: int = 900

    def with_dynamic_budgets(self, user_input: str) -> "ContextPack":
        """Adjust budgets based on query length/complexity.

        Rationale (tech.tex): allocate attention to the smallest high-signal set.
        If the query is long, give more to task/working_set and less to auxiliary guidance.
        """
        q_len = len(user_input or "")
        pack = self

        # Long query: prioritize task + working_set; trim guidance and brief.
        if q_len >= 1200:
            pack.budget_task = 1800
            pack.budget_working_set = 1100
            pack.budget_memory_notes = 2400
            pack.budget_session_summary = 1800
            pack.budget_memory_brief = 2200
            pack.budget_tool_guidance = 600
            pack.max_context_chars = 9000
            return pack

        # Medium query
        if q_len >= 400:
            pack.budget_task = 1400
            pack.budget_working_set = 1000
            pack.budget_memory_notes = 2400
            pack.budget_session_summary = 2000
            pack.budget_memory_brief = 2800
            pack.budget_tool_guidance = 800
            pack.max_context_chars = 9000
            return pack

        # Short query: allow a bit more room for memory brief.
        pack.budget_task = 1100
        pack.budget_working_set = 900
        pack.budget_memory_notes = 2400
        pack.budget_session_summary = 2000
        pack.budget_memory_brief = 3400
        pack.budget_tool_guidance = 900
        pack.max_context_chars = 9000
        return pack

    def _render_parts(self, include_brief: bool) -> List[str]:
        parts: List[str] = []
        parts.append("<task>\n" + _truncate(self.task.strip(), self.budget_task) + "\n</task>")

        if self.working_set.strip():
            parts.append(
                "<working_set>\n" + _truncate(self.working_set.strip(), self.budget_working_set) + "\n</working_set>"
            )

        if self.memory_notes.strip():
            parts.append(
                "<memory_notes>\n" + _truncate(self.memory_notes.strip(), self.budget_memory_notes) + "\n</memory_notes>"
            )

        if self.session_summary.strip():
            parts.append(
                "<session_summary>\n"
                + _truncate(self.session_summary.strip(), self.budget_session_summary)
                + "\n</session_summary>"
            )

        if include_brief and self.memory_brief.strip():
            parts.append(
                "<memory_brief>\n" + _truncate(self.memory_brief.strip(), self.budget_memory_brief) + "\n</memory_brief>"
            )

        if self.tool_guidance.strip():
            parts.append(
                "<tool_guidance>\n" + _truncate(self.tool_guidance.strip(), self.budget_tool_guidance) + "\n</tool_guidance>"
            )
        return parts

    def render_without_brief(self) -> str:
        """Render context without memory_brief.

        Used to compute remaining budget for brief (budget closed-loop).
        """
        rendered = "\n\n".join(self._render_parts(include_brief=False))
        return _truncate(rendered, self.max_context_chars)

    def remaining_budget_for_brief(self) -> int:
        """Compute remaining character budget for memory brief.

        Returns a non-negative integer. The brief should also respect budget_memory_brief.
        """
        base = self.render_without_brief()
        remaining = max(0, self.max_context_chars - len(base) - len("\n\n<memory_brief>\n\n</memory_brief>"))
        return max(0, min(self.budget_memory_brief, remaining))

    def render(self) -> str:
        """Render the full context pack with all enabled layers."""
        rendered = "\n\n".join(self._render_parts(include_brief=True))
        return _truncate(rendered, self.max_context_chars)


@dataclass
class MemoryBriefConfig:
    """Controls how much memory we retrieve and how we compress it."""

    recall_candidates: int = 40  # fetch more for recall
    brief_candidates: int = 20  # how many top candidates to show the summarizer
    rerank_k: int = 6  # keep a small set for injection (pre-brief)
    score_threshold: float = 0.0
    max_brief_chars: int = 4000  # final injected brief size budget
    max_item_chars: int = 700  # per-memory truncation during fallback
    include_short_tier: bool = True


@dataclass
class StructuredNotesConfig:
    """Controls structured note-taking and retrieval policy."""

    enabled: bool = True
    top_k_per_type: int = 3
    max_notes_chars: int = 2000


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _dedupe_results(results: Sequence[SearchResult]) -> List[SearchResult]:
    """Dedupe results by content_hash when available, else by text prefix."""
    seen: set[str] = set()
    out: List[SearchResult] = []
    for r in results:
        key = None
        try:
            key = str((r.metadata or {}).get("content_hash") or "")
        except Exception:
            key = ""
        if not key:
            key = (r.text or "")[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _truncate(s: str, limit: int) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= limit else (s[: max(0, limit - 3)] + "...")


def _format_reference(r: SearchResult) -> str:
    md = r.metadata or {}
    ts = md.get("timestamp") or md.get("expires_at") or ""
    tags = md.get("tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    ch = md.get("content_hash") or ""
    tier = md.get("tier") or ""
    score = f"{r.rerank_score:.3f}" if r.rerank_score is not None else f"{r.score:.3f}"
    return f"hash={ch} tier={tier} score={score} ts={ts} tags={tags}"


class Mem0ContextChatAgent:
    """A context-engineered Mem0 chat agent with tool use.

    Public API mirrors `Mem0HANARAGAgent`:
    - constructor accepts similar parameters
    - `chat(user_input: str) -> str` supports the same bang-commands

    Differences:
    - memory injection uses a synthesized Memory Brief (high precision)
    - stores short per-turn summaries (structured note-taking)
    """

    def __init__(
        self,
        tools: List[Tool],
        llm: Any,
        memory_window: int = 10,
        rerank_candidates: int = 40,
        rerank_k: int = 6,
        score_threshold: float = 0.0,
        hana_vector_table: Optional[str] = None,
        drop_existing_hana_vector_table: bool = False,
        verbose: bool = False,
        session_id: str = "global_session",
        memory_brief: Optional[MemoryBriefConfig] = None,
        progress_bar: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
        _auto_init_agent: bool = True,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.memory_window = memory_window
        self.verbose = verbose
        self.session_id = session_id
        self.drop_existing_hana_vector_table = drop_existing_hana_vector_table

        # Optional progress reporting.
        # - progress_bar=True: emit human-readable stage updates to stdout (good for notebooks)
        # - progress_callback: override emission target/format (e.g., integrate with a UI)
        self.progress_bar = bool(progress_bar)
        self.progress_callback = progress_callback
        if self.progress_bar and self.progress_callback is None:
            self.progress_callback = self._default_progress_printer

        # Memory retrieval knobs (kept for compatibility)
        self.rerank_candidates = rerank_candidates
        self.rerank_k = rerank_k
        self.score_threshold = score_threshold

        # Context engineering knobs
        self.memory_brief_cfg = memory_brief or MemoryBriefConfig(
            recall_candidates=rerank_candidates,
            rerank_k=rerank_k,
            score_threshold=score_threshold,
        )
        self.structured_notes_cfg = StructuredNotesConfig()

        # Detect HANA connection_context from tools
        self.hana_connection_context = None
        for tool in self.tools:
            if hasattr(tool, "connection_context"):
                self.hana_connection_context = tool.connection_context
                break
        if not self.hana_connection_context:
            raise ValueError("HANA connection context is required for Mem0ContextChatAgent")

        # Prepare embedding service and optional reranker
        self.embedding_service: Optional[Embeddings] = HANAVectorEmbeddings(self.hana_connection_context)
        reranker = None
        try:
            if check_pal_function_exist(self.hana_connection_context, "%PAL_CROSSENCODER%", like=True):
                reranker = PALCrossEncoder(self.hana_connection_context)
        except Exception:
            reranker = None

        # Table naming
        self.user = ""
        if hana_vector_table is None and self.hana_connection_context:
            self.user = _get_user_info(self.hana_connection_context)
            self.hana_vector_table = f"HANA_AI_CHAT_HISTORY_{self.user}"
        else:
            self.hana_vector_table = hana_vector_table or "HANA_AI_CHAT_HISTORY_DEFAULT"

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
            partition_defaults={"agent_id": "mem0_context_chat_agent", "session_id": self.session_id},
            ingestion_rules=ingestion_rules,
            auto_classification_enabled=True,
            auto_entity_extraction_enabled=True,
            entity_assignment_mode="merge",
        )

        if getattr(self, "user", None):
            self.memory_manager.set_entity(self.user, "user")

        try:
            self.memory_manager.set_classifier(Mem0IngestionClassifier(self.llm))
        except Exception:
            pass

        try:
            self.memory_manager.set_entity_extractor(Mem0EntityExtractor(self.llm))
        except Exception:
            pass

        if _auto_init_agent:
            self._initialize_agent()

    def _emit_progress(self, message: str, *, step: Optional[int] = None, total_steps: Optional[int] = None) -> None:
        """Emit a lightweight progress update.

        - If progress_callback is provided, it is called.
        - Else if verbose=True, logs at INFO.
        """
        prefix = ""
        if step is not None and total_steps is not None and total_steps > 0:
            prefix = f"[{step}/{total_steps}] "
        text = f"{prefix}{message}".strip()

        cb = getattr(self, "progress_callback", None)
        if cb is not None:
            try:
                cb(text)
                return
            except Exception:
                # Never allow progress reporting to break the agent.
                pass

        if getattr(self, "verbose", False):
            logger.info(text)

    def _default_progress_printer(self, text: str) -> None:
        """Default progress printer used when progress_bar=True.

        Prints one line per stage to stdout; intentionally simple and dependency-free.
        """
        try:
            sys.stdout.write(str(text).rstrip() + "\n")
            sys.stdout.flush()
        except Exception:
            # Never fail the agent due to progress printing.
            pass

    # ------------------------------------------------------------------
    # Agent executor
    # ------------------------------------------------------------------
    def _initialize_agent(self) -> None:
        # Structured system prompt ("right altitude"), aligned with mem_paper/tech.tex.
        system_prompt = (
            "<background_information>\n"
            "- You are an AI assistant operating in a tool-enabled environment.\n"
            "- Context is a finite resource; keep it tight and high-signal.\n"
            "</background_information>\n\n"
            "<instructions>\n"
            "- Use tools when appropriate.\n"
            "- Treat tool outputs (especially structured JSON) as source of truth.\n"
            "- Prefer just-in-time retrieval over loading large text.\n"
            "- If information is missing or ambiguous, ask a clarifying question; do not guess.\n"
            "- For forecasting/prediction tools, never confuse input table with output table.\n"
            "  Always refer to the exact 'predicted_results_table' (or equivalent) returned by tools.\n"
            "</instructions>\n\n"
            "## Tool guidance\n"
            "- Keep tool calls token-efficient: request only needed fields, use limits/pagination when available.\n"
            "- If a tool result is large, summarize key facts and keep references (query/path/ids) instead of pasting raw output.\n\n"
            "## Output description\n"
            "- Answer directly and concisely; include next steps if the user needs to act.\n"
            "- When you cite tool-derived facts (tables/ids), use the exact values returned by tools.\n"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.agent, self.executor = build_agent_executor(
            self.llm,
            self.tools,
            prompt=prompt,
            system_prompt=system_prompt,
            max_iterations=20,
            verbose=self.verbose,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            executor_cls=FormatSafeAgentExecutor,
            return_agent=True,
        )

    # ------------------------------------------------------------------
    # Memory retrieval + compaction
    # ------------------------------------------------------------------
    def _retrieve_candidates(self, query: str) -> List[SearchResult]:
        # High recall: pull more, with reranking enabled when available.
        results_long = self.memory_manager.retrieve(
            query=query,
            top_k=self.memory_brief_cfg.recall_candidates,
            threshold=self.memory_brief_cfg.score_threshold,
            tags=None,
            rerank=True,
        )

        results: List[SearchResult] = list(results_long)
        if self.memory_brief_cfg.include_short_tier:
            try:
                results_short = self.memory_manager.retrieve_by_tier(
                    query=query,
                    tier="short",
                    top_k=max(10, self.memory_brief_cfg.recall_candidates // 2),
                    threshold=self.memory_brief_cfg.score_threshold,
                    rerank=True,
                )
                results.extend(list(results_short))
            except Exception:
                pass

        results = _dedupe_results(results)
        # Sort by rerank_score when present, else by score.
        results.sort(key=lambda r: (r.rerank_score if r.rerank_score is not None else r.score), reverse=True)
        return results

    def _llm_summarize_memory_brief(
        self,
        query: str,
        items: Sequence[SearchResult],
        *,
        max_brief_chars: int,
        max_item_chars: int,
    ) -> Optional[str]:
        # Build a strict summarization prompt.
        candidates_text = []
        for i, r in enumerate(items, start=1):
            candidates_text.append(
                f"[{i}] { _format_reference(r) }\n{_truncate(r.text, max_item_chars)}"
            )
        payload = "\n\n".join(candidates_text)

        prompt = (
            "You are a context engineer.\n"
            "Task: Given a user query and candidate memory snippets, produce a compact 'Memory Brief' "
            "that maximizes usefulness for answering the query while minimizing length.\n\n"
            "Hard rules:\n"
            "- Only use information present in the provided snippets.\n"
            "- If a snippet is irrelevant, omit it.\n"
            "- Prefer concrete facts, constraints, decisions, and user preferences.\n"
            "- Add a short References section listing the indices you used.\n\n"
            "Output format (plain text):\n"
            "Memory Brief:\n"
            "- <bullet>\n"
            "...\n"
            "References: [<idx>, ...]\n\n"
            f"User query:\n{query}\n\n"
            f"Candidate snippets:\n{payload}\n\n"
            f"Max length: {max_brief_chars} characters."
        )
        try:
            out = self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)
            text = getattr(out, "content", None) if out is not None else None
            if text is None and isinstance(out, dict):
                text = out.get("content")
            if text is None and isinstance(out, str):
                text = out
            text = (text or "").strip() if isinstance(text, str) else ""
            if not text:
                return None
            # Enforce max chars hard.
            return _truncate(text, max_brief_chars)
        except Exception as e:
            logger.debug("Memory brief LLM summarization failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Structured note-taking (NOTES/TODO/DECISIONS) per tech.tex
    # ------------------------------------------------------------------
    def _llm_extract_structured_notes(
        self,
        user_input: str,
        assistant_output: str,
        tool_facts: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.structured_notes_cfg.enabled:
            return None

        prompt = (
            "You are a structured note-taking engine for an AI agent.\n"
            "Extract high-signal items ONLY (avoid repetition).\n"
            "Output MUST be a single JSON object with these keys:\n"
            "{\n"
            "  \"decisions\": string[],\n"
            "  \"todos\": string[],\n"
            "  \"constraints\": string[],\n"
            "  \"preferences\": string[],\n"
            "  \"facts\": string[]\n"
            "}\n"
            "Rules:\n"
            "- If none, return empty arrays.\n"
            "- Use concise sentences, no markdown.\n"
            "- Do NOT invent facts beyond given text.\n\n"
            f"Tool facts (may be empty):\n{tool_facts.strip()}\n\n"
            f"User input:\n{_truncate(user_input, 1200)}\n\n"
            f"Assistant output:\n{_truncate(assistant_output, 2000)}\n"
        )
        try:
            out = self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)
            content = getattr(out, "content", None) if out is not None else None
            if content is None and isinstance(out, dict):
                content = out.get("content")
            if content is None and isinstance(out, str):
                content = out
            if not isinstance(content, str):
                return None
            data = json.loads(content)
            if not isinstance(data, dict):
                return None
            # Normalize arrays
            norm: Dict[str, List[str]] = {}
            for k in ("decisions", "todos", "constraints", "preferences", "facts"):
                v = data.get(k, [])
                if isinstance(v, list):
                    norm[k] = [str(x).strip() for x in v if str(x).strip()]
                else:
                    norm[k] = []
            return norm
        except Exception as e:
            logger.debug("Structured note extraction failed: %s", e)
            return None

    def _write_structured_notes(self, notes: Dict[str, List[str]]) -> None:
        if not notes or not self.structured_notes_cfg.enabled:
            return

        prev = getattr(self.memory_manager, "auto_classification_enabled", False)
        try:
            if prev:
                self.memory_manager.set_auto_classification_enabled(False)

            def add_items(items: List[str], tags: List[str], tier: str) -> None:
                for item in items[:10]:
                    self.memory_manager.add_memory(
                        text=item,
                        tags=["chat", "note", *tags],
                        tier=tier,
                        ttl_seconds=None,
                        extra_metadata={"session_id": self.session_id},
                    )

            add_items(notes.get("decisions", []), ["decisions"], "long")
            add_items(notes.get("constraints", []), ["constraints"], "long")
            add_items(notes.get("preferences", []), ["preferences"], "long")
            add_items(notes.get("facts", []), ["facts"], "long")
            add_items(notes.get("todos", []), ["todo"], "short")
        finally:
            if prev:
                self.memory_manager.set_auto_classification_enabled(True)

    def _write_tool_fact_note(self, tool_facts: str) -> None:
        """Persist authoritative tool facts as dedicated long-term notes.

        This reduces ambiguity and makes future retrieval higher precision than relying
        on full chat logs.
        """
        tool_facts = (tool_facts or "").strip()
        if not tool_facts:
            return
        prev = getattr(self.memory_manager, "auto_classification_enabled", False)
        try:
            if prev:
                self.memory_manager.set_auto_classification_enabled(False)
            self.memory_manager.add_memory(
                text=_truncate(tool_facts, 800),
                tags=["chat", "note", "facts", "tool_facts"],
                tier="long",
                ttl_seconds=None,
                extra_metadata={"session_id": self.session_id},
            )
        finally:
            if prev:
                self.memory_manager.set_auto_classification_enabled(True)

    def _retrieve_structured_notes(self, query: str) -> str:
        if not self.structured_notes_cfg.enabled:
            return ""

        sections: List[str] = []
        for tag, title in (
            (["decisions"], "Decisions"),
            (["constraints"], "Constraints"),
            (["todo"], "TODO"),
            (["preferences"], "Preferences"),
            (["facts"], "Facts"),
        ):
            try:
                results = self.memory_manager.retrieve(
                    query=query,
                    top_k=self.structured_notes_cfg.top_k_per_type,
                    threshold=self.memory_brief_cfg.score_threshold,
                    tags=tag,
                    rerank=True,
                )
                texts = [r.text for r in results if isinstance(r.text, str) and r.text.strip()]
                if texts:
                    bullets = "\n".join(f"- {_truncate(t, 300)}" for t in texts)
                    sections.append(f"{title}:\n{bullets}")
            except Exception:
                continue

        return _truncate("\n\n".join(sections), self.structured_notes_cfg.max_notes_chars)

    def _load_session_summary(self) -> str:
        """Load compacted session summary from short-term memory (if present)."""
        key = f"SESSION_SUMMARY::{self.session_id}"
        try:
            # Retrieve using vector search on a stable key string.
            results = self.memory_manager.retrieve_by_tier(
                query=key,
                tier="short",
                top_k=3,
                threshold=self.memory_brief_cfg.score_threshold,
                rerank=True,
            )
            for r in results:
                if isinstance(r.text, str) and key in r.text:
                    # Keep only the summary content after the key line.
                    return r.text.split("\n", 1)[1].strip() if "\n" in r.text else ""
        except Exception:
            return ""
        return ""

    def _write_session_summary(self, summary: str) -> None:
        """Persist session summary as short-term memory (note-taking / compaction)."""
        summary = (summary or "").strip()
        if not summary:
            return

        key = f"SESSION_SUMMARY::{self.session_id}"
        text = key + "\n" + _truncate(summary, 2500)

        # Enforce tier to 'short' by disabling auto-classification for this write.
        prev = getattr(self.memory_manager, "auto_classification_enabled", False)
        try:
            if prev:
                self.memory_manager.set_auto_classification_enabled(False)
            self.memory_manager.add_memory(
                text=text,
                tags=["chat", "session_summary", "compaction"],
                tier="short",
                ttl_seconds=None,
                extra_metadata={"session_id": self.session_id},
            )
        finally:
            if prev:
                self.memory_manager.set_auto_classification_enabled(True)

    def _llm_compact_session_summary(self, previous: str, turn_summary: str) -> Optional[str]:
        """Compaction: merge previous session summary + latest turn summary into a tight summary."""
        prompt = (
            "You are a context compaction engine.\n"
            "Goal: maintain a high-fidelity but compact session summary for a long-horizon task.\n"
            "Keep: goals, key decisions, constraints, unresolved issues, important tool facts.\n"
            "Drop: raw tool outputs, repetition, irrelevant chatter.\n"
            "Output: plain text, max 2000 characters.\n\n"
            f"Previous session summary (may be empty):\n{previous.strip()}\n\n"
            f"Latest turn summary:\n{turn_summary.strip()}\n\n"
            "Write the updated session summary:"  # instruction anchor
        )
        try:
            out = self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)
            text = getattr(out, "content", None) if out is not None else None
            if text is None and isinstance(out, dict):
                text = out.get("content")
            if text is None and isinstance(out, str):
                text = out
            text = (text or "").strip() if isinstance(text, str) else ""
            return _truncate(text, 2000) if text else None
        except Exception as e:
            logger.debug("Session compaction failed: %s", e)
            return None

    def _fallback_memory_brief(self, items: Sequence[SearchResult], *, max_brief_chars: int, max_item_chars: int) -> str:
        used = items[: max(self.memory_brief_cfg.rerank_k, 1)]
        lines = ["Memory Brief:"]
        refs: List[int] = []
        for idx, r in enumerate(used, start=1):
            refs.append(idx)
            cleaned = (r.text or "").replace("\n", " ").replace("\\n", " ")
            lines.append(f"- ({idx}) {_truncate(cleaned, max_item_chars)}")
        lines.append(f"References: {refs}")
        return _truncate("\n".join(lines), max_brief_chars)

    def _build_memory_brief(self, query: str, *, max_brief_chars: int) -> Tuple[str, List[SearchResult]]:
        candidates = self._retrieve_candidates(query)
        n = max(1, min(self.memory_brief_cfg.brief_candidates, len(candidates)))
        top_for_brief = candidates[:n]
        # Scale per-item truncation with remaining budget.
        max_item_chars = min(self.memory_brief_cfg.max_item_chars, max(200, max_brief_chars // max(1, n)))
        brief = self._llm_summarize_memory_brief(
            query,
            top_for_brief,
            max_brief_chars=max_brief_chars,
            max_item_chars=max_item_chars,
        )
        if brief is None:
            brief = self._fallback_memory_brief(
                top_for_brief,
                max_brief_chars=max_brief_chars,
                max_item_chars=max_item_chars,
            )
        return brief, candidates

    def _sanitize_for_long_term_memory(self, text: str) -> str:
        """Remove low-signal repeated tool result blocks from stored chat logs.

        We persist authoritative tool facts separately as dedicated notes; keeping
        repeated raw blocks in long-term chat logs tends to pollute retrieval.
        """
        if not text:
            return ""
        s = str(text)
        # Remove repeated tool result suffix blocks.
        s = re.sub(r"\n\n\[Tool Result\].*", "", s, flags=re.DOTALL)
        return s.strip()

    def _write_turn_summary(self, user_input: str, assistant_output: str) -> None:
        # Persist a minimal short-term summary; helps long-horizon continuity.
        try:
            text = f"Turn summary. User: {_truncate(user_input, 500)} | Assistant: {_truncate(assistant_output, 700)}"
            prev = getattr(self.memory_manager, "auto_classification_enabled", False)
            try:
                if prev:
                    self.memory_manager.set_auto_classification_enabled(False)
                self.memory_manager.add_memory(
                    text=text,
                    tags=["chat", "turn_summary", "compaction"],
                    tier="short",
                    ttl_seconds=None,
                    extra_metadata={"session_id": self.session_id},
                )
            finally:
                if prev:
                    self.memory_manager.set_auto_classification_enabled(True)
        except Exception as e:
            logger.debug("Failed to write turn summary: %s", e)

    def _update_long_term_memory(self, user_input: str, assistant_output: str) -> None:
        # Reduce context pollution: store a bounded-size interaction and rely on brief+summaries for long horizon.
        sanitized = self._sanitize_for_long_term_memory(str(assistant_output))
        bounded_output = _truncate(sanitized, 6000)
        self.memory_manager.add_interaction(
            user_input=_truncate(user_input, 2000),
            assistant_output=bounded_output,
            tags=["chat", "conversation"],
            tier="long",
        )

    def _build_context_pack(self, user_input: str, memory_brief: str, session_summary: str, tool_facts: str) -> ContextPack:
        task = f"User question: {user_input.strip()}"
        working_set = tool_facts.strip()
        memory_notes = self._retrieve_structured_notes(user_input)
        tool_guidance = (
            "- Use tools when they can fetch/verify facts.\n"
            "- Prefer small, targeted tool calls (limit, specific fields).\n"
            "- If tool output is large, summarize key facts and keep references; do not paste everything.\n"
        )
        pack = ContextPack(
            task=task,
            tool_guidance=tool_guidance,
            working_set=working_set,
            session_summary=session_summary.strip(),
            memory_notes=memory_notes,
            memory_brief=memory_brief.strip(),
        )
        return pack.with_dynamic_budgets(user_input)

    # ------------------------------------------------------------------
    # Public chat API (same command surface as Mem0HANARAGAgent)
    # ------------------------------------------------------------------
    def clear_long_term_memory(self) -> None:
        """Delete all stored memories in the configured vector table."""
        try:
            self.memory_manager.clear_all()
            logger.info("Mem0 long-term memory cleared.")
        except Exception as e:
            logger.error("Failed to clear long-term memory: %s", e)

    def chat(self, user_input: str) -> str:
        """Process user input and return agent response."""
        total_steps = 7
        self._emit_progress("Preparing context", step=1, total_steps=total_steps)

        # --- Compatibility commands ---
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
                results = self.memory_manager.retrieve_by_tier(
                    query=q,
                    tier="short",
                    top_k=self.memory_brief_cfg.rerank_k,
                    threshold=self.memory_brief_cfg.score_threshold,
                    rerank=True,
                )
                texts = [r.text for r in results]
                return f"Short-term hits ({len(texts)}):\n" + "\n".join(texts)
            except Exception as e:
                return f"search_short failed: {e}"

        if user_input.startswith("!search_long "):
            try:
                _, q = user_input.split(maxsplit=1)
                results = self.memory_manager.retrieve_by_tier(
                    query=q,
                    tier="long",
                    top_k=self.memory_brief_cfg.rerank_k,
                    threshold=self.memory_brief_cfg.score_threshold,
                    rerank=True,
                )
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

        # --- Context engineering: budget closed-loop ---
        self._emit_progress("Loading session summary", step=2, total_steps=total_steps)
        session_summary = self._load_session_summary()

        # Build a skeleton pack first (without brief) to compute remaining budget.
        skeleton = ContextPack(
            task=f"User question: {user_input.strip()}",
            tool_guidance=(
                "- Use tools when they can fetch/verify facts.\n"
                "- Prefer small, targeted tool calls (limit, specific fields).\n"
                "- If tool output is large, summarize key facts and keep references; do not paste everything.\n"
            ),
            working_set="",
            session_summary=session_summary.strip(),
            memory_notes=self._retrieve_structured_notes(user_input),
            memory_brief="",
        ).with_dynamic_budgets(user_input)

        # Determine max chars for memory brief given remaining budget.
        max_brief_chars = min(self.memory_brief_cfg.max_brief_chars, skeleton.remaining_budget_for_brief())
        if max_brief_chars < 300:
            max_brief_chars = 300

        self._emit_progress("Retrieving memory candidates", step=3, total_steps=total_steps)
        memory_brief, candidates = self._build_memory_brief(user_input, max_brief_chars=max_brief_chars)

        # Working set: high-priority tool facts from memory (avoid table-name confusion).
        candidate_texts = [c.text for c in candidates[: max(self.memory_brief_cfg.brief_candidates, 1)]]
        remembered_facts = _extract_last_predicted_table_from_texts(candidate_texts)
        tool_facts = ""
        if remembered_facts.get("predicted_results_table"):
            tool_facts = f"Known tool facts: predicted_results_table={remembered_facts['predicted_results_table']}"
            if remembered_facts.get("input_predict_table"):
                tool_facts += f", input_predict_table={remembered_facts['input_predict_table']}"

        pack = self._build_context_pack(
            user_input=user_input,
            memory_brief=memory_brief,
            session_summary=session_summary,
            tool_facts=tool_facts,
        )
        prompt_text = pack.render()

        self._emit_progress("Invoking tool-using agent", step=4, total_steps=total_steps)
        agent_input = {"input": [{"type": "text", "text": prompt_text}]}
        response = self.executor.invoke(agent_input)
        output = response.get("output") if isinstance(response, dict) else str(response)

        intermediate_steps = response.get("intermediate_steps") if isinstance(response, dict) else None
        facts = _extract_table_facts_from_steps(intermediate_steps)

        # Append authoritative tool facts to reduce table-name ambiguity across turns.
        if facts.get("predicted_results_table"):
            input_table = facts.get("input_predict_table")
            predicted_table = facts.get("predicted_results_table")
            suffix = f"\n\n[Tool Result] Predicted output table: {predicted_table}"
            if input_table:
                suffix += f" (input table: {input_table})"
            if suffix not in output:
                output = f"{output}{suffix}"

        # Persist memories (high recall + long-horizon continuity)
        self._emit_progress("Persisting memories", step=5, total_steps=total_steps)
        self._update_long_term_memory(user_input, output)
        self._write_turn_summary(user_input, output)

        # Structured note-taking (decisions/todos/constraints/preferences/facts)
        extracted = self._llm_extract_structured_notes(user_input, output, tool_facts)
        if extracted:
            self._write_structured_notes(extracted)

        # Persist tool facts as dedicated notes (high precision)
        self._write_tool_fact_note(tool_facts)

        # Update session summary (compaction)
        self._emit_progress("Compacting session summary", step=6, total_steps=total_steps)
        prev_summary = session_summary
        turn_summary = f"User: {_truncate(user_input, 500)}\nAssistant: {_truncate(output, 900)}"
        compacted = self._llm_compact_session_summary(prev_summary, turn_summary)
        if compacted:
            self._write_session_summary(compacted)

        self._emit_progress("Done", step=7, total_steps=total_steps)

        return output
