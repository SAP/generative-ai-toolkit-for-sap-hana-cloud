"""Context-engineered agent (file-based, no HANA/Mem0 dependencies).

This agent implements the pipeline described in mem_paper/tech.tex:

- ContextPack layering + budgeting
- Hybrid retrieval (fast lexical BM25/grep) over Markdown memory
- Just-in-time style: store references + compact facts instead of raw dumps
- Session compaction + structured notes (NOTES/TODO/DECISIONS)

Storage
-------
All memory is persisted as Markdown files under a storage directory.

Notes
-----
This module intentionally avoids importing mem0/hana_ml. It is designed to be
LLM-provider agnostic by accepting an llm callable or object with .invoke().
"""

# pylint: disable=bad-indentation,trailing-newlines

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import json
import math
import re
import sys

from hana_ai.langchain_compat import (
	ChatPromptTemplate,
	FormatSafeAgentExecutor,
	HumanMessagePromptTemplate,
	MessagesPlaceholder,
	SystemMessage,
	build_agent_executor,
)


def _utc_now_iso() -> str:
	return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_text(x: Any) -> str:
	if x is None:
		return ""
	return str(x)


def _truncate(text: str, limit: int) -> str:
	s = _safe_text(text)
	return s if len(s) <= limit else (s[: max(0, limit - 3)] + "...")


def _tokenize(text: str) -> List[str]:
	s = _safe_text(text).lower()
	tokens = re.split(r"[^a-z0-9_]+", s)
	return [t for t in tokens if len(t) >= 2]


def _ensure_dir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


def _atomic_append(path: Path, content: str) -> None:
	_ensure_dir(path.parent)
	with path.open("a", encoding="utf-8") as f:
		f.write(content)


@dataclass(frozen=True)
class Tool:
	"""Minimal tool interface (dependency-free)."""

	name: str
	description: str
	func: Callable[[Dict[str, Any]], Any]


@dataclass
class RetrievalChunk:
	"""A retrieved Markdown chunk with provenance and score."""

	text: str
	source: str
	score: float


class _BM25Index:
	"""Lightweight BM25 index for small/medium Markdown corpora."""

	def __init__(self) -> None:
		self._docs: List[Tuple[str, Dict[str, Any]]] = []
		self._doc_tokens: List[List[str]] = []
		self._df: Dict[str, int] = {}
		self._avgdl: float = 0.0
		self._version: int = 0

	@property
	def version(self) -> int:
		"""Monotonic version counter updated on rebuilds."""
		return self._version

	def clear(self) -> None:
		"""Clear all indexed documents and statistics."""
		self._docs = []
		self._doc_tokens = []
		self._df = {}
		self._avgdl = 0.0
		self._version += 1

	def add_documents(self, docs: Sequence[Tuple[str, Dict[str, Any]]]) -> None:
		"""Add documents as (text, metadata) pairs."""
		if not docs:
			return
		for text, meta in docs:
			tokens = _tokenize(text)
			self._docs.append((text, meta))
			self._doc_tokens.append(tokens)
			seen = set(tokens)
			for t in seen:
				self._df[t] = self._df.get(t, 0) + 1

		total_len = sum(len(toks) for toks in self._doc_tokens)
		self._avgdl = total_len / max(1, len(self._doc_tokens))
		self._version += 1

	def search(self, query: str, *, top_k: int = 8) -> List[RetrievalChunk]:
		"""Search documents using BM25 scoring."""
		q_tokens = _tokenize(query)
		if not q_tokens or not self._docs:
			return []

		k1 = 1.2
		b = 0.75
		n_docs = len(self._docs)

		idf: Dict[str, float] = {}
		for t in set(q_tokens):
			df = self._df.get(t, 0)
			idf[t] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

		scored: List[Tuple[float, int]] = []
		raw_q = _safe_text(query).lower().strip()
		for i, doc_tokens in enumerate(self._doc_tokens):
			if not doc_tokens:
				continue
			dl = len(doc_tokens)
			tf: Dict[str, int] = {}
			for t in doc_tokens:
				tf[t] = tf.get(t, 0) + 1

			score = 0.0
			for t in q_tokens:
				if t not in tf:
					continue
				freq = tf[t]
				denom = freq + k1 * (1.0 - b + b * (dl / max(1e-9, self._avgdl)))
				score += idf.get(t, 0.0) * ((freq * (k1 + 1.0)) / max(1e-9, denom))

			if raw_q:
				raw_doc = self._docs[i][0].lower()
				if raw_q in raw_doc:
					score *= 1.25

			if score > 0:
				scored.append((score, i))

		scored.sort(key=lambda x: x[0], reverse=True)
		out: List[RetrievalChunk] = []
		for score, idx in scored[: max(1, top_k)]:
			text, meta = self._docs[idx]
			source = _safe_text(meta.get("source", ""))
			out.append(RetrievalChunk(text=text, source=source, score=float(score)))
		return out


@dataclass
class ContextBudgets:
	"""Character budgets per context layer."""

	max_context_chars: int = 9000
	budget_system: int = 1700
	budget_task: int = 1200
	budget_tool_guidance: int = 900
	budget_working_set: int = 1000
	budget_session_summary: int = 1800
	budget_memory_notes: int = 2200
	budget_retrieved: int = 2200


@dataclass
class ContextPack:
	"""Layered context payload rendered into a single prompt string."""

	system: str
	task: str
	tool_guidance: str
	working_set: str
	session_summary: str
	memory_notes: str
	retrieved: str
	budgets: ContextBudgets

	def render(self) -> str:
		"""Render the layered context (tech.tex: explicit sections + truncation)."""
		b = self.budgets
		parts: List[str] = []
		if self.system.strip():
			parts.append("<system>\n" + _truncate(self.system.strip(), b.budget_system) + "\n</system>")
		parts.append("<task>\n" + _truncate(self.task.strip(), b.budget_task) + "\n</task>")
		if self.tool_guidance.strip():
			parts.append(
				"<tool_guidance>\n" + _truncate(self.tool_guidance.strip(), b.budget_tool_guidance) + "\n</tool_guidance>"
			)
		if self.working_set.strip():
			parts.append("<working_set>\n" + _truncate(self.working_set.strip(), b.budget_working_set) + "\n</working_set>")
		if self.session_summary.strip():
			parts.append(
				"<session_summary>\n"
				+ _truncate(self.session_summary.strip(), b.budget_session_summary)
				+ "\n</session_summary>"
			)
		if self.memory_notes.strip():
			parts.append(
				"<memory_notes>\n" + _truncate(self.memory_notes.strip(), b.budget_memory_notes) + "\n</memory_notes>"
			)
		if self.retrieved.strip():
			parts.append("<retrieved>\n" + _truncate(self.retrieved.strip(), b.budget_retrieved) + "\n</retrieved>")

		rendered = "\n\n".join(parts)
		return _truncate(rendered, b.max_context_chars)


@dataclass
class AgentConfig:
	"""Configuration knobs for retrieval, compaction, and persistence."""

	budgets: ContextBudgets = field(default_factory=ContextBudgets)
	max_tool_iterations: int = 8
	retrieval_top_k: int = 8
	retrieval_chunk_chars: int = 900
	retrieval_min_query_len: int = 6
	enable_retrieval: bool = True
	enable_session_summary: bool = True
	enable_compaction: bool = True
	compaction_trigger_chars: int = 22000
	keep_recent_chat_chars: int = 6000
	enable_structured_notes: bool = True
	notes_max_items_per_turn: int = 8


class ContextAgent:
	"""Context-engineered agent backed by Markdown memory."""

	def __init__(
		self,
		llm: Any,
		tools: Optional[Sequence[Tool]] = None,
		*,
		storage_dir: str = ".context_agent",
		session_id: str = "global_session",
		config: Optional[AgentConfig] = None,
		progress_bar: bool = False,
		progress_callback: Optional[Callable[[str], None]] = None,
	) -> None:
		self.llm = llm
		self.tools: List[Tool] = list(tools or [])
		self.session_id = session_id
		self.config = config or AgentConfig()

		self.storage_dir = Path(storage_dir).expanduser().resolve()
		self.session_dir = self.storage_dir / "sessions" / self.session_id

		self.progress_bar = bool(progress_bar)
		self.progress_callback = progress_callback
		if self.progress_bar and self.progress_callback is None:
			self.progress_callback = self._default_progress_printer

		self._index = _BM25Index()
		self._indexed_mtimes: Dict[Path, float] = {}
		self._executor = None

		self._init_storage()
		self._initialize_executor()

	def _initialize_executor(self) -> None:
		"""Create an OpenAI-native tool calling agent executor.

		Tool selection is delegated to the model via function/tool calling.
		"""
		if not self.tools:
			self._executor = None
			return
		system_prompt = self._system_prompt()
		prompt = ChatPromptTemplate.from_messages(
			[
				SystemMessage(content=system_prompt),
				HumanMessagePromptTemplate.from_template("{input}"),
				MessagesPlaceholder(variable_name="agent_scratchpad"),
			]
		)
		self._executor = build_agent_executor(
			self.llm,
			list(self.tools),
			prompt=prompt,
			system_prompt=system_prompt,
			verbose=False,
			max_iterations=self.config.max_tool_iterations,
			handle_parsing_errors=True,
			return_intermediate_steps=True,
			executor_cls=FormatSafeAgentExecutor,
		)

	def _default_progress_printer(self, text: str) -> None:
		try:
			sys.stdout.write(str(text).rstrip() + "\n")
			sys.stdout.flush()
		except Exception:
			pass

	def _emit_progress(self, message: str, *, step: Optional[int] = None, total_steps: Optional[int] = None) -> None:
		prefix = ""
		if step is not None and total_steps is not None and total_steps > 0:
			prefix = f"[{step}/{total_steps}] "
		text = (prefix + message).strip()
		cb = self.progress_callback
		if cb is not None:
			try:
				cb(text)
			except Exception:
				pass

	# ---------------- Storage ----------------
	def _init_storage(self) -> None:
		_ensure_dir(self.session_dir)
		_ensure_dir(self.storage_dir)
		for name in ("NOTES.md", "TODO.md", "DECISIONS.md", "CONTEXT.md"):
			p = self.storage_dir / name
			if not p.exists():
				_atomic_append(p, f"# {name.replace('.md','')}\n\n")
		for name in ("chat.md", "session_summary.md"):
			p = self.session_dir / name
			if not p.exists():
				_atomic_append(p, f"# Session {self.session_id}\n\n")

	def _chat_path(self) -> Path:
		return self.session_dir / "chat.md"

	def _summary_path(self) -> Path:
		return self.session_dir / "session_summary.md"

	def _append_chat(self, role: str, content: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
		ts = _utc_now_iso()
		meta = f"ts={ts}"
		if extra:
			meta += " " + " ".join(f"{k}={_safe_text(v)}" for k, v in extra.items())
		block = f"## {role} ({meta})\n\n{content.strip()}\n\n"
		_atomic_append(self._chat_path(), block)

	def _read_text(self, path: Path, *, limit_chars: Optional[int] = None) -> str:
		try:
			s = path.read_text(encoding="utf-8")
		except Exception:
			return ""
		if limit_chars is None:
			return s
		return s[-limit_chars:]

	# -------------- Retrieval --------------
	def _memory_files(self) -> List[Path]:
		files: List[Path] = [
			self.storage_dir / "NOTES.md",
			self.storage_dir / "TODO.md",
			self.storage_dir / "DECISIONS.md",
			self.storage_dir / "CONTEXT.md",
			self._chat_path(),
			self._summary_path(),
		]
		return [p for p in files if p.exists()]

	def _split_markdown(self, text: str, *, max_chars: int) -> List[str]:
		s = _safe_text(text)
		parts = re.split(r"\n(?=#+\s)", s)
		chunks: List[str] = []
		for part in parts:
			part = part.strip()
			if not part:
				continue
			if len(part) <= max_chars:
				chunks.append(part)
				continue
			paras = re.split(r"\n\s*\n", part)
			buf = ""
			for para in paras:
				para = para.strip()
				if not para:
					continue
				if not buf:
					buf = para
				elif len(buf) + 2 + len(para) <= max_chars:
					buf = buf + "\n\n" + para
				else:
					chunks.append(buf)
					buf = para
			if buf:
				chunks.append(buf)
		return chunks

	def _rebuild_index_if_needed(self) -> None:
		changed = False
		for p in self._memory_files():
			try:
				mtime = p.stat().st_mtime
			except Exception:
				continue
			if self._indexed_mtimes.get(p) != mtime:
				self._indexed_mtimes[p] = mtime
				changed = True

		if not changed and self._index.version > 0:
			return

		docs: List[Tuple[str, Dict[str, Any]]] = []
		self._index.clear()
		for p in self._memory_files():
			text = self._read_text(p)
			if not text.strip():
				continue
			chunks = self._split_markdown(text, max_chars=self.config.retrieval_chunk_chars)
			for i, ch in enumerate(chunks):
				docs.append((ch, {"source": f"{p.name}#chunk{i}"}))
		self._index.add_documents(docs)

	def _retrieve(self, query: str) -> List[RetrievalChunk]:
		if not self.config.enable_retrieval:
			return []
		if len(_safe_text(query).strip()) < self.config.retrieval_min_query_len:
			return []
		self._rebuild_index_if_needed()
		return self._index.search(query, top_k=self.config.retrieval_top_k)

	# -------------- LLM / Tools --------------
	def _llm_call(self, prompt: str) -> str:
		model = self.llm
		if hasattr(model, "invoke"):
			out = model.invoke(prompt)
			content = getattr(out, "content", None)
			if isinstance(content, str) and content.strip():
				return content
			if isinstance(out, str):
				return out
			return _safe_text(out)
		if callable(model):
			return _safe_text(model(prompt))
		return _safe_text(model)

	def _tools_schema_text(self) -> str:
		if not self.tools:
			return "(no tools)"
		lines: List[str] = []
		for t in self.tools:
			name = _safe_text(getattr(t, "name", ""))
			desc = _safe_text(getattr(t, "description", ""))
			lines.append(f"- {name}: {desc}".strip())
		return "\n".join(lines)

	def _execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
		for t in self.tools:
			tool_name = _safe_text(getattr(t, "name", "")).strip()
			if tool_name != name:
				continue

			# 1) Our minimal Tool dataclass
			if hasattr(t, "func") and callable(getattr(t, "func")):
				return t.func(args)  # type: ignore[attr-defined]

			# 2) LangChain-style tools (BaseTool.invoke)
			if hasattr(t, "invoke") and callable(getattr(t, "invoke")):
				return t.invoke(args)  # type: ignore[attr-defined]

			# 3) Fallback: try call conventions
			if callable(t):
				try:
					return t(**args)  # type: ignore[misc]
				except TypeError:
					return t(args)  # type: ignore[misc]

			raise TypeError(f"Tool '{name}' is not callable")
		raise KeyError(f"Unknown tool: {name}")

	# -------------- Notes / Compaction --------------
	def _load_session_summary(self) -> str:
		if not self.config.enable_session_summary:
			return ""
		return self._read_text(self._summary_path(), limit_chars=self.config.budgets.budget_session_summary)

	def _write_session_summary(self, text: str) -> None:
		p = self._summary_path()
		_ensure_dir(p.parent)
		p.write_text(_safe_text(text).strip() + "\n", encoding="utf-8")

	def _maybe_compact(self) -> None:
		if not (self.config.enable_compaction and self.config.enable_session_summary):
			return
		chat_text = self._read_text(self._chat_path())
		if len(chat_text) < self.config.compaction_trigger_chars:
			return
		prompt = (
			"You are a context compaction engine.\n"
			"Goal: compress the session trace into a concise, high-fidelity continuation summary.\n"
			"Keep: goals, key decisions, constraints, unresolved issues, important tool facts.\n"
			"Drop: raw tool outputs, repetition, irrelevant chatter.\n"
			"Output sections (plain text):\n"
			"- Goal\n- Current state\n- Decisions\n- Open issues\n- Next steps (ordered)\n- References\n"
			f"Max length: {self.config.budgets.budget_session_summary} characters.\n\n"
			"Session trace:\n"
			+ _truncate(chat_text, 26000)
		)
		summary = _truncate(self._llm_call(prompt), self.config.budgets.budget_session_summary)
		if summary.strip():
			self._write_session_summary(summary)
			tail = chat_text[-self.config.keep_recent_chat_chars :]
			self._chat_path().write_text("# Session " + self.session_id + "\n\n" + tail, encoding="utf-8")

	def _extract_structured_notes(self, user_input: str, assistant_output: str) -> Dict[str, List[str]]:
		if not self.config.enable_structured_notes:
			return {"decisions": [], "todos": [], "constraints": [], "preferences": [], "facts": []}
		prompt = (
			"You are a structured note-taking engine for an AI agent.\n"
			"Extract high-signal items ONLY (avoid repetition).\n"
			"Output MUST be a single JSON object with keys:\n"
			"{\"decisions\": string[], \"todos\": string[], \"constraints\": string[], \"preferences\": string[], \"facts\": string[]}\n"
			"Rules: if none, return empty arrays; no markdown; do not invent facts.\n\n"
			f"User input:\n{_truncate(user_input, 1200)}\n\n"
			f"Assistant output:\n{_truncate(assistant_output, 2000)}\n"
		)
		raw = self._llm_call(prompt)
		try:
			obj = json.loads(raw)
		except Exception:
			return {"decisions": [], "todos": [], "constraints": [], "preferences": [], "facts": []}
		out: Dict[str, List[str]] = {}
		for k in ("decisions", "todos", "constraints", "preferences", "facts"):
			v = obj.get(k, []) if isinstance(obj, dict) else []
			if isinstance(v, list):
				items = [str(x).strip() for x in v if str(x).strip()]
			else:
				items = []
			out[k] = items[: self.config.notes_max_items_per_turn]
		return out

	def _append_notes(self, notes: Dict[str, List[str]]) -> None:
		ts = _utc_now_iso()
		if notes.get("facts") or notes.get("constraints") or notes.get("preferences"):
			p = self.storage_dir / "NOTES.md"
			lines = [f"## {ts}"]
			for k, title in (("facts", "Facts"), ("constraints", "Constraints"), ("preferences", "Preferences")):
				items = notes.get(k, [])
				if items:
					lines.append(f"### {title}")
					lines.extend([f"- {i}" for i in items])
			_atomic_append(p, "\n".join(lines) + "\n\n")

		if notes.get("todos"):
			p = self.storage_dir / "TODO.md"
			lines = [f"## {ts}"]
			lines.extend([f"- [ ] {i}" for i in notes["todos"]])
			_atomic_append(p, "\n".join(lines) + "\n\n")

		if notes.get("decisions"):
			p = self.storage_dir / "DECISIONS.md"
			lines = [f"## {ts}"]
			lines.extend([f"- {i}" for i in notes["decisions"]])
			_atomic_append(p, "\n".join(lines) + "\n\n")

	# -------------- Context building --------------
	def _system_prompt(self) -> str:
		"""Right-altitude system instructions (tech.tex)."""
		return (
			"<background_information>\n"
			"- You are a tool-enabled assistant.\n"
			"- Context is a finite resource; keep it tight and high-signal.\n"
			"</background_information>\n\n"
			"<instructions>\n"
			"- Prefer just-in-time retrieval over loading large text.\n"
			"- If information is missing or ambiguous, ask a clarifying question; do not guess.\n"
			"- Use tools when the request requires external facts/actions (DB queries, stats checks, training, prediction, plots, artifact generation).\n"
			"- It is OK to answer directly (no tool) for purely conceptual questions that do not depend on external data.\n"
			"- Never fabricate table rows, metrics, or model results. If asked for data/results, call the appropriate tool.\n"
			"</instructions>"
		)

	def _tool_guidance(self) -> str:
		if not self.tools:
			return "(No tools available.)"
		return (
			"Tool catalog:\n"
			+ self._tools_schema_text()
			+ "\n\n"
			"Decision policy (when to call tools):\n"
			"- Call a tool if the user requests: table records, statistics checks (e.g., ts_check), dataset reports, training/fitting, prediction/forecasting, plotting, or artifact generation.\n"
			"- Ask a clarifying question before calling a tool if required inputs are missing (e.g., table name, key column, target column).\n\n"
			"Tool usage:\n"
			"- Use native tool/function calling to execute tools.\n"
			"- After receiving a tool result, continue reasoning and answer the user.\n"
			"- Keep tool results token-efficient: summarize key facts + keep references.\n"
			"- In your final answer, include the most important tool outputs (briefly)."
		)

	def _build_context(self, user_input: str) -> ContextPack:
		"""Build a ContextPack for the current turn (tech.tex pipeline)."""
		retrieved_chunks = self._retrieve(user_input) if self.config.enable_retrieval else []
		retrieved_text = ""
		if retrieved_chunks:
			lines = ["Retrieved snippets:"]
			for i, ch in enumerate(retrieved_chunks, start=1):
				lines.append(f"[{i}] source={ch.source} score={ch.score:.3f}\n{_truncate(ch.text, 700)}")
			retrieved_text = "\n\n".join(lines)

		session_summary = self._load_session_summary() if self.config.enable_session_summary else ""
		memory_notes = self._read_text(self.storage_dir / "NOTES.md", limit_chars=1600)

		return ContextPack(
			system=self._system_prompt(),
			task=f"User question: {user_input.strip()}",
			tool_guidance=self._tool_guidance(),
			working_set="",
			session_summary=session_summary.strip(),
			memory_notes=_truncate(memory_notes.strip(), self.config.budgets.budget_memory_notes),
			retrieved=_truncate(retrieved_text.strip(), self.config.budgets.budget_retrieved),
			budgets=self.config.budgets,
		)

	# -------------- Public API --------------
	def chat(self, user_input: str) -> str:
		"""Run one conversational turn and persist it to Markdown memory."""
		total_steps = 6
		self._emit_progress("Building context", step=1, total_steps=total_steps)
		pack = self._build_context(user_input)
		prompt = pack.render()
		self._append_chat("user", user_input)

		self._emit_progress("LLM reasoning", step=2, total_steps=total_steps)
		tool_trace: List[str] = []
		tool_return_snippets: List[str] = []
		assistant_text = ""

		if not self.tools or self._executor is None:
			assistant_text = _safe_text(self._llm_call(prompt)).strip()
		else:
			self._emit_progress("Tool calling", step=3, total_steps=total_steps)
			result = self._executor.invoke({"input": prompt})
			assistant_text = _safe_text(result.get("output") if isinstance(result, dict) else result).strip()
			steps = result.get("intermediate_steps") if isinstance(result, dict) else None
			if isinstance(steps, list):
				for action, observation in steps:
					tool_name = _safe_text(getattr(action, "tool", "tool"))
					tool_args = getattr(action, "tool_input", {})
					try:
						args_text = json.dumps(tool_args, ensure_ascii=False)
					except Exception:
						args_text = _safe_text(tool_args)

					obs_text = ""
					if isinstance(observation, list):
						obs_text = "\n".join([_safe_text(x.get("text") if isinstance(x, dict) else x) for x in observation])
					else:
						obs_text = _safe_text(observation)

					tool_trace.append(f"### TOOL {tool_name} args={args_text}\n\n{_truncate(obs_text, 3000)}")
					tool_return_snippets.append(f"[Tool Return] {tool_name} args={args_text}\n{_truncate(obs_text, 1200)}")

			if tool_trace:
				self._append_chat("tool", "\n\n".join(tool_trace))
			if tool_return_snippets:
				assistant_text = assistant_text + "\n\n" + "\n\n".join(tool_return_snippets)

		self._emit_progress("Persisting memory", step=4, total_steps=total_steps)
		self._append_chat("assistant", assistant_text)

		if self.config.enable_structured_notes:
			self._emit_progress("Writing structured notes", step=5, total_steps=total_steps)
			notes = self._extract_structured_notes(user_input, assistant_text)
			self._append_notes(notes)

		if self.config.enable_compaction and self.config.enable_session_summary:
			self._emit_progress("Compaction check", step=6, total_steps=total_steps)
			self._maybe_compact()

		return assistant_text

