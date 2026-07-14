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
import threading
import time

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


def _sanitize_llm_sampling(llm: Any) -> Any:
	"""Drop ``top_p`` when ``temperature`` is also set on the LLM.

	SAP Cloud SDK for AI's ``init_llm`` always passes both ``temperature`` and
	``top_p`` to the underlying chat model. Newer Anthropic Claude models on
	Bedrock (e.g. ``anthropic--claude-4.6-sonnet``) reject this combination with
	``temperature and top_p cannot both be specified for this model``. We
	normalize on temperature here (deterministic-friendly, the parameter the
	user is far more likely to have set on purpose) by clearing ``top_p`` to
	``None`` when both are present. Plain callables and arbitrary objects pass
	through untouched.
	"""
	if llm is None:
		return llm
	try:
		has_temp = getattr(llm, "temperature", None) is not None
		has_top_p = getattr(llm, "top_p", None) is not None
	except Exception:
		return llm
	if has_temp and has_top_p:
		try:
			llm.top_p = None
		except Exception:
			# Pydantic v2 may guard against direct assignment; fall back to model_copy.
			try:
				llm = llm.model_copy(update={"top_p": None})
			except Exception:
				pass
	# Some Bedrock wrappers carry sampling parameters inside ``model_kwargs``
	# instead of on the model itself; clear ``top_p`` there as well when
	# ``temperature`` is set.
	try:
		model_kwargs = getattr(llm, "model_kwargs", None)
		if isinstance(model_kwargs, dict):
			if "temperature" in model_kwargs and model_kwargs.get("top_p") is not None:
				model_kwargs.pop("top_p", None)
	except Exception:
		pass
	return llm


def _reset_markdown_file(path: Path, heading: str) -> None:
	_ensure_dir(path.parent)
	path.write_text(f"# {heading}\n\n", encoding="utf-8")


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
	budget_skills: int = 1400
	budget_working_set: int = 1000
	budget_session_summary: int = 1800
	budget_memory_notes: int = 3000
	budget_retrieved: int = 2200


@dataclass
class ContextPack:
	"""Layered context payload rendered into a single prompt string."""

	system: str
	task: str
	tool_guidance: str
	skills: str
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
		if self.skills.strip():
			parts.append("<skills>\n" + _truncate(self.skills.strip(), b.budget_skills) + "\n</skills>")
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
	# Wall-clock ceiling (seconds) for a single ``chat()`` turn. Once the executor
	# has spent longer than this on tool calling, the current turn short-circuits
	# with an actionable failure message instead of letting HANA/PAL retries pile
	# up while the user watches the notebook cell hang. Set to 0 or None to
	# disable the deadline (previous behaviour). 300s ≈ five minutes and is safe
	# for the demo notebooks — a single fit/predict round trip is well under a
	# minute even on a warm connection, so eight iterations × ~30s stays under
	# the ceiling in the happy path.
	max_turn_seconds: Optional[float] = 300.0
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

	# Skills (tech.tex: examples/skills layer)
	enable_skills: bool = True
	max_active_skills: int = 2
	# True: let the LLM choose which skills to activate for the current request.
	# False: use deterministic keyword-based fallback routing, which is better for demos/tests.
	skills_use_llm_selector: bool = True
	skills_selector_max_chars: int = 1200
	skills_cache_turns: int = 3
	skills_markdown_path: Optional[str] = None


@dataclass(frozen=True)
class Skill:
	"""A reusable, high-signal workflow snippet injected into context."""

	name: str
	title: str
	description: str
	content: str


def _builtin_skills() -> Dict[str, Skill]:
	"""Built-in skills aligned with mem_paper/tech.tex."""
	return {
		"timeseries_forecasting": Skill(
			name="timeseries_forecasting",
			title="Time-series forecasting workflow",
			description="Analyze time series tables, suggest/train a model, predict, and plot.",
			content=(
				"Goal: reliably analyze a time series table, choose a forecasting model, train/save it, predict future, and summarize results.\n"
				"When to use tools: whenever you need table content, statistics, training, prediction, plotting.\n\n"
				"Playbook (minimal, high-signal):\n"
				"1) Confirm inputs: training table, key/time column, target(endog) column. If missing -> ask.\n"
				"2) Use TimeSeriesDatasetReport + TimeSeriesCheck to understand seasonality, missingness, frequency, outliers.\n"
				"3) Suggest model: prefer AutomaticTimeSeriesFitAndSave for single series; use AdditiveModel/Intermittent only if data suggests.\n"
				"4) Train: call *FitAndSave tool; record model name.\n"
				"5) Predict: build predict table if needed (TSMakeFutureTableTool), then call *LoadModelAndPredict.\n"
				"6) Evaluate: if you have actuals, call AccuracyMeasure (or compute via SQL join) and report MAD(≈MAE)/RMSE/MAPE (use tool-supported metric names).\n"
				"7) Plot: ForecastLinePlot when a visual is requested.\n\n"
				"Rules:\n"
				"- Never fabricate rows/metrics; always use tools.\n"
				"- Always refer to the exact output table names returned by tools (predicted_results_table, etc.).\n"
			),
		),
		"prediction_result_analysis": Skill(
			name="prediction_result_analysis",
			title="Prediction results analysis",
			description="Analyze forecast outputs; compare predicted vs actual using HANA SQL/AccuracyMeasure.",
			content=(
				"Goal: after a prediction tool produces an output table, analyze quality and provide actionable insights.\n\n"
				"Checklist:\n"
				"- Identify the prediction output table name (from tool output).\n"
				"- If an actual table exists, compare predicted vs actual on the key/time column.\n"
				"- Compute: MAD(≈MAE), RMSE, MAPE (or sMAPE), bias (mean error), and detect outliers.\n"
				"- Summarize: trend/seasonality fit, intervals (if available), obvious data issues (zeros, missing periods).\n\n"
				"Preferred tools:\n"
				"- AccuracyMeasure tool when compatible.\n"
				"- Otherwise, use SelectStatementToTableTool to create a comparison table via SQL join, then FetchDataTool to preview results.\n\n"
				"SQL pattern (conceptual):\n"
				"- Join predicted and actual by key/time; compute error columns; aggregate metrics.\n"
			),
		),
		"massive_forecast_comparison": Skill(
			name="massive_forecast_comparison",
			title="Massive forecasting comparison",
			description="When using Massive* tools, compare massive predictions against a baseline using SQL/accuracy measures.",
			content=(
				"Goal: for Massive* forecasting outputs (many series), validate and compare quality.\n\n"
				"Workflow:\n"
				"1) Use MassiveTimeSeriesCheck for dataset stats across series.\n"
				"2) Train/predict via MassiveAutomaticTimeSeriesFitAndSave + MassiveAutomaticTimeSeriesLoadModelAndPredict.\n"
				"3) If a non-massive baseline exists (single-series AutomaticTimeSeries* or additive), run it for a representative subset.\n"
				"4) Compare: use AccuracyMeasure where possible; else use SQL (SelectStatementToTableTool) to compute per-series and overall metrics.\n"
				"5) Report: which series perform poorly, distribution of errors, and recommended next action (outlier handling, re-train, different model family).\n\n"
				"Rules:\n"
				"- Keep comparisons reproducible: always include table names + key columns used.\n"
			),
		),
	}


def _parse_skills_markdown(text: str) -> Dict[str, Skill]:
	"""Parse markdown headings into Skill objects.

	A skill block starts at a heading like ``## skill_name`` where the heading text is
	lowercase_with_underscores. The block continues until the next heading of the same form.
	"""
	def _has_required_sections(block_text: str) -> bool:
		lines = [line.strip() for line in block_text.splitlines() if line.strip()]
		has_goal_header = any(line == "Goal:" for line in lines)
		has_goal_content = False
		for i, line in enumerate(lines):
			if line != "Goal:":
				continue
			for next_line in lines[i + 1 :]:
				if next_line.endswith(":") and not next_line.startswith("-"):
					break
				if next_line:
					has_goal_content = True
					break
			if has_goal_content:
				break
		return has_goal_header and has_goal_content

	s = _safe_text(text)
	pattern = re.compile(r"^## ([a-z0-9_]+)\s*$", re.MULTILINE)
	matches = list(pattern.finditer(s))
	if not matches:
		return {}

	parsed: Dict[str, Skill] = {}
	for i, match in enumerate(matches):
		name = match.group(1).strip()
		start = match.end()
		end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
		block = s[start:end].strip()
		if not block or not _has_required_sections(block):
			continue

		title = name.replace("_", " ").title()
		description = ""
		lines = [line.rstrip() for line in block.splitlines()]
		for index, raw_line in enumerate(lines):
			line = raw_line.strip()
			if not line or line.startswith("Status:"):
				continue
			if line.startswith("Goal:"):
				description = line[len("Goal:") :].strip()
				if description:
					break
				for next_raw in lines[index + 1 :]:
					next_line = next_raw.strip()
					if not next_line:
						continue
					if next_line.endswith(":") and not next_line.startswith("-"):
						break
					if next_line.startswith("- "):
						description = next_line[2:].strip()
						break
					description = next_line
					break
				break
			description = line
			break

		if not description:
			description = title

		parsed[name] = Skill(
			name=name,
			title=title,
			description=_truncate(description, 180),
			content=block,
		)
	return parsed


def _load_skills_from_markdown(path: Path) -> Dict[str, Skill]:
	"""Load skills from a markdown file, returning an empty mapping on failure."""
	try:
		text = path.read_text(encoding="utf-8")
	except Exception:
		return {}
	return _parse_skills_markdown(text)


def _safe_json_loads(text: str) -> Optional[Any]:
	try:
		return json.loads(text)
	except Exception:
		return None


def _extract_html_file_path(text: str) -> Optional[Path]:
	payload = _safe_json_loads(text)
	if isinstance(payload, dict):
		html_file = _safe_text(payload.get("html_file")).strip()
		if html_file:
			return Path(html_file).expanduser()
	match = re.search(r'"html_file"\s*:\s*"([^"]+\.html)"', _safe_text(text))
	if not match:
		return None
	return Path(match.group(1)).expanduser()


class ContextAgent:
	"""Context-engineered agent backed by Markdown memory.

	The agent persists long-lived and session-scoped state as Markdown files under
	``storage_dir`` and supports command-style memory maintenance through ``chat()``.

	Memory cleanup commands:
	- Aggregate commands:
	  - ``!clear_notes`` clears ``NOTES.md``, ``TODO.md``, ``DECISIONS.md``, and ``CONTEXT.md``.
	  - ``!clear_session`` clears the current session ``chat.md`` and ``session_summary.md``.
	  - ``!reset_memory`` clears both the global note files and the current session files.
	- File-level commands:
	  - ``!clear_notes_file`` clears only ``NOTES.md``.
	  - ``!clear_todo`` clears only ``TODO.md``.
	  - ``!clear_decisions`` clears only ``DECISIONS.md``.
	  - ``!clear_context`` clears only ``CONTEXT.md``.
	  - ``!clear_chat`` clears only the current session ``chat.md``.
	  - ``!clear_summary`` clears only the current session ``session_summary.md``.

	Skill control commands:
	- Aggregate commands:
	  - ``!list_skills`` lists all available skills and their short descriptions.
	  - ``!active_skills`` reports the skills currently active for the request-routing state.
	  - ``!skills_on`` re-enables skill injection for subsequent requests.
	  - ``!skills_off`` disables skill injection for subsequent requests.
	- Skill-level commands:
	  - ``!enable_skill <skill_name>`` force-enables a named skill when it exists in the catalog.
	  - ``!disable_skill <skill_name>`` suppresses a named skill from future activation.

	All cleanup commands preserve the storage directory layout and recreate files with
	their default Markdown headings so retrieval and future writes can continue normally.
	Skill commands update only in-memory routing state for the current agent instance; they
	do not modify the persisted Markdown memory files.
	"""

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
		self.llm = _sanitize_llm_sampling(llm)
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
		self._skills = self._load_skills_catalog()
		self._skills_enabled = True
		self._skills_user_enabled: set[str] = set()
		self._skills_user_disabled: set[str] = set()
		self._skills_cache: Tuple[int, List[str]] = (0, [])
		self._turn_counter = 0
		self._last_predicted_results_table: Optional[str] = None

		self._init_storage()
		self._initialize_executor()

	def _load_skills_catalog(self) -> Dict[str, Skill]:
		"""Load skills from markdown when available, falling back to built-ins."""
		catalog = dict(_builtin_skills())
		configured = _safe_text(self.config.skills_markdown_path).strip()
		if configured:
			skills_path = Path(configured).expanduser().resolve()
		else:
			skills_path = Path(__file__).with_name("skills.md")
		loaded = _load_skills_from_markdown(skills_path)
		if loaded:
			catalog.update(loaded)
		return catalog

	def _list_skills_text(self) -> str:
		lines = []
		for s in self._skills.values():
			lines.append(f"- {s.name}: {s.description}")
		return "\n".join(lines)

	def _select_skills_fallback(self, user_input: str) -> List[str]:
		q = _safe_text(user_input).lower()
		chosen: List[str] = []
		multi_series_request = any(k in q for k in ("massive", "many series", "multiple series", "grouped by", "group_key"))
		data_preparation_request = any(
			k in q
			for k in (
				"csv",
				"file into hana",
				"bring this file",
				"load this file",
				"import file",
				"import data",
				"upload csv",
				"load csv",
				"get the data into hana",
				"prepare the data",
				"get this dataset ready",
				"training table",
				"training set",
				"training data",
				"train table",
				"test set",
				"test table",
				"validation set",
				"validation table",
				"validation tables",
				"training, test, and validation",
				"training test validation",
				"train, test, and validation",
				"train test and validation",
				"train/test",
				"train test",
				"train, test",
				"train test validation",
				"train, test, validation",
				"split this table",
				"split the table",
				"split the data",
				"split data",
				"split dataset",
				"time order",
				"time-ordered",
				"respecting time order",
				"respect time order",
				"holdout",
				"prepare dataset",
			)
		)
		post_prediction_analysis = any(
			k in q
			for k in (
				"insight",
				"predicted result",
				"prediction result",
				"forecast result",
				"forecast did",
				"how did the forecast",
				"what do you see in the forecast",
				"error pattern",
				"actual vs",
				"compared with the actual",
				"comparison table",
			)
		)
		if any(k in q for k in ("artifact", "cap", "hdi", "deploy", "model storage", "list model", "delete model")):
			chosen.append("model_lifecycle_and_artifacts")
		if data_preparation_request:
			chosen.append("data_ingestion_and_dataset_preparation")
		if any(k in q for k in ("outlier", "anomaly", "spike", "abnormal", "clean data")):
			chosen.append("outlier_detection_and_repair_prep")
		if multi_series_request:
			chosen.append("massive_forecasting")
		if any(
			k in q
			for k in (
				"dataset report",
				"report",
				"profile",
				"diagnose",
				"stationarity",
				"trend",
				"seasonality",
				"white noise",
				"what does this data look like",
				"what does the data look like",
				"understand this dataset",
				"understand the data",
				"check this series",
				"check the series",
				"tell me about this time series",
				"recommend a model",
				"suggest a model",
				"which forecasting model",
			)
		):
			chosen.append("timeseries_data_profiling")
		if (not multi_series_request) and (not post_prediction_analysis) and (not data_preparation_request) and any(
			k in q
			for k in (
				"time series",
				"forecast",
				"predict",
				"ts_",
				"model",
				"train",
				"fit a model",
				"train a model",
				"use that model",
				"run a forecast",
				"future values",
			)
		):
			chosen.append("timeseries_forecasting")
		if any(
			k in q
			for k in (
				"accuracy",
				"mae",
				"rmse",
				"mape",
				"compare",
				"analysis",
				"evaluate",
				"insight",
				"predicted result",
				"prediction result",
				"forecast result",
				"bias",
				"actual vs",
			)
		):
			chosen.append("prediction_result_analysis")
		if any(k in q for k in ("sql", "select statement", "dataframe", "transform", "custom metric", "python snippet")):
			chosen.append("hana_dataframe_fallback")
		if "massive_forecast_comparison" in self._skills and any(k in q for k in ("compare groups", "group comparison", "per-group accuracy")):
			chosen.append("massive_forecast_comparison")
		# Respect config max
		out = []
		for name in chosen:
			if name in self._skills and name not in out:
				out.append(name)
		return out[: max(0, self.config.max_active_skills)]

	def _select_skills_llm(self, user_input: str) -> List[str]:
		# Keep selection cheap and cacheable.
		cache_turn, cached = self._skills_cache
		if self.config.skills_cache_turns > 0 and (self._turn_counter - cache_turn) <= self.config.skills_cache_turns:
			return list(cached)

		catalog = self._list_skills_text()
		prompt = (
			"You are selecting skills for an agent.\n"
			"Return a JSON array of skill names to activate (0..N), based on the user request.\n"
			"Only choose from the catalog; do not invent names.\n"
			f"Max skills: {self.config.max_active_skills}.\n\n"
			"Skill catalog:\n"
			+ catalog
			+ "\n\nUser request:\n"
			+ _truncate(user_input, self.config.skills_selector_max_chars)
		)
		raw = _safe_text(self._llm_call(prompt)).strip()
		obj = _safe_json_loads(raw)
		if not isinstance(obj, list):
			return self._select_skills_fallback(user_input)
		picked: List[str] = []
		for x in obj:
			name = _safe_text(x).strip()
			if name in self._skills and name not in picked:
				picked.append(name)
		fallback_picked = self._select_skills_fallback(user_input)
		if not picked:
			return fallback_picked
		for name in fallback_picked:
			if name not in picked:
				picked.append(name)
		if self.config.max_active_skills >= 0:
			picked = picked[: self.config.max_active_skills]
		self._skills_cache = (self._turn_counter, list(picked))
		return picked

	def _active_skill_names(self, user_input: str) -> List[str]:
		if not self.config.enable_skills or not self._skills_enabled:
			return []
		# User overrides take precedence.
		override = [s for s in self._skills_user_enabled if s in self._skills]
		if override:
			out = [s for s in override if s not in self._skills_user_disabled]
			return out[: self.config.max_active_skills]
		picked = self._select_skills_llm(user_input) if self.config.skills_use_llm_selector else self._select_skills_fallback(user_input)
		return [s for s in picked if s not in self._skills_user_disabled]

	def _render_skills_text(self, skill_names: Sequence[str]) -> str:
		if not skill_names:
			return ""
		blocks: List[str] = []
		for name in skill_names:
			s = self._skills.get(name)
			if not s:
				continue
			blocks.append(f"## Skill: {s.title} ({s.name})\n{s.content.strip()}")
		return "\n\n".join(blocks)

	def _initialize_executor(self) -> None:
		"""Create an OpenAI-native tool calling agent executor.

		Tool selection is delegated to the model via function/tool calling.
		"""
		if not self.tools:
			self._executor = None
			return

		# Make tool failures non-fatal (convert validation/tool errors to observations)
		for t in self.tools:
			for attr in ("handle_tool_error", "handle_validation_error"):
				try:
					if hasattr(t, attr):
						setattr(t, attr, True)
				except Exception:
					pass
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

	def _invalidate_retrieval_cache(self) -> None:
		"""Clear retrieval index state so future searches re-read persisted files."""
		self._indexed_mtimes = {}
		self._index.clear()

	def _reset_storage_file(self, path: Path, heading: str) -> None:
		"""Reset one persisted Markdown file and invalidate retrieval state."""
		_reset_markdown_file(path, heading)
		self._invalidate_retrieval_cache()

	def clear_notes_file(self) -> None:
		"""Clear the persisted NOTES markdown file."""
		self._reset_storage_file(self.storage_dir / "NOTES.md", "NOTES")

	def clear_todo_file(self) -> None:
		"""Clear the persisted TODO markdown file."""
		self._reset_storage_file(self.storage_dir / "TODO.md", "TODO")

	def clear_decisions_file(self) -> None:
		"""Clear the persisted DECISIONS markdown file."""
		self._reset_storage_file(self.storage_dir / "DECISIONS.md", "DECISIONS")

	def clear_context_file(self) -> None:
		"""Clear the persisted CONTEXT markdown file."""
		self._reset_storage_file(self.storage_dir / "CONTEXT.md", "CONTEXT")

	def clear_chat_history(self) -> None:
		"""Clear the current session chat history markdown file."""
		self._reset_storage_file(self._chat_path(), f"Session {self.session_id}")

	def clear_session_summary(self) -> None:
		"""Clear the current session summary markdown file."""
		self._reset_storage_file(self._summary_path(), f"Session {self.session_id}")

	def clear_memory_notes(self) -> None:
		"""Clear all persisted non-session memory note files."""
		self.clear_notes_file()
		self.clear_todo_file()
		self.clear_decisions_file()
		self.clear_context_file()

	def clear_session_memory(self) -> None:
		"""Clear the current session chat history and session summary files."""
		self.clear_chat_history()
		self.clear_session_summary()

	def reset_memory(self) -> None:
		"""Clear both persisted note files and the current session memory files."""
		self.clear_memory_notes()
		self.clear_session_memory()

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

	def _invoke_executor_with_deadline(self, agent_input: Dict[str, Any], *, user_input: str) -> Dict[str, Any]:
		"""Run the agent executor with an optional wall-clock ceiling.

		The default LangChain ``AgentExecutor`` bounds work by
		``max_iterations`` but not by elapsed time. When a tool call gets
		stuck on the backend (HANA/PAL retries, connection blips) each
		iteration can take tens of seconds, so eight iterations may still
		exceed ten minutes — long enough for a user to give up and Ctrl+C the
		notebook cell. ``max_turn_seconds`` gives us a hard turn-level
		deadline: we run the executor on a background daemon thread and, if
		the deadline passes before it returns, hand the user an actionable
		"we stopped waiting" message instead of hanging.

		The runaway thread is left daemonised — we cannot cleanly kill a
		blocking hdbcli call from Python — so any partial result it eventually
		produces is discarded. This is intentional: correctness beats letting
		zombie work bleed into the next turn.
		"""
		deadline = getattr(self.config, "max_turn_seconds", None)
		if not deadline or deadline <= 0:
			return self._executor.invoke(agent_input)

		box: Dict[str, Any] = {}

		def _target() -> None:
			try:
				box["result"] = self._executor.invoke(agent_input)
			except BaseException as exc:  # pylint: disable=broad-except
				box["exc"] = exc

		worker = threading.Thread(target=_target, daemon=True, name="context-agent-executor")
		start = time.monotonic()
		worker.start()
		worker.join(timeout=float(deadline))
		if worker.is_alive():
			elapsed = time.monotonic() - start
			timeout_msg = (
				f"Tool calling exceeded the ``max_turn_seconds={deadline:.0f}`` deadline "
				f"(elapsed ~{elapsed:.0f}s) and was abandoned.\n\n"
				"This usually means a backend HANA/PAL call is stuck, an identifier "
				"reference is being retried in a loop, or the model keeps re-issuing the "
				"same failing tool call. Look at the most recent tool observations for a "
				"``259 invalid table name`` / ``73001007`` / ``connection`` error and re-run "
				"with the exact identifiers reported in <memory_notes>. To wait longer, "
				"increase ``AgentConfig.max_turn_seconds`` when constructing ContextAgent."
			)
			return {"output": timeout_msg, "intermediate_steps": []}
		if "exc" in box:
			raise box["exc"]
		return box.get("result", {"output": "", "intermediate_steps": []})

	def _classify_error_text(self, err_text: str, user_input: str) -> Optional[str]:
		"""Translate known HANA/PAL/tool error strings into actionable hints.

		Returns ``None`` when no pattern matches so callers can fall back to a
		generic message. Shared between :meth:`_format_tool_executor_error`
		(which handles executor-level exceptions) and
		:meth:`_summarize_tool_observation` (which handles tool calls that
		swallowed the exception and returned a ``{"error": ...}`` JSON payload)
		so the agent gets the same hint no matter which path an error travels
		down. Without this shared translation, the agent would see raw HANA
		error tuples like ``(259, 'invalid table name: ...')`` and keep
		retrying blindly — exactly the loop-until-Ctrl+C behaviour we've
		observed against the SANDBOX schema.
		"""
		text = _safe_text(err_text).strip()
		if not text:
			return None
		lower_err = text.lower()
		lower_user = _safe_text(user_input).lower()
		is_massive = any(token in lower_user for token in ("group key", "group_key", "multiple time series", "many series", "massive"))

		# HANA error 259 "invalid table name / Could not find table/view ..." —
		# almost always caused by an unquoted mixed-case reference. Give the
		# agent the concrete fix so it stops retrying with the wrong identifier.
		if "259" in lower_err and ("invalid table name" in lower_err or "could not find table/view" in lower_err):
			return (
				"Tool execution failed: HANA could not find the referenced table.\n\n"
				"HANA folds unquoted identifiers to UPPER case at parse time. If the source "
				"table was created with a mixed-case name (e.g. ``..._my_hana_ai_model_8``), an "
				"unquoted ``SELECT ... FROM ...`` will look for the uppercase version and miss "
				"the real table.\n\n"
				"Do NOT invent, guess, or lowercase the table name. Reuse the exact "
				"``predicted_results_table`` returned by the prediction tool's JSON output "
				"(shown in <memory_notes>/<working_set>/prior tool observations). If the name "
				"contains lowercase characters, double-quote it in SQL (e.g. "
				"``SELECT * FROM \"my_hana_ai_model_8\"``).\n\n"
				f"Error: {text}"
			)

		# PAL feature-count / 73001007 — predict shape mismatch, including the
		# "Input Table could not empty, while FORECAST_LENGTH is not provided"
		# variant which fires when the predict table is empty or the model is
		# waiting on a forecast horizon.
		if any(token in lower_err for token in (
			"feature number of predict table does not match the trained model",
			"predict table features do not match the trained model",
			"73001007",
		)):
			shape_guidance = (
				"For forecasting prediction, the predict table should usually contain only the time key"
				+ (", the group key" if is_massive else "")
				+ ", and any explicit exogenous columns used at training time."
			)
			extra = ""
			if "forecast_length" in lower_err or "input table could not empty" in lower_err:
				extra = (
					"\nThe backend also reports ``Input Table could not empty, while FORECAST_LENGTH is not provided``. "
					"Verify that the predict table is non-empty and that its rows share the same key type as training. "
					"If you truly want to forecast beyond the last known point, build a future table with "
					"``TSMakeFutureTableTool`` and use that as the predict input.\n"
				)
			return (
				"Tool execution failed because the predict input shape does not match the trained model.\n\n"
				+ shape_guidance
				+ "\n"
				+ "Do not include the label/endog column in the predict table; keep that column only for scoring or evaluation.\n"
				+ extra
				+ f"\nError: {text}\n\n"
				+ "Tip: if you are predicting from a holdout table that still contains the target column, create or use a prediction input table that drops the target column first."
			)

		if "afl describe for nested call failed" in lower_err and "any-procedure call" in lower_err:
			backend_hint = (
				"This looks like a backend HANA PAL / hana_ml runtime issue rather than a simple missing parameter."
			)
			predict_hint = (
				" For massive forecasting, verify that predict inputs contain only group_key + key (+ exog) and score inputs contain group_key + key + endog (+ exog)."
				if is_massive else ""
			)
			return (
				"Tool execution failed due to a tool validation/runtime error.\n\n"
				+ backend_hint
				+ predict_hint
				+ f"\n\nError: {text}"
			)

		return None

	def _format_tool_executor_error(self, err: str, user_input: str) -> str:
		"""Convert common tool/runtime failures into actionable guidance."""
		err_text = _safe_text(err).strip()
		classified = self._classify_error_text(err_text, user_input)
		if classified is not None:
			return classified
		return (
			"Tool execution failed due to a tool validation/runtime error. "
			"I can continue if you confirm the right parameters.\n\n"
			f"Error: {err_text}\n\n"
			"Tip: for AccuracyMeasure, use supported evaluation_metric values such as 'mad'(≈MAE), 'rmse', 'mape', 'smape', etc."
		)

	def _summarize_tool_observation(self, tool_name: str, observation: str) -> Optional[str]:
		"""Summarize structured tool outputs that indicate auto-repair or actionable failures."""
		payload = _safe_json_loads(observation)
		if not isinstance(payload, dict):
			return None

		if payload.get("auto_repaired_predict_input"):
			used_columns = payload.get("predict_table_columns_used_for_prediction") or payload.get("columns_required_for_retry")
			before_columns = payload.get("predict_table_columns_before_repair") or payload.get("predict_table_columns")
			return (
				f"The tool {tool_name} auto-corrected the prediction input by dropping extra columns. "
				f"Before: {before_columns}. Used for prediction: {used_columns}."
			)

		if payload.get("error_category") == "predict_table_feature_mismatch":
			needed = payload.get("columns_required_for_retry")
			missing = payload.get("missing_required_columns")
			return (
				f"The prediction input did not match the trained model. Required inference columns: {needed}. "
				f"Missing columns: {missing}. Keep target/endog columns only for scoring, not prediction."
			)

		return None

	def _display_html_artifact(self, html_path: Path) -> bool:
		"""Render a generated HTML artifact when running in a notebook-capable frontend."""
		path = html_path.expanduser()
		if path.suffix.lower() != ".html" or not path.exists():
			return False
		try:
			from IPython.display import HTML, display  # type: ignore  # pylint: disable=import-error,import-outside-toplevel
		except Exception:
			return False
		try:
			html_text = path.read_text(encoding="utf-8")
		except Exception:
			return False
		try:
			display(HTML(html_text))
			return True
		except Exception:
			return False

	def _maybe_render_tool_artifact(self, tool_name: str, observation: str) -> Optional[str]:
		"""Render known artifact payloads and return a concise status string."""
		html_path = _extract_html_file_path(observation)
		if html_path is None:
			return None
		if self._display_html_artifact(html_path):
			return f"Rendered HTML artifact from {html_path} in the active notebook/output view."
		if html_path.exists():
			return f"Generated HTML artifact: {html_path}"
		return f"Tool {tool_name} reported HTML artifact path, but the file was not found: {html_path}"

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
			"- Never invent table names, model names, or model versions. Always reuse the exact identifiers recorded in <memory_notes> (NOTES/DECISIONS/TODO), <working_set>, or returned by a prior tool call. If a needed identifier (e.g. the predict table, model name, model version) is not present in context, ask the user before calling a tool — even if the name you have in mind 'looks right' (e.g. do not substitute SALES_REFUNDS_TEST for SALES_REFUNEDS_PREDICT, and do not assume model versions).\n"
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
			"- For split_table_for_forecasting, preserve chronology and use split_mode=time_ordered with the temporal ordering column.\n"
			"- For forecasting prediction tools, use predict tables that contain only inference columns: key, group_key when applicable, and any explicit exogenous columns. Do not include the label/endog column in predict inputs.\n"
			"- For scoring or evaluation tools, keep the label/endog column in the score table.\n"
			"- Natural requests like 'bring this csv into HANA', 'split this into train/test/validation', 'what does this series look like', 'recommend a model', or 'show me how the forecast did' should still trigger the corresponding tool workflow.\n"
			"- If the user refers to the current table/model/result from recent context, reuse that context instead of asking them to restate obvious names.\n"
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
		# Load NOTES (facts/constraints/preferences), DECISIONS (committed table/model
		# names, choices), and TODO (planned next steps) so the agent can reuse
		# names it has already committed to instead of inventing new ones like
		# ``SALES_REFUNDS_TEST`` when DECISIONS recorded ``SALES_REFUNEDS_PREDICT``.
		def _strip_md_title(text: str, title: str) -> str:
			s = text.lstrip()
			head = f"# {title}"
			if s.startswith(head):
				s = s[len(head):].lstrip("\n")
			return s.strip()

		notes_text = _strip_md_title(self._read_text(self.storage_dir / "NOTES.md", limit_chars=1600), "NOTES")
		decisions_text = _strip_md_title(self._read_text(self.storage_dir / "DECISIONS.md", limit_chars=1200), "DECISIONS")
		todo_text = _strip_md_title(self._read_text(self.storage_dir / "TODO.md", limit_chars=800), "TODO")
		memory_parts: List[str] = []
		if notes_text:
			memory_parts.append("# NOTES\n" + notes_text)
		if decisions_text:
			memory_parts.append("# DECISIONS\n" + decisions_text)
		if todo_text:
			memory_parts.append("# TODO\n" + todo_text)
		memory_notes = "\n\n".join(memory_parts)
		skill_names = self._active_skill_names(user_input)
		skills_text = self._render_skills_text(skill_names)

		working_set_lines: List[str] = []
		if self._last_predicted_results_table:
			working_set_lines.append(f"Latest predicted_results_table: {self._last_predicted_results_table}")
		working_set = "\n".join(working_set_lines).strip()

		return ContextPack(
			system=self._system_prompt(),
			task=f"User question: {user_input.strip()}",
			tool_guidance=self._tool_guidance(),
			skills=skills_text,
			working_set=working_set,
			session_summary=session_summary.strip(),
			memory_notes=_truncate(memory_notes.strip(), self.config.budgets.budget_memory_notes),
			retrieved=_truncate(retrieved_text.strip(), self.config.budgets.budget_retrieved),
			budgets=self.config.budgets,
		)

	# -------------- Public API --------------
	def chat(self, user_input: str) -> str:
		"""Run one conversational turn and persist it to Markdown memory."""
		self._turn_counter += 1

		# Skill management commands
		cmd = _safe_text(user_input).strip()
		if cmd == "!list_skills":
			return self._list_skills_text()
		if cmd == "!skills_on":
			self._skills_enabled = True
			return "Skills enabled."
		if cmd == "!skills_off":
			self._skills_enabled = False
			return "Skills disabled."
		if cmd.startswith("!enable_skill "):
			name = cmd.split(" ", 1)[1].strip()
			if name not in self._skills:
				return f"Unknown skill: {name}. Use !list_skills."
			self._skills_user_enabled.add(name)
			if name in self._skills_user_disabled:
				self._skills_user_disabled.remove(name)
			return f"Enabled skill: {name}."
		if cmd.startswith("!disable_skill "):
			name = cmd.split(" ", 1)[1].strip()
			self._skills_user_disabled.add(name)
			return f"Disabled skill: {name}."
		if cmd == "!active_skills":
			names = self._active_skill_names(cmd)
			return "Active skills: " + (", ".join(names) if names else "(none)")
		if cmd == "!clear_notes_file":
			self.clear_notes_file()
			return "Cleared NOTES."
		if cmd == "!clear_todo":
			self.clear_todo_file()
			return "Cleared TODO."
		if cmd == "!clear_decisions":
			self.clear_decisions_file()
			return "Cleared DECISIONS."
		if cmd == "!clear_context":
			self.clear_context_file()
			return "Cleared CONTEXT."
		if cmd == "!clear_chat":
			self.clear_chat_history()
			return f"Cleared chat history for session '{self.session_id}'."
		if cmd == "!clear_summary":
			self.clear_session_summary()
			return f"Cleared session summary for session '{self.session_id}'."
		if cmd == "!clear_notes":
			self.clear_memory_notes()
			return "Cleared NOTES, TODO, DECISIONS, and CONTEXT."
		if cmd == "!clear_session":
			self.clear_session_memory()
			return f"Cleared chat and session summary for session '{self.session_id}'."
		if cmd == "!reset_memory":
			self.reset_memory()
			return f"Reset memory notes and session state for session '{self.session_id}'."
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
			try:
				result = self._invoke_executor_with_deadline({"input": prompt}, user_input=user_input)
			except Exception as exc:
				# Never hard-fail a notebook cell due to tool validation/runtime errors.
				err = _safe_text(exc)
				self._append_chat("tool", f"### TOOL_EXECUTOR_ERROR\n\n{err}")
				assistant_text = self._format_tool_executor_error(err, user_input)
				result = {"output": assistant_text, "intermediate_steps": []}

			assistant_text = _safe_text(result.get("output") if isinstance(result, dict) else result).strip()
			steps = result.get("intermediate_steps") if isinstance(result, dict) else None
			tool_diagnostics: List[str] = []
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

					# Track the most recent predicted results table so the agent can
					# reliably reference it in follow-up requests like plotting.
					payload = _safe_json_loads(obs_text)
					if isinstance(payload, dict):
						latest = payload.get("predicted_results_table")
						if isinstance(latest, str) and latest.strip():
							self._last_predicted_results_table = latest.strip()
					diagnostic = self._summarize_tool_observation(tool_name, obs_text)
					if diagnostic:
						tool_diagnostics.append(diagnostic)
					artifact_note = self._maybe_render_tool_artifact(tool_name, obs_text)
					if artifact_note:
						tool_diagnostics.append(artifact_note)

					tool_trace.append(f"### TOOL {tool_name} args={args_text}\n\n{_truncate(obs_text, 3000)}")
					tool_return_snippets.append(f"[Tool Return] {tool_name} args={args_text}\n{_truncate(obs_text, 1200)}")

			if tool_diagnostics:
				diagnostics_text = "\n\n".join(tool_diagnostics)
				assistant_text = (assistant_text + "\n\n" + diagnostics_text).strip() if assistant_text else diagnostics_text

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

