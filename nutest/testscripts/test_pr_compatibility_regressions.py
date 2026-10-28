"""Compatibility regressions in the security PR; no external services required."""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from hana_ai.client.mcp_client import _validate_stdio_command
from hana_ai.mem0.hana_mem0_adapter import Mem0HanaAdapter
from hana_ai.mem0.memory_manager import Mem0MemoryManager, default_hana_export_handler


SCOPE = {"tenant_id": "tenant-a", "session_id": "session-1"}
METADATA = {
    **SCOPE, "category": "forecast", "year": 2026, "tags": ["sales", "plan"],
    "day": "2026-10-28", "label": "a.b[1]", "enabled": True,
}


@pytest.mark.parametrize("consumer", ["search", "manager_export", "hana_export"])
@pytest.mark.parametrize(
    "filters,matches",
    [
        ({"category": "forecast"}, True),
        ({"category": {"$eq": "forecast"}}, True),
        ({"category": {"$ne": "other"}}, True),
        ({"year": {"$gt": 2025}}, True),
        ({"year": {"$gte": 2026}}, True),
        ({"year": {"$lt": 2027}}, True),
        ({"year": {"$lte": 2026}}, True),
        ({"category": {"$in": ["forecast", "other"]}}, True),
        ({"category": {"$nin": ["other"]}}, True),
        ({"year": {"$between": [2025, 2026]}}, True),
        ({"year": {"$gte": 2025, "$lt": 2027}}, True),
        ({"category": {"$like": "fore_ast%"}}, True),
        ({"label": {"$like": "a.b[1]"}}, True),
        ({"day": {"type": "date", "date": "2026-10-28"}}, True),
        ({"day": {"$gte": {"type": "date", "date": "2026-10-27"}}}, True),
        ({"tags": {"$contains": ["sales", "plan"]}}, True),
        ({"tags": {"$contains": "sales"}}, True),
        ({"$and": [{"year": {"$gte": 2025}}, {"category": "forecast"}]}, True),
        ({"$or": [{"year": 2025}, {"category": "forecast"}]}, True),
        ({"$and": [{"$or": [{"year": 2026}, {"year": 2025}]}]}, True),
        ({"missing": {"$eq": None}}, True),
        ({"category": {"$ne": None}}, True),
        ({"category": {"$eq": "other"}}, False),
        ({"year": {"$gt": 2026}}, False),
        ({"year": {"$gte": 2027}}, False),
        ({"year": {"$lt": 2026}}, False),
        ({"year": {"$lte": 2025}}, False),
        ({"category": {"$ne": "forecast"}}, False),
        ({"category": {"$in": ["other"]}}, False),
        ({"category": {"$nin": ["forecast"]}}, False),
        ({"year": {"$nin": ["2026"]}}, False),
        ({"category": {"$nin": [{"$invalid": "forecast"}]}}, False),
        ({"category": {"$nin": [None]}}, False),
        ({"year": {"$between": [2024, 2025]}}, False),
        ({"category": {"$like": "other%"}}, False),
        ({"missing": {"$ne": "other"}}, False),
        ({"missing": {"$nin": ["other"]}}, False),
        ({"year": {"$lte": "invalid-number"}}, False),
        ({"enabled": {"$eq": 1}}, False),
        ({"category": {"$unknown": "forecast"}}, False),
        ({"tags": {"$contains": "sales", "$unknown": True}}, False),
        ({"tags": {"$contains": "sales", "$eq": ["other"]}}, False),
        ({"category": {"$in": "forecast"}}, False),
        ({"category": {}}, False),
        ({"$or": []}, False),
        ({"$and": []}, False),
        ({"$or": [None]}, False),
        ({"$and": {"year": 2026}}, False),
        ({"day": {"$gte": {"type": "date", "date": "bad-date"}}}, False),
    ],
)
def test_memory_filters_preserve_matches_and_isolation(consumer, filters, matches):
    # An unfiltered backend includes matching rows from another tenant/session.
    documents = [
        Document(page_content="allowed", metadata=METADATA),
        Document(page_content="other-tenant", metadata={**METADATA, "tenant_id": "tenant-b"}),
        Document(page_content="other-session", metadata={**METADATA, "session_id": "session-2"}),
        Document(page_content="missing-scope", metadata={"category": "forecast", "year": 2026}),
    ]
    scoped_filters = {**filters, **SCOPE}
    if consumer == "hana_export":
        cursor = Mock()
        cursor.description = [("CONTENT",), ("METADATA",)]
        cursor.fetchall.return_value = [
            (doc.page_content, json.dumps(doc.metadata)) for doc in documents
        ]
        connection = Mock()
        connection.cursor.return_value = cursor
        rows = default_hana_export_handler(
            SimpleNamespace(connection=connection), "MEMORY", scoped_filters
        )
        actual = [row["content"] for row in rows]
        cursor.close.assert_called_once()
    else:
        store = SimpleNamespace(
            similarity_search_with_score=Mock(return_value=[(doc, 0.9) for doc in documents])
        )
        adapter = Mem0HanaAdapter(
            vectorstore=store,
            export_handler=lambda *_: [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in documents
            ],
        )
        if consumer == "search":
            actual = [item.text for item in adapter.search("forecast", filters=scoped_filters)]
            assert store.similarity_search_with_score.call_args.kwargs["filter"] == scoped_filters
        else:
            manager = Mem0MemoryManager(None, "MEMORY", adapter=adapter, **SCOPE)
            actual = [row["content"] for row in manager.export(filters)]
    assert actual == (["allowed"] if matches else [])


@pytest.mark.parametrize("search_api", ["relevance", "score_without_filters", "plain"])
def test_scope_survives_nested_or_with_backend_api_fallback(search_api):
    documents = [
        Document(page_content="allowed", metadata=METADATA),
        Document(page_content="other-tenant", metadata={**METADATA, "tenant_id": "tenant-b"}),
    ]
    if search_api == "relevance":
        store = SimpleNamespace(similarity_search_with_relevance_scores=lambda **_: [
            (doc, 0.9) for doc in documents
        ])
    elif search_api == "score_without_filters":
        store = SimpleNamespace(similarity_search_with_score=lambda query, k: [
            (doc, 0.9) for doc in documents
        ])
    else:
        store = SimpleNamespace(similarity_search=lambda query, k: documents)
    adapter = Mem0HanaAdapter(vectorstore=store)
    filters = {**SCOPE, "$or": [{"tenant_id": "tenant-b"}, {"year": {"$gte": 2025}}]}
    assert [row.text for row in adapter.search("forecast", filters=filters)] == ["allowed"]


def _approve_stdio(monkeypatch, args, additional=False):
    monkeypatch.setenv("HANA_MCP_STDIO_ALLOWED_PROFILES", json.dumps([{
        "command": sys.executable, "args_prefix": args, "allow_additional_args": additional,
    }]))


@pytest.mark.parametrize("path", [
    "/code/server.py", "/config/server.py", "/c/server.py", "/opt/server.py",
    "/code dir/my server.py",
])
def test_stdio_approved_absolute_script_paths(monkeypatch, path):
    _approve_stdio(monkeypatch, [path])
    assert _validate_stdio_command(sys.executable, [path]) == os.path.realpath(sys.executable)


def test_stdio_approved_absolute_data_path(monkeypatch):
    args = ["trusted_server.py", "--config", "/config/settings.json"]
    _approve_stdio(monkeypatch, args)
    assert _validate_stdio_command(sys.executable, args) == os.path.realpath(sys.executable)


@pytest.mark.parametrize("additional", [False, True])
def test_stdio_absolute_path_additional_args_need_opt_in(monkeypatch, additional):
    _approve_stdio(monkeypatch, ["trusted_server.py"], additional)
    args = ["trusted_server.py", "/config/settings.json"]
    if additional:
        assert _validate_stdio_command(sys.executable, args) == os.path.realpath(sys.executable)
    else:
        with pytest.raises(ValueError, match="do not match"):
            _validate_stdio_command(sys.executable, args)


def test_stdio_unapproved_absolute_script_stays_blocked(monkeypatch):
    _approve_stdio(monkeypatch, ["/code/trusted.py"])
    with pytest.raises(ValueError, match="do not match"):
        _validate_stdio_command(sys.executable, ["/code/untrusted.py"])


@pytest.mark.parametrize("flag", [
    "-c", "-cprint(1)", "-econsole.log(1)", "--eval=code", "-Command",
    "-EncodedCommand", "/c", "/Cwhoami", "/c whoami", "/c /bin/echo", "/c\\whoami",
])
def test_stdio_inline_flags_stay_blocked_with_host_approval(monkeypatch, flag):
    _approve_stdio(monkeypatch, ["trusted_server.py"], additional=True)
    with pytest.raises(ValueError, match="inline code"):
        _validate_stdio_command(sys.executable, ["trusted_server.py", flag])
