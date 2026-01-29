"""
Mem0 compatibility adapter for HANA vector store.

This adapter provides a Mem0-like API (`add`, `search`, `delete`, `update`) on top of
SAP HANA via the LangChain `HanaDB` vector store and the existing `HANAVectorEmbeddings`.
It also supports optional cross-encoder reranking using `PALCrossEncoder`.

Goals:
- Quick compatibility mode without modifying Mem0 OSS: drop-in class mirroring key methods
  so existing code can call `add/search/delete` similarly.
- Injectable components for testing: `vectorstore`, `embedder`, `reranker` can be passed
  in to avoid real HANA connections during unit tests.

Key methods:
- add(memories, user_id=None, metadata=None): Ingest a list of text memories and return IDs.
- search(query, top_k=5, threshold=None, filters=None, rerank=True): Retrieve ranked results.
- delete(filter): Delete entries matching filter in the backend.
- update(id, new_text, metadata=None): Replace content (fallback to delete+add when direct update isn't available).

Note: This is a compatibility layer, not a full Mem0 provider. Once Mem0 OSS `hana` provider
is added upstream, this adapter can be replaced by the official provider.
"""

#pylint: disable=edefined-builtin

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import logging
import hashlib

from langchain_community.vectorstores.hanavector import HanaDB
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings

try:
    # Optional reranker interface: requires predict([(query, doc_text), ...]) -> List[float]
    from hana_ai.vectorstore.pal_cross_encoder import PALCrossEncoder  # type: ignore
except Exception:  # pragma: no cover - optional
    PALCrossEncoder = None

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Result item from memory search.
    """
    text: str
    score: float
    metadata: Dict[str, Any]
    rerank_score: Optional[float] = None


class Mem0HanaAdapter:
    """
    Mem0-like adapter for HANA vector store.

    Parameters
    ----------
    connection_context : Any
        HANA connection context providing `.connection` (DBAPI) and helpers.
    table_name : str
        Target HANA vector table name for long-term memory storage.
    embedder : Embeddings, optional
        LangChain `Embeddings` implementation. Defaults to `HANAVectorEmbeddings(connection_context)`
        if not provided and a connection_context is available.
    reranker : Any, optional
        Cross-encoder reranker. Must implement `predict(pairs: List[Tuple[str, str]]) -> List[float]`.
        Defaults to `PALCrossEncoder(connection_context)` when available, else `None`.
    vectorstore : Any, optional
        Injected vector store (for tests). If not provided, initializes `HanaDB`.
    score_threshold : float, optional
        Default threshold for similarity search (when backend supports it). Defaults to 0.0.
    """

    def __init__(
        self,
        connection_context: Optional[Any] = None,
        table_name: Optional[str] = None,
        embedder: Optional[Embeddings] = None,
        reranker: Optional[Any] = None,
        vectorstore: Optional[Any] = None,
        score_threshold: float = 0.0,
        ingest_filter: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        max_length: Optional[int] = None,
        default_ttl_seconds: Optional[int] = None,
        short_term_ttl_seconds: Optional[int] = None,
        partition_defaults: Optional[Dict[str, Any]] = None,
        export_handler: Optional[Callable[[Any, str, Optional[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
    ) -> None:
        self.connection_context = connection_context
        self.table_name = table_name
        self.embedder = embedder
        self.reranker = reranker
        self.score_threshold = score_threshold
        self.ingest_filter = ingest_filter
        self.max_length = max_length
        self.default_ttl_seconds = default_ttl_seconds
        self.short_term_ttl_seconds = short_term_ttl_seconds
        self.partition_defaults = partition_defaults or {}
        self.export_handler = export_handler

        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            if not self.connection_context:
                raise ValueError("connection_context is required to initialize HanaDB vectorstore")
            if not self.table_name:
                raise ValueError("table_name is required to initialize HanaDB vectorstore")
            if self.embedder is None:
                # Lazy import to avoid circulars
                from hana_ai.vectorstore.embedding_service import HANAVectorEmbeddings  # type: ignore
                self.embedder = HANAVectorEmbeddings(self.connection_context)
            self.vectorstore = HanaDB(
                embedding=self.embedder,
                connection=self.connection_context.connection,
                table_name=self.table_name,
            )

        if self.reranker is None and PALCrossEncoder is not None and self.connection_context is not None:
            try:
                self.reranker = PALCrossEncoder(self.connection_context)
            except Exception as e:
                logger.warning("PALCrossEncoder init failed, continuing without reranker: %s", e)
                self.reranker = None

    # ---------------------------------------------------------------------
    # Public API (Mem0-like)
    # ---------------------------------------------------------------------
    def add(self, memories: List[Dict[str, Any]], user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add a batch of memories.

        Each memory item is expected to contain either `text` or `content`.
        Returns a list of backend IDs (if available) or empty strings.
        """
        now = datetime.now().isoformat()
        docs: List[Document] = []
        for m in memories:
            text = m.get("text") or m.get("content")
            if not text:
                raise ValueError("Each memory must include 'text' or 'content'")
            if self.max_length is not None and len(text) > self.max_length:
                logger.info("Skip ingestion due to max_length: %s chars", len(text))
                continue
            md_input = dict(metadata or {})
            md_input.update(m.get("metadata", {}))
            # Merge partition defaults
            md = {**self.partition_defaults, **md_input}
            # tags / entity partitioning
            if "tags" in m and isinstance(m["tags"], list):
                md["tags"] = m["tags"]
            if "entity_id" in m:
                md["entity_id"] = m["entity_id"]
            if "entity_type" in m:
                md["entity_type"] = m["entity_type"]
            if user_id:
                md.setdefault("user_id", user_id)
            md.setdefault("timestamp", now)
            # content hash for dedup / organization
            md.setdefault("content_hash", hashlib.sha256(text.encode("utf-8")).hexdigest())
            # TTL / expiration
            ttl = m.get("ttl_seconds")
            tier = m.get("tier")  # 'short' or 'long'
            if tier:
                md["tier"] = tier
            if ttl is None:
                if tier == "short" and self.short_term_ttl_seconds:
                    ttl = self.short_term_ttl_seconds
                elif self.default_ttl_seconds:
                    ttl = self.default_ttl_seconds
            if ttl:
                try:
                    base_time_iso = md.get("timestamp", now)
                    expires_at = datetime.fromisoformat(base_time_iso)
                    expires_at = expires_at.timestamp() + int(ttl)
                    md["expires_at"] = datetime.fromtimestamp(expires_at).isoformat()
                except Exception:
                    pass
            # ingestion predicate last (so we can inspect metadata)
            if self.ingest_filter is not None:
                try:
                    if not self.ingest_filter(text, md):
                        logger.info("Skip ingestion by ingest_filter")
                        continue
                except Exception as e:
                    logger.warning("ingest_filter error, skipping check: %s", e)
            docs.append(Document(page_content=text, metadata=md))

        try:
            ids = self.vectorstore.add_documents(docs)
        except TypeError:
            # Some vectorstores return None; keep interface consistent
            self.vectorstore.add_documents(docs)
            ids = [""] * len(docs)
        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
    ) -> List[SearchResult]:
        """
        Search memories with optional reranking.

        Returns a list of `SearchResult` with `text`, `score`, `metadata`, and optional `rerank_score`.
        """
        score_thr = self.score_threshold if threshold is None else threshold

        # Prefer HANA relevance API when available
        if hasattr(self.vectorstore, "similarity_search_with_relevance_scores"):
            candidates = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=top_k,
                score_threshold=score_thr,
                filter=filters,
            )
            # candidates: List[Tuple[Document, float]]
            docs, vec_scores = zip(*candidates) if candidates else ([], [])
        else:
            # Fallback to (doc, score) API or plain similarity_search
            if hasattr(self.vectorstore, "similarity_search_with_score"):
                pairs = self.vectorstore.similarity_search_with_score(query=query, k=top_k)
                docs, vec_scores = zip(*pairs) if pairs else ([], [])
            else:
                docs = self.vectorstore.similarity_search(query=query, k=top_k)
                vec_scores = [0.0] * len(docs)

        results: List[SearchResult] = [
            SearchResult(text=d.page_content, score=float(s), metadata=dict(d.metadata))
            for d, s in zip(docs, vec_scores)
        ]

        # Optional reranking
        if rerank and self.reranker is not None and results:
            pairs = [(query, r.text) for r in results]
            try:
                rerank_scores = self.reranker.predict(pairs)
                for r, rs in zip(results, rerank_scores):
                    r.rerank_score = float(rs)
                # Sort by rerank score desc, tie-break by vec score
                results.sort(key=lambda x: (x.rerank_score if x.rerank_score is not None else 0.0, x.score), reverse=True)
            except Exception as e:
                logger.warning("Rerank failed, returning vector scores only: %s", e)
        else:
            # Sort by vector score desc
            results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def delete(self, filter: Dict[str, Any]) -> int:
        """
        Delete memories matching filter. Returns deletion count when available.
        """
        try:
            deleted = self.vectorstore.delete(filter=filter)
            if isinstance(deleted, int):
                return deleted
            # Some backends return None or bool
            return 0
        except Exception as e:
            logger.error("Delete failed: %s", e)
            return 0

    # ---------------------- Advanced features ----------------------
    def delete_expired(self, now_iso: Optional[str] = None) -> int:
        """
        Delete expired memories based on `expires_at` metadata.
        """
        now_iso = now_iso or datetime.now().isoformat()
        return self.delete({"expires_at": {"$lte": now_iso}})

    def search_by_tags(
        self,
        tags: List[str],
        query: str = "",
        top_k: int = 5,
        threshold: Optional[float] = None,
        rerank: bool = True,
    ) -> List[SearchResult]:
        """
        Search memories filtered by tags.
        """
        filters: Dict[str, Any] = {"tags": {"$contains": tags}}
        return self.search(query=query, top_k=top_k, threshold=threshold, filters=filters, rerank=rerank)

    def export(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Export memories via a pluggable handler.

        Because LangChain vectorstores typically don't expose a full scan API,
        this adapter accepts `export_handler(connection_context, table_name, filters)`.
        Implementations can use HANA SQL to dump rows from the underlying table.
        """
        if self.export_handler is None:
            raise NotImplementedError("export_handler is not configured for Mem0HanaAdapter")
        return self.export_handler(self.connection_context, self.table_name, filters)

    def update(self, id: str, new_text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update memory content. If backend doesn't support update, emulate via delete + add.
        """
        # HanaDB may not expose an update API; emulate.
        try:
            # Attempt direct update if available
            if hasattr(self.vectorstore, "update"):
                return bool(self.vectorstore.update(id=id, text=new_text, metadata=metadata or {}))
        except Exception:
            pass

        # Fallback: delete by id, then add new doc (id cannot be preserved)
        try:
            self.delete({"id": id})
            self.add([{"text": new_text, "metadata": metadata or {}}])
            return True
        except Exception as e:
            logger.error("Update emulation failed: %s", e)
            return False

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------
    def add_texts(self, texts: List[str], user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add a list of text strings as memories.
        """
        memories = [{"text": t, "metadata": metadata or {}} for t in texts]
        return self.add(memories, user_id=user_id, metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize adapter configuration to a dictionary.
        """
        return {
            "backend": "hana",
            "table_name": self.table_name,
            "score_threshold": self.score_threshold,
            "has_reranker": self.reranker is not None,
        }
