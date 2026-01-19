"""
High-level Memory Manager for Mem0-like features on HANA.

This module separates memory management concerns from the agent logic.
It wraps `Mem0HanaAdapter` and provides opinionated utilities aligned with
Mem0 cookbooks:

- Entity partitioning: `entity_id` + `entity_type` scoping and filters
- Controlling ingestion: predicate-based rules (length, allow/deny tags, toggle)
- Memory expiration: short- vs long-term TTLs and expiration cleanup
- Tagging & organizing: tag-based search helpers
- Exporting memories: pluggable HANA export handler using SQL
- Architecture selection: `vector` (default) vs `graph` (placeholder)

Note: Graph architecture is declared for API completeness but not implemented.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import logging

from hana_ai.mem0.hana_mem0_adapter import Mem0HanaAdapter, SearchResult
from hana_ai.mem0.memory_classifier import Mem0IngestionClassifier
from hana_ai.mem0.memory_entity_extractor import Mem0EntityExtractor

logger = logging.getLogger(__name__)


@dataclass
class IngestionRules:
    enabled: bool = True
    min_length: int = 1
    max_length: Optional[int] = None
    allow_tags: Optional[List[str]] = None
    deny_tags: Optional[List[str]] = None


def default_hana_export_handler(connection_context: Any, table_name: str, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Basic HANA export handler. Tries to select content + metadata columns.
    This is best-effort due to vectorstore schema variations.
    """
    rows: List[Dict[str, Any]] = []
    try:
        conn = connection_context.connection  # DBAPI connection
        cur = conn.cursor()
        # Best-effort: select all then normalize
        cur.execute(f'SELECT * FROM "{table_name}"')
        colnames = [d[0] for d in cur.description]
        for rec in cur.fetchall():
            row = {colnames[i]: rec[i] for i in range(len(colnames))}
            # Normalize common fields
            content = row.get('CONTENT') or row.get('content') or row.get('PAGE_CONTENT') or row.get('page_content')
            metadata = row.get('METADATA') or row.get('metadata')
            if isinstance(metadata, str):
                # HANA may store JSON as NCLOB text
                try:
                    import json
                    metadata = json.loads(metadata)
                except Exception:
                    pass
            rows.append({
                'content': content,
                'metadata': metadata if isinstance(metadata, dict) else {},
                '_raw': row,
            })
        cur.close()
    except Exception as e:
        logger.error("Export handler failed for table %s: %s", table_name, e)
    return rows


class Mem0MemoryManager:
    """
    High-level memory manager that encapsulates Mem0HanaAdapter usage and policies.
    """

    def __init__(
        self,
        connection_context: Any,
        table_name: str,
        embedder: Optional[Any] = None,
        reranker: Optional[Any] = None,
        architecture: str = "vector",  # 'vector' | 'graph'
        default_ttl_seconds: Optional[int] = None,
        short_term_ttl_seconds: Optional[int] = None,
        partition_defaults: Optional[Dict[str, Any]] = None,
        ingestion_rules: Optional[IngestionRules] = None,
        auto_classification_enabled: bool = False,
        classifier: Optional[Mem0IngestionClassifier] = None,
        category_routing: Optional[Dict[str, Dict[str, Optional[int]]]] = None,
        auto_entity_extraction_enabled: bool = False,
        entity_extractor: Optional[Mem0EntityExtractor] = None,
        entity_assignment_mode: str = "merge",  # 'manager' | 'extract' | 'merge'
    ) -> None:
        self.architecture = architecture
        if self.architecture != "vector":
            raise NotImplementedError("Only 'vector' architecture is implemented currently")

        self.entity_id: Optional[str] = None
        self.entity_type: Optional[str] = None

        self.default_ttl_seconds = default_ttl_seconds
        self.short_term_ttl_seconds = short_term_ttl_seconds

        self.ingestion_rules = ingestion_rules or IngestionRules()
        self._ingest_predicate = self._build_ingest_predicate(self.ingestion_rules)

        self.auto_classification_enabled = auto_classification_enabled
        self.classifier = classifier
        self.category_routing = category_routing or {
            "preference": {"tier": "long", "ttl_seconds": None},
            "fact": {"tier": "long", "ttl_seconds": None},
            "task": {"tier": "short", "ttl_seconds": self.short_term_ttl_seconds or 3 * 24 * 3600},
            "session_state": {"tier": "short", "ttl_seconds": self.short_term_ttl_seconds or 3 * 24 * 3600},
            "ephemeral": {"tier": "short", "ttl_seconds": self.short_term_ttl_seconds or 24 * 3600},
        }

        self.auto_entity_extraction_enabled = auto_entity_extraction_enabled
        self.entity_extractor = entity_extractor
        if entity_assignment_mode not in ("manager", "extract", "merge"):
            entity_assignment_mode = "merge"
        self.entity_assignment_mode = entity_assignment_mode

        self.adapter = Mem0HanaAdapter(
            connection_context=connection_context,
            table_name=table_name,
            embedder=embedder,
            reranker=reranker,
            score_threshold=0.0,
            ingest_filter=self._ingest_predicate,
            max_length=self.ingestion_rules.max_length,
            default_ttl_seconds=default_ttl_seconds,
            short_term_ttl_seconds=short_term_ttl_seconds,
            partition_defaults=partition_defaults,
            export_handler=default_hana_export_handler,
        )

    # ----------------------- Entity partitioning -----------------------
    def set_entity(self, entity_id: Optional[str], entity_type: Optional[str] = None) -> None:
        self.entity_id = entity_id
        self.entity_type = entity_type

    # ----------------------- Ingestion control -------------------------
    def _build_ingest_predicate(self, rules: IngestionRules) -> Callable[[str, Dict[str, Any]], bool]:
        def predicate(text: str, metadata: Dict[str, Any]) -> bool:
            if not rules.enabled:
                return False
            if text is None:
                return False
            if len(text) < max(0, rules.min_length or 0):
                return False
            if rules.max_length is not None and len(text) > rules.max_length:
                return False
            tags = metadata.get('tags', [])
            if rules.deny_tags:
                if any(t in rules.deny_tags for t in (tags or [])):
                    return False
            if rules.allow_tags is not None:
                # If allow_tags specified, at least one tag must be in allowed list
                if not any(t in rules.allow_tags for t in (tags or [])):
                    return False
            return True
        return predicate

    def update_ingestion_rules(self, rules: IngestionRules) -> None:
        self.ingestion_rules = rules
        self._ingest_predicate = self._build_ingest_predicate(rules)
        # Re-wire adapter predicate and max_length
        self.adapter.ingest_filter = self._ingest_predicate
        self.adapter.max_length = rules.max_length

    # ----------------------- Add memories ------------------------------
    def add_memory(
        self,
        text: str,
        tags: Optional[List[str]] = None,
        tier: str = "long",  # 'short'|'long'
        ttl_seconds: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        md: Dict[str, Any] = extra_metadata.copy() if extra_metadata else {}
        extracted: Optional[Dict[str, str]] = None
        if self.auto_entity_extraction_enabled and self.entity_extractor is not None:
            extracted = self.entity_extractor.extract(text)

        # Assign entity fields
        if self.entity_assignment_mode == "manager":
            if self.entity_id:
                md["entity_id"] = self.entity_id
            if self.entity_type:
                md["entity_type"] = self.entity_type
            if extracted:
                md["subject_entity_id"] = extracted.get("entity_id", "unknown")
                md["subject_entity_type"] = extracted.get("entity_type", "other")
                alias = extracted.get("entity_name", "unknown")
                md.setdefault("entity_alias", alias)
                md["subject_entity_alias"] = alias
                md.setdefault("tags", []).append(f"entity:{extracted.get('entity_type','other')}:{extracted.get('entity_id','unknown')}")
        elif self.entity_assignment_mode == "extract":
            if extracted:
                md["entity_id"] = extracted.get("entity_id", "unknown")
                md["entity_type"] = extracted.get("entity_type", "other")
                md["entity_alias"] = extracted.get("entity_name", "unknown")
            elif self.entity_id or self.entity_type:
                # fallback to manager settings if extractor returns unknown
                if self.entity_id:
                    md["entity_id"] = self.entity_id
                if self.entity_type:
                    md["entity_type"] = self.entity_type
        else:  # merge
            if self.entity_id:
                md["entity_id"] = self.entity_id
            if self.entity_type:
                md["entity_type"] = self.entity_type
            if extracted:
                md["subject_entity_id"] = extracted.get("entity_id", "unknown")
                md["subject_entity_type"] = extracted.get("entity_type", "other")
                alias = extracted.get("entity_name", "unknown")
                md.setdefault("entity_alias", alias)
                md["subject_entity_alias"] = alias
                md.setdefault("tags", []).append(f"entity:{extracted.get('entity_type','other')}:{extracted.get('entity_id','unknown')}")
        final_tags = list(tags or [])
        final_tier = tier
        final_ttl = ttl_seconds

        if self.auto_classification_enabled and self.classifier is not None:
            cls = self.classifier.classify(text)
            cat = cls.get("category")
            ctags = cls.get("tags", [])
            prio = cls.get("priority", 0.5)
            final_tags.extend([t for t in ctags if t not in final_tags])
            if cat in self.category_routing:
                route = self.category_routing[cat]
                final_tier = str(route.get("tier", final_tier))
                if final_ttl is None:
                    rt = route.get("ttl_seconds")
                    if rt is not None:
                        final_ttl = int(rt)
            if final_ttl is None:
                if final_tier == "short" and self.short_term_ttl_seconds:
                    final_ttl = int(self.short_term_ttl_seconds)
                elif self.default_ttl_seconds:
                    final_ttl = int(self.default_ttl_seconds)
            md["priority"] = prio
            md["category"] = cat

        payload = {
            "text": text,
            "metadata": md,
            "tags": final_tags,
            "tier": final_tier,
        }
        if final_ttl is not None:
            payload["ttl_seconds"] = int(final_ttl)
        self.adapter.add([payload])

    def add_interaction(self, user_input: str, assistant_output: str, tags: Optional[List[str]] = None, tier: str = "long") -> None:
        text = f"User: {user_input}\nAssistant: {assistant_output}"
        self.add_memory(text=text, tags=(tags or ["chat", "conversation"]), tier=tier)

    # ----------------------- Retrieve memories -------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
        tags: Optional[List[str]] = None,
        rerank: bool = True,
    ) -> List[SearchResult]:
        filters: Optional[Dict[str, Any]] = None
        if self.entity_id or self.entity_type or tags:
            filters = {}
            if self.entity_id:
                filters["entity_id"] = self.entity_id
            if self.entity_type:
                filters["entity_type"] = self.entity_type
            if tags:
                filters["tags"] = {"$contains": tags}
        return self.adapter.search(query=query, top_k=top_k, threshold=threshold, filters=filters, rerank=rerank)

    def retrieve_texts(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
        tags: Optional[List[str]] = None,
        rerank: bool = True,
    ) -> List[str]:
        results = self.retrieve(query=query, top_k=top_k, threshold=threshold, tags=tags, rerank=rerank)
        return [r.text for r in results]

    def retrieve_by_tier(
        self,
        query: str,
        tier: str,  # 'short'|'long'
        top_k: int = 5,
        threshold: Optional[float] = None,
        rerank: bool = True,
    ) -> List[SearchResult]:
        filters: Dict[str, Any] = {"tier": tier}
        # Preserve entity scoping
        if self.entity_id:
            filters["entity_id"] = self.entity_id
        if self.entity_type:
            filters["entity_type"] = self.entity_type
        return self.adapter.search(query=query, top_k=top_k, threshold=threshold, filters=filters, rerank=rerank)

    # ----------------------- Expiration & cleanup ----------------------
    def delete_expired(self) -> int:
        return self.adapter.delete_expired()

    def clear_all(self) -> None:
        try:
            # Best-effort clear: delete all by timestamp <= now
            from datetime import datetime
            now = datetime.now().isoformat()
            self.adapter.delete({"timestamp": {"$lte": now}})
        except Exception as e:
            logger.error("Failed to clear memories: %s", e)

    # ----------------------- Tag-based helpers ------------------------
    def search_by_tags(
        self,
        tags: List[str],
        query: str = "",
        top_k: int = 5,
        threshold: Optional[float] = None,
        rerank: bool = True,
    ) -> List[SearchResult]:
        return self.adapter.search_by_tags(tags=tags, query=query, top_k=top_k, threshold=threshold, rerank=rerank)

    # ----------------------- Export -----------------------------------
    def export(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self.adapter.export(filters=filters)

    # ----------------------- Controls ---------------------------------
    def set_auto_classification_enabled(self, enabled: bool) -> None:
        self.auto_classification_enabled = enabled

    def set_classifier(self, classifier: Mem0IngestionClassifier) -> None:
        self.classifier = classifier

    def update_category_routing(self, routing: Dict[str, Dict[str, Optional[int]]]) -> None:
        self.category_routing = routing

    def set_auto_entity_extraction_enabled(self, enabled: bool) -> None:
        self.auto_entity_extraction_enabled = enabled

    def set_entity_extractor(self, extractor: Mem0EntityExtractor) -> None:
        self.entity_extractor = extractor

    def set_entity_assignment_mode(self, mode: str) -> None:
        if mode in ("manager", "extract", "merge"):
            self.entity_assignment_mode = mode

    def set_default_ttl_seconds(self, seconds: Optional[int]) -> None:
        self.default_ttl_seconds = seconds
        self.adapter.default_ttl_seconds = seconds

    def set_short_term_ttl_seconds(self, seconds: Optional[int]) -> None:
        self.short_term_ttl_seconds = seconds
        self.adapter.short_term_ttl_seconds = seconds
