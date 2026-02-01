import unittest
from datetime import datetime, timedelta
import hashlib

from hana_ai.mem0.hana_mem0_adapter import Mem0HanaAdapter, SearchResult


class FakeDoc:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}


class FakeVectorStore:
    """A minimal in-memory vectorstore for tests."""
    def __init__(self):
        self.items = []  # list of (FakeDoc)

    def add_documents(self, docs):
        self.items.extend(docs)
        # Return placeholder IDs
        return [str(i) for i in range(len(self.items) - len(docs), len(self.items))]

    def similarity_search_with_relevance_scores(self, query, k=5, score_threshold=0.0, filter=None):
        def score(text):
            # naive score: proportion of query tokens contained in text
            q_tokens = set(query.lower().split())
            t_tokens = set(text.lower().split())
            if not q_tokens:
                return 0.0
            return len(q_tokens & t_tokens) / max(1, len(q_tokens))

        candidates = []
        for doc in self.items:
            s = score(doc.page_content)
            if s >= score_threshold:
                # apply timestamp filter if provided
                if filter and "timestamp" in filter:
                    ts_filter = filter["timestamp"]
                    # support lte
                    if "$lte" in ts_filter:
                        if doc.metadata.get("timestamp", "9999") > ts_filter["$lte"]:
                            continue
                candidates.append((doc, s))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def delete(self, filter):
        # support timestamp lte deletion
        count = 0
        key = None
        cutoff = None
        if "timestamp" in filter and "$lte" in filter["timestamp"]:
            key = "timestamp"
            cutoff = filter["timestamp"]["$lte"]
        if "expires_at" in filter and "$lte" in filter["expires_at"]:
            key = "expires_at"
            cutoff = filter["expires_at"]["$lte"]
        if key and cutoff:
            keep = []
            for doc in self.items:
                if doc.metadata.get(key, "9999") <= cutoff:
                    count += 1
                else:
                    keep.append(doc)
            self.items = keep
        return count


class FakeReranker:
    def predict(self, pairs):
        # Higher score if query substring appears in doc
        scores = []
        for q, d in pairs:
            scores.append(1.0 if q.lower() in d.lower() else 0.0)
        return scores


class TestMem0HanaAdapter(unittest.TestCase):
    def setUp(self):
        self.vs = FakeVectorStore()
        self.adapter = Mem0HanaAdapter(
            connection_context=None,  # not used when vectorstore injected
            table_name=None,
            embedder=None,
            reranker=FakeReranker(),
            vectorstore=self.vs,
            score_threshold=0.0,
            ingest_filter=lambda text, md: "skip" not in text,
            max_length=1000,
            default_ttl_seconds=60,
            short_term_ttl_seconds=10,
            partition_defaults={"agent_id": "agentA"},
        )

    def test_add_and_search_basic(self):
        ids = self.adapter.add([
            {"text": "User bought apples and bananas", "metadata": {"source": "s1"}},
            {"text": "Assistant suggested fruits and healthy snacks", "metadata": {"source": "s2"}},
            {"text": "User visited the store yesterday", "metadata": {"source": "s3"}},
        ], user_id="u1")
        self.assertEqual(len(ids), 3)

        results = self.adapter.search(query="apples bananas", top_k=2, rerank=True)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SearchResult)
        # reranker should prefer exact substring match
        self.assertIn("apples", results[0].text.lower())
        self.assertGreaterEqual(results[0].rerank_score or 0.0, results[-1].rerank_score or 0.0)

    def test_search_threshold_and_filters(self):
        now = datetime.now().isoformat()
        past = (datetime.now() - timedelta(days=1)).isoformat()
        self.adapter.add([
            {"text": "Order includes apples", "metadata": {"timestamp": past}, "tags": ["order", "fruit"]},
            {"text": "Recommendation: bananas", "metadata": {"timestamp": now}, "tags": ["rec", "fruit"]},
        ])
        # filter out items newer than 'past' by lte on timestamp
        results = self.adapter.search(
            query="apples",
            top_k=5,
            threshold=0.0,
            filters={"timestamp": {"$lte": past}},
            rerank=False,
        )
        # only the 'apples' doc should pass the filter
        self.assertEqual(len(results), 1)
        self.assertIn("apples", results[0].text)
        # tag search wrapper
        tag_res = self.adapter.search_by_tags(["order"], query="apples", top_k=5, rerank=False)
        self.assertTrue(any("apples" in r.text for r in tag_res))

    def test_delete_by_timestamp(self):
        past = (datetime.now() - timedelta(days=7)).isoformat()
        mid = (datetime.now() - timedelta(days=3)).isoformat()
        now = datetime.now().isoformat()
        self.adapter.add([
            {"text": "old memory", "metadata": {"timestamp": past}},
            {"text": "mid memory", "metadata": {"timestamp": mid}},
            {"text": "new memory", "metadata": {"timestamp": now}},
        ])
        deleted = self.adapter.delete({"timestamp": {"$lte": mid}})
        self.assertEqual(deleted, 2)
        # ensure only "new memory" remains
        remaining = self.adapter.search(query="memory", top_k=10, rerank=False)
        self.assertEqual(len(remaining), 1)
        self.assertIn("new", remaining[0].text)

    def test_update_emulation(self):
        ids = self.adapter.add([{ "text": "initial", "metadata": {} }])
        ok = self.adapter.update(id=ids[0] if ids else "0", new_text="updated", metadata={"source": "s"})
        self.assertTrue(ok)
        res = self.adapter.search(query="updated", top_k=5, rerank=False)
        self.assertTrue(any("updated" in r.text for r in res))

    def test_ingest_control_ttl_and_hash(self):
        # skipped by filter
        ids = self.adapter.add([{ "text": "please skip this", "metadata": {} }])
        self.assertEqual(len(ids), 0)
        # short term tier uses short ttl
        now = datetime.now().isoformat()
        self.adapter.add([{ "text": "short term note", "tier": "short", "metadata": {"timestamp": now}}])
        res = self.adapter.search(query="short", top_k=5, rerank=False)
        self.assertTrue(any("short" in r.text for r in res))
        # content hash present
        ids = self.adapter.add([{ "text": "hash me", "metadata": {} }])
        res = self.adapter.search(query="hash", top_k=5, rerank=False)
        self.assertTrue(any("content_hash" in r.metadata for r in res))

    def test_delete_expired(self):
        now = datetime.now()
        past = (now - timedelta(seconds=120)).isoformat()
        near = (now - timedelta(seconds=30)).isoformat()
        # create one already expired and one not
        self.adapter.add([
            {"text": "expired note", "metadata": {"timestamp": past}, "ttl_seconds": 30},
            {"text": "fresh note", "metadata": {"timestamp": near}, "ttl_seconds": 3600},
        ])
        # delete expired
        deleted = self.adapter.delete_expired(now_iso=now.isoformat())
        self.assertGreaterEqual(deleted, 1)


if __name__ == "__main__":
    unittest.main()
