import unittest
from datetime import datetime

from hana_ai.agents.mem0_hana_agent import Mem0HANARAGAgent
from hana_ai.mem0.hana_mem0_adapter import Mem0HanaAdapter


class FakeDoc:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}


class FakeVectorStore:
    def __init__(self):
        self.items = []

    def add_documents(self, docs):
        self.items.extend(docs)
        return [str(i) for i in range(len(self.items) - len(docs), len(self.items))]

    def similarity_search_with_relevance_scores(self, query, k=5, score_threshold=0.0, filter=None):
        def score(text):
            q = set(query.lower().split())
            t = set(text.lower().split())
            if not q:
                return 0.0
            return len(q & t) / max(1, len(q))
        cands = []
        for d in self.items:
            if filter and any(
                d.metadata.get(key) != value
                for key, value in filter.items()
                if not isinstance(value, dict)
            ):
                continue
            s = score(d.page_content)
            if s >= score_threshold:
                cands.append((d, s))
        cands.sort(key=lambda x: x[1], reverse=True)
        return cands[:k]

    def delete(self, filter):
        count = 0
        keep = []
        for d in self.items:
            matches = True
            for key, value in filter.items():
                actual = d.metadata.get(key)
                if isinstance(value, dict) and '$lte' in value:
                    matches = matches and actual is not None and actual <= value['$lte']
                else:
                    matches = matches and actual == value
            if matches:
                count += 1
            else:
                keep.append(d)
        self.items = keep
        return count


class FakeReranker:
    def predict(self, pairs):
        # boost exact "apples" match
        return [1.0 if 'apples' in doc.lower() else 0.5 for q, doc in pairs]


class TestMem0HanaAgent(unittest.TestCase):
    def setUp(self):
        vs = FakeVectorStore()
        adapter = Mem0HanaAdapter(
            connection_context=None,
            table_name=None,
            embedder=None,
            reranker=FakeReranker(),
            vectorstore=vs,
            score_threshold=0.0,
        )
        # tools empty, llm not needed for test (auto_init_agent=False)
        self.agent = Mem0HANARAGAgent(
            tools=[],
            llm=None,
            memory_window=3,
            rerank_candidates=5,
            rerank_k=2,
            score_threshold=0.0,
            tenant_id="tenant-a",
            session_id="session-a",
            _adapter=adapter,
            _auto_init_agent=False,
        )
        # preload some memories via adapter
        now = datetime.now().isoformat()
        metadata = {
            "timestamp": now,
            "entity_id": "tenant-a",
            "entity_type": "tenant",
        }
        self.agent.mem0_adapter.add([
            {"text": "User mentioned apples and bananas", "metadata": metadata},
            {"text": "Assistant suggested fruits and snacks", "metadata": metadata},
            {"text": "User bought oranges", "metadata": metadata},
        ])

    def test_retrieve_relevant_memories(self):
        ctx = self.agent._retrieve_relevant_memories("apples")
        self.assertEqual(len(ctx), 2)
        self.assertTrue(any("apples" in c.lower() for c in ctx))

    def test_update_long_term_memory(self):
        self.agent._update_long_term_memory("User asks about apples", "We discuss apples.")
        ctx = self.agent._retrieve_relevant_memories("apples")
        self.assertTrue(any("We discuss apples" in c for c in ctx))

    def test_clear_long_term_memory(self):
        self.agent.clear_long_term_memory()
        ctx = self.agent._retrieve_relevant_memories("apples")
        self.assertEqual(len(ctx), 0)


if __name__ == "__main__":
    unittest.main()
