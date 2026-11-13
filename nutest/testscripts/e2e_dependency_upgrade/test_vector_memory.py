"""Real embeddings, HANA retrieval, reconnection, and scoped deletion."""
import json
import os


def test_real_embedding_memory_survives_reconnect_and_scoped_delete(hana_case, tmp_path):
    from hana_ml import dataframe
    from hana_ai.mem0.memory_manager import Mem0MemoryManager
    from hana_ai.vectorstore.embedding_service import GenAIHubEmbeddings
    cc, prefix = hana_case
    table = prefix + "_MEMORY"
    model = os.environ.get("HANA_E2E_EMBEDDING_MODEL", "text-embedding-ada-002")
    embedder = GenAIHubEmbeddings(model_id=model)
    text = "The refund approval limit for the North store is 731 euros."
    bob_text = "The refund approval limit for the South store is 942 euros."
    other_text = "The warehouse forklift maintenance appointment is on Tuesday."

    def manager(connection, tenant, session="session-1"):
        return Mem0MemoryManager(connection, table, embedder=embedder,
                                 tenant_id=tenant, session_id=session)

    alice = manager(cc, "alice")
    alice.add_memory(text, extra_metadata={"source": "refund-policy"})
    alice.add_memory(other_text)
    manager(cc, "bob").add_memory(bob_text)
    manager(cc, "alice", "session-2").add_memory("Another session has a refund limit of 123 euros.")
    query = "What is the refund approval limit for the North store?"
    results = alice.retrieve(query, top_k=1, rerank=False)
    assert [row.text for row in results] == [text]
    assert results[0].metadata["source"] == "refund-policy"
    assert results[0].metadata["tenant_id"] == "alice"
    assert results[0].metadata["session_id"] == "session-1"
    # A new database connection and manager must read persisted vectors, not an
    # in-process cache. Exercise the public manager, not deprecated RAG agents.
    reconnected = dataframe.ConnectionContext(userkey=os.environ["HANA_USERKEY"],
                                              encrypt=True, sslValidateCertificate=True)
    try:
        restored = manager(reconnected, "alice")
        assert restored.retrieve_texts(query, top_k=1, rerank=False) == [text]
        restored.clear_all()
        assert restored.retrieve_texts(query, rerank=False) == []
        assert manager(reconnected, "bob").retrieve_texts(bob_text, rerank=False) == [bob_text]
        assert manager(reconnected, "alice", "session-2").retrieve_texts(
            "refund limit", rerank=False) == ["Another session has a refund limit of 123 euros."]
        assert alice.retrieve_texts(query, rerank=False) == []
    finally:
        reconnected.close()
    (tmp_path / "memory.json").write_text(json.dumps(dict(
        embedding_model=model, expected_top_hit=text, actual_top_hit=results[0].text,
        score=results[0].score, reconnect_verified=True, scoped_delete_verified=True,
    ), indent=2), encoding="utf-8")
