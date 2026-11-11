"""Behavior checks shared by the old and upgraded LangChain environments.

Models are deterministic and local. The HANA case uses a unique synthetic table
and runs only when HANA_USERKEY is explicitly configured.
"""
import os
import uuid
from unittest.mock import Mock

import pytest
from hana_ml.dataframe import ConnectionContext, DataFrame
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, FunctionMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hana_ai import langchain_compat as compat
from hana_ai.agents.hana_dataframe_agent import create_hana_dataframe_agent
from hana_ai.agents.hanaml_agent_with_memory import HANAMLAgentWithMemory
from hana_ai.mem0.hana_mem0_adapter import Mem0HanaAdapter


class _CaptureMessages(BaseCallbackHandler):
    def __init__(self):
        self.calls = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.calls.extend(messages)


def test_function_agent_calls_tool_and_preserves_memory_and_output():
    calls = []

    @tool
    def lookup(value: str) -> str:
        """Look up a known test value."""
        calls.append(value)
        return "answer=42"

    capture = _CaptureMessages()
    model = FakeMessagesListChatModel(responses=[
        AIMessage(content="", additional_kwargs={"function_call": {
            "name": "lookup", "arguments": '{"value": "sample"}'}}),
        AIMessage(content="The answer is 42."),
        AIMessage(content="You asked about sample."),
    ], callbacks=[capture])
    memory = compat.get_conversation_buffer_window_memory(
        k=2, memory_key="chat_history", return_messages=True,
        input_key="input", output_key="output")
    prompt = compat.ChatPromptTemplate.from_messages([
        ("system", "Use the lookup tool."),
        compat.MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        compat.MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    executor = compat.build_agent_executor(
        model, [lookup], prompt=prompt, memory=memory,
        return_intermediate_steps=True, max_iterations=3)
    first = executor.invoke({"input": "Find sample"})
    second = executor.invoke({"input": "What did I ask?"})
    assert first["output"] == "The answer is 42."
    assert calls == ["sample"]
    assert first["intermediate_steps"][0][1] == "answer=42"
    assert any(isinstance(m, FunctionMessage) and m.content == "answer=42"
               for m in capture.calls[1])
    assert [m.content for m in capture.calls[2]][1:3] == ["Find sample", "The answer is 42."]
    assert second["output"] == "You asked about sample."
    assert len(memory.load_memory_variables({})["chat_history"]) == 4


def test_memory_window_keeps_complete_turns_and_can_clear():
    memory = compat.get_conversation_buffer_window_memory(
        k=2, memory_key="history", return_messages=True)
    for number in range(3):
        memory.save_context({"input": f"question {number}"}, {"output": f"answer {number}"})
    assert [m.content for m in memory.load_memory_variables({})["history"]] == [
        "question 1", "answer 1", "question 2", "answer 2"]
    memory.clear()
    assert memory.load_memory_variables({}) == {"history": []}


def test_dataframe_agent_preserves_tool_steps_and_iteration_limit():
    calls = []

    @tool
    def inspect_columns(value: str) -> str:
        """Inspect test schema metadata."""
        calls.append(value)
        return "AMOUNT"

    frame = Mock(spec=DataFrame)
    frame.columns = ["AMOUNT"]
    model = FakeListLLM(responses=[
        "Thought: inspect the schema\nAction: inspect_columns\nAction Input: columns",
        "Final Answer: AMOUNT",
    ])
    executor = create_hana_dataframe_agent(
        model, frame, tools=[inspect_columns], return_intermediate_steps=True, max_iterations=2)
    result = executor.invoke({"input": "List the columns"})
    assert result["output"] == "AMOUNT"
    assert calls == ["columns"]
    assert result["intermediate_steps"][0][1] == "AMOUNT"
    assert executor.max_iterations == 2


def test_structured_memory_agent_preserves_public_run_api():
    model = FakeListLLM(responses=[
        'Thought: answer\nAction:\n```json\n{"action": "Final Answer", "action_input": "Hello"}\n```',
        'Thought: answer\nAction:\n```json\n{"action": "Final Answer", "action_input": "Welcome back"}\n```',
    ])
    agent = HANAMLAgentWithMemory(model, [], session_id="dependency-test")
    assert agent.run("Hello") == "Hello"
    assert agent.run("Hello again") == "Welcome back"
    assert any(m.content == "Hello" for m in agent.memory.messages)
    assert any(m.content == "Welcome back" for m in agent.memory.messages)


def test_document_splitting_preserves_content_and_metadata():
    splitter = RecursiveCharacterTextSplitter(chunk_size=12, chunk_overlap=0, separators=[" "],
                                             add_start_index=True)
    text = "alpha beta gamma delta epsilon"
    documents = splitter.create_documents([text], metadatas=[{"source": "sample"}])
    assert [d.page_content for d in documents] == ["alpha beta", "gamma delta", "epsilon"]
    assert [d.metadata for d in documents] == [
        {"source": "sample", "start_index": 0},
        {"source": "sample", "start_index": 11},
        {"source": "sample", "start_index": 23},
    ]


class _FixedEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0] if "apple" in text else [0.0, 1.0, 0.0]


@pytest.mark.skipif(not os.environ.get("HANA_USERKEY"), reason="HANA_USERKEY is not configured")
def test_live_hana_memory_ingest_search_filter_and_delete():
    connection = ConnectionContext(userkey=os.environ["HANA_USERKEY"], encrypt=True,
                                   sslValidateCertificate=True)
    table_name = "SEC_DEP_VECTOR_" + uuid.uuid4().hex.upper()
    created = False
    try:
        adapter = Mem0HanaAdapter(connection, table_name, embedder=_FixedEmbeddings())
        created = True
        adapter.add_texts(["apple for Alice"], metadata={"tenant_id": "alice"})
        adapter.add_texts(["apple for Bob"], metadata={"tenant_id": "bob"})
        results = adapter.search("apple", filters={"tenant_id": "alice"}, rerank=False)
        assert [r.text for r in results] == ["apple for Alice"]
        adapter.delete({"tenant_id": "alice"})
        assert adapter.search("apple", filters={"tenant_id": "alice"}, rerank=False) == []
        assert [r.text for r in adapter.search("apple", filters={"tenant_id": "bob"}, rerank=False)] == [
            "apple for Bob"]
    finally:
        try:
            if created:
                connection.drop_table(table_name)
        finally:
            connection.close()
