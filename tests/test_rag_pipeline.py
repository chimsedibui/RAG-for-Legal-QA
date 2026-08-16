import json

from core.config import RetrievalSettings
from services.rag_pipeline import RAGPipeline
from services.search import SemanticSearchService
from tests.conftest import (
    FakeEmbeddingProvider,
    FakeLLMProvider,
    FakeReranker,
    FakeVectorStore,
    make_chunk,
    make_message,
    make_tool_call_delta,
)
from tools.registry import ToolRegistry


def _settings(**overrides):
    defaults = dict(
        threshold=0.0, semantic_top_k=10, tool_search_top_k=5,
        max_context_chunks=50, max_tool_iterations=3,
    )
    defaults.update(overrides)
    return RetrievalSettings(**{
        "RETRIEVAL_THRESHOLD": defaults["threshold"],
        "SEMANTIC_TOP_K": defaults["semantic_top_k"],
        "TOOL_SEARCH_TOP_K": defaults["tool_search_top_k"],
        "MAX_CONTEXT_CHUNKS": defaults["max_context_chunks"],
        "MAX_TOOL_ITERATIONS": defaults["max_tool_iterations"],
    })


def _make_semantic_search(hits=None, chunks_text_map=None):
    hits = hits if hits is not None else [{"chunk_id": "c1", "faiss_id": 0, "score": 1.0}]
    chunks_text_map = chunks_text_map if chunks_text_map is not None else {"c1": "noi dung c1"}
    return SemanticSearchService(
        embedder=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(hits),
        reranker=FakeReranker(),
        chunk_map={"c1": {"title": "T"}},
        chunks_text_map=chunks_text_map,
        threshold=0.0,
    )


def _sub_query_response(queries):
    return lambda messages: make_message(json.dumps({"queries": queries}))


class _FakeTool:
    name = "search_referenced_document"
    schema = {"name": "search_referenced_document", "description": "d", "parameters": {}}

    def __init__(self, docs=None):
        self._docs = docs if docs is not None else [{"chunk_id": "extra", "content": "extra content"}]
        self.calls = []

    def execute(self, args, *, question):
        self.calls.append(args)
        return self._docs


def _collect(gen):
    return list(gen)


# ---------------------------------------------------------------------------
# (a) no tool call -> terminal event must include sources (regression test)
# ---------------------------------------------------------------------------

def test_answer_done_event_includes_sources():
    llm = FakeLLMProvider([
        _sub_query_response(["question 1"]),
        lambda messages: [make_chunk(content="Xin chao")],
    ])
    pipeline = RAGPipeline(llm, _make_semantic_search(), ToolRegistry(), _settings())

    events = _collect(pipeline.process([{"role": "user", "content": "hoi gi do"}], stream=True))

    done_events = [e for e in events if e["step"] == "answer" and e["status"] == "done"]
    assert len(done_events) == 1
    data = done_events[0]["data"]
    assert data["text"] == "Xin chao"
    assert "sources" in data
    assert data["sources"] == [{"chunk_id": "c1", "score": 1.0, "metadata": {"title": "T"}, "content": "noi dung c1"}]


# ---------------------------------------------------------------------------
# (b) one tool-call iteration -> event order + tool message appended
# ---------------------------------------------------------------------------

def test_tool_call_iteration_emits_expected_events_and_history():
    tool = _FakeTool()
    registry = ToolRegistry()
    registry.register(tool)

    llm = FakeLLMProvider([
        _sub_query_response(["q1"]),
        lambda messages: [
            make_chunk(tool_calls=[make_tool_call_delta(0, "call_1", "search_referenced_document", '{"doc_ref": "1/2020/ND-CP", "content_query": "abc"}')]),
        ],
        lambda messages: [make_chunk(content="Cau tra loi cuoi cung")],
    ])
    pipeline = RAGPipeline(llm, _make_semantic_search(), registry, _settings())

    events = _collect(pipeline.process([{"role": "user", "content": "hoi gi do"}], stream=True))

    steps = [(e["step"], e["status"]) for e in events]
    assert ("tool_call", "detected") in steps
    assert ("tool_call", "executed") in steps
    assert steps.count(("context_ready", "done")) == 2  # initial retrieval + post-tool-call
    assert tool.calls == [{"doc_ref": "1/2020/ND-CP", "content_query": "abc"}]

    # second LLM call should carry the tool result in history
    second_call_messages = llm.calls[2]["messages"]
    roles = [m.get("role") for m in second_call_messages]
    assert "tool" in roles

    final = [e for e in events if e["step"] == "answer" and e["status"] == "done"][0]
    assert final["data"]["text"] == "Cau tra loi cuoi cung"


# ---------------------------------------------------------------------------
# (c) unknown tool name -> graceful fallback, no crash
# ---------------------------------------------------------------------------

def test_unknown_tool_name_is_handled_gracefully():
    llm = FakeLLMProvider([
        _sub_query_response(["q1"]),
        lambda messages: [
            make_chunk(tool_calls=[make_tool_call_delta(0, "call_1", "some_unregistered_tool", "{}")]),
        ],
        lambda messages: [make_chunk(content="OK")],
    ])
    pipeline = RAGPipeline(llm, _make_semantic_search(), ToolRegistry(), _settings())

    events = _collect(pipeline.process([{"role": "user", "content": "hoi"}], stream=True))
    final = [e for e in events if e["step"] == "answer" and e["status"] == "done"][0]
    assert final["data"]["text"] == "OK"

    second_call_messages = llm.calls[2]["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert any("không được hỗ trợ" in m["content"] for m in tool_messages)


# ---------------------------------------------------------------------------
# (d) with_no_think applied to what's sent to the LLM, never mutating history
# ---------------------------------------------------------------------------

def test_with_no_think_applied_without_mutating_history():
    llm = FakeLLMProvider([
        _sub_query_response(["q1"]),
        lambda messages: [make_chunk(content="OK")],
    ])
    pipeline = RAGPipeline(llm, _make_semantic_search(), ToolRegistry(), _settings())

    _collect(pipeline.process([{"role": "user", "content": "cau hoi"}], stream=True))

    # Every user message sent to the LLM should end with /no_think.
    for call in llm.calls:
        for m in call["messages"]:
            if m.get("role") == "user":
                assert m["content"].endswith("/no_think")

    # The original conversation dict passed by the caller must be untouched.
    original = [{"role": "user", "content": "cau hoi"}]
    assert original[0]["content"] == "cau hoi"


# ---------------------------------------------------------------------------
# (e) MAX_TOOL_ITERATIONS exhaustion still yields a terminal answer/done
# ---------------------------------------------------------------------------

def test_max_tool_iterations_exhaustion_still_yields_terminal_event():
    tool = _FakeTool()
    registry = ToolRegistry()
    registry.register(tool)

    def always_tool_call(messages):
        return [make_chunk(tool_calls=[make_tool_call_delta(0, "call_x", "search_referenced_document", '{"doc_ref": "1/2020/ND-CP", "content_query": "x"}')])]

    llm = FakeLLMProvider([
        _sub_query_response(["q1"]),
        always_tool_call,
        always_tool_call,
        always_tool_call,
    ])
    pipeline = RAGPipeline(llm, _make_semantic_search(), registry, _settings(max_tool_iterations=3))

    events = _collect(pipeline.process([{"role": "user", "content": "hoi"}], stream=True))
    final = [e for e in events if e["step"] == "answer" and e["status"] == "done"]
    assert len(final) == 1
    assert final[0]["data"]["text"] == ""
