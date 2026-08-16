"""Shared fakes for offline testing — no network, no real FAISS index/model
download needed. FaissVectorStore tests build a tiny real in-memory FAISS
index instead of faking faiss itself, since faiss-cpu is already a fast,
local dependency.
"""
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest


class FakeLLMProvider:
    """Scriptable LLMProvider: pop() one scripted response per .chat() call.

    Each scripted item is either:
      - a callable(messages) -> list of "chunks" (for stream=True), or
      - a callable(messages) -> single message-like object (for stream=False)
    Records every call's kwargs so tests can assert on what was sent (e.g.
    to verify with_no_think() was applied).
    """

    def __init__(self, script: List[Any]):
        self._script = list(script)
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, *, tools=None, response_format=None, stream=False):
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "response_format": response_format,
            "stream": stream,
        })
        if not self._script:
            raise AssertionError("FakeLLMProvider script exhausted")
        item = self._script.pop(0)
        result = item(messages) if callable(item) else item

        if stream:
            for chunk in result:
                yield chunk
        else:
            yield result


def make_chunk(content: Optional[str] = None, tool_calls: Optional[list] = None):
    """Builds a fake OpenAI-SDK-shaped streaming chunk."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


def make_tool_call_delta(index: int, call_id: str = "", name: str = "", arguments: str = ""):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, function=function)


def make_message(content: str):
    return SimpleNamespace(content=content)


class FakeEmbeddingProvider:
    def __init__(self, vector: Optional[List[float]] = None):
        self._vector = vector if vector is not None else [1.0, 0.0, 0.0]

    def embed(self, text: str) -> List[float]:
        return list(self._vector)


class FakeReranker:
    """Returns scores in the given order; defaults to identity (1.0 each)."""

    def __init__(self, scores: Optional[List[float]] = None):
        self._scores = scores

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        if self._scores is not None:
            return list(self._scores)
        return [1.0] * len(documents)


class FakeVectorStore:
    """In-memory VectorStore: chunk_id -> (faiss_id, score-if-searched)."""

    def __init__(self, hits: List[Dict[str, Any]]):
        # hits: list of {"chunk_id": ..., "faiss_id": ..., "score": ...}
        self._hits = hits
        self._by_chunk_id = {h["chunk_id"]: h for h in hits}
        self._by_faiss_id = {h["faiss_id"]: h for h in hits}

    @property
    def total(self) -> int:
        return len(self._hits)

    def chunk_id_for(self, faiss_id: int) -> Optional[str]:
        h = self._by_faiss_id.get(faiss_id)
        return h["chunk_id"] if h else None

    def faiss_id_for(self, chunk_id: str) -> Optional[int]:
        h = self._by_chunk_id.get(chunk_id)
        return h["faiss_id"] if h else None

    def search(self, vector, top_k):
        ordered = sorted(self._hits, key=lambda h: h["score"], reverse=True)
        return [{"chunk_id": h["chunk_id"], "score": h["score"]} for h in ordered[:top_k]]

    def search_subset(self, vector, candidate_faiss_ids, top_k):
        candidates = [h for h in self._hits if h["faiss_id"] in candidate_faiss_ids]
        candidates.sort(key=lambda h: h["score"], reverse=True)
        return [{"chunk_id": h["chunk_id"], "score": h["score"]} for h in candidates[:top_k]]


@pytest.fixture
def required_env(monkeypatch):
    """Sets the minimum env vars Settings() needs to construct without error."""
    monkeypatch.setenv("CHAT_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("CHAT_API_KEY", "test-key")
    monkeypatch.setenv("CHAT_MODEL_NAME", "test-model")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "test-embedding-model")
