import json
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from providers.faiss_store import FaissVectorStore
from providers.openai_embedding import OpenAIEmbeddingProvider
from providers.openai_llm import OpenAILLMProvider
from providers.reranker import NullReranker, VLLMReranker


# ---------------------------------------------------------------------------
# OpenAILLMProvider
# ---------------------------------------------------------------------------

class _FakeChatCompletions:
    def __init__(self, result, raise_error=False):
        self._result = result
        self._raise_error = raise_error

    def create(self, **kwargs):
        if self._raise_error:
            raise RuntimeError("boom")
        return self._result


def _make_llm(monkeypatch, result=None, raise_error=False):
    provider = OpenAILLMProvider("http://localhost:1234/v1", "key", "model")
    provider._client.chat = SimpleNamespace(
        completions=_FakeChatCompletions(result, raise_error=raise_error)
    )
    return provider


def test_llm_provider_non_stream_yields_single_message(monkeypatch):
    message = SimpleNamespace(content="hello")
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    llm = _make_llm(monkeypatch, result=response)

    results = list(llm.chat(messages=[{"role": "user", "content": "hi"}], stream=False))
    assert results == [message]


def test_llm_provider_stream_yields_each_chunk(monkeypatch):
    chunks = [SimpleNamespace(choices=[]), SimpleNamespace(choices=[])]
    llm = _make_llm(monkeypatch, result=iter(chunks))

    results = list(llm.chat(messages=[{"role": "user", "content": "hi"}], stream=True))
    assert results == chunks


def test_llm_provider_stream_error_yields_error_dict(monkeypatch):
    llm = _make_llm(monkeypatch, raise_error=True)
    results = list(llm.chat(messages=[{"role": "user", "content": "hi"}], stream=True))
    assert len(results) == 1
    assert "error" in results[0]


def test_llm_provider_non_stream_error_raises(monkeypatch):
    llm = _make_llm(monkeypatch, raise_error=True)
    with pytest.raises(Exception):
        list(llm.chat(messages=[{"role": "user", "content": "hi"}], stream=False))


# ---------------------------------------------------------------------------
# OpenAIEmbeddingProvider
# ---------------------------------------------------------------------------

def test_embedding_provider_returns_vector(monkeypatch):
    provider = OpenAIEmbeddingProvider("http://localhost:1234/v1", "key", "model")
    fake_response = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])
    provider._client.embeddings = SimpleNamespace(create=lambda **kw: fake_response)

    assert provider.embed("hello") == [0.1, 0.2, 0.3]


def test_embedding_provider_returns_empty_list_on_error(monkeypatch):
    provider = OpenAIEmbeddingProvider("http://localhost:1234/v1", "key", "model")

    def _raise(**kw):
        raise RuntimeError("boom")

    provider._client.embeddings = SimpleNamespace(create=_raise)
    assert provider.embed("hello") == []


# ---------------------------------------------------------------------------
# VLLMReranker / NullReranker
# ---------------------------------------------------------------------------

def test_null_reranker_returns_identity_scores():
    reranker = NullReranker()
    assert reranker.rerank("q", ["a", "b", "c"]) == [1.0, 1.0, 1.0]


def test_null_reranker_handles_empty_documents():
    assert NullReranker().rerank("q", []) == []


def test_vllm_reranker_maps_scores_by_index(monkeypatch):
    reranker = VLLMReranker("http://localhost:9999/v1", "key", "reranker-model")

    payload = {
        "id": "1",
        "model": "reranker-model",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
        "results": [
            {"index": 1, "document": {"text": "b"}, "relevance_score": 0.9},
            {"index": 0, "document": {"text": "a"}, "relevance_score": 0.3},
        ],
    }

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr("providers.reranker.requests.post", lambda *a, **kw: _FakeResponse())

    scores = reranker.rerank("q", ["a", "b"])
    assert scores == [0.3, 0.9]


def test_vllm_reranker_defaults_missing_index_to_zero(monkeypatch):
    reranker = VLLMReranker("http://localhost:9999/v1", "key", "reranker-model")

    payload = {
        "id": "1",
        "model": "reranker-model",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
        "results": [{"index": 0, "document": {"text": "a"}, "relevance_score": 0.9}],
    }

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr("providers.reranker.requests.post", lambda *a, **kw: _FakeResponse())

    scores = reranker.rerank("q", ["a", "b"])
    assert scores == [0.9, 0.0]


def test_vllm_reranker_falls_back_to_identity_on_error(monkeypatch):
    reranker = VLLMReranker("http://localhost:9999/v1", "key", "reranker-model")

    def _raise(*a, **kw):
        raise RuntimeError("network error")

    monkeypatch.setattr("providers.reranker.requests.post", _raise)
    assert reranker.rerank("q", ["a", "b"]) == [1.0, 1.0]


# ---------------------------------------------------------------------------
# FaissVectorStore — built against a real tiny in-memory index (fast, no
# download needed; faiss-cpu is already a runtime dependency).
# ---------------------------------------------------------------------------

@pytest.fixture
def small_faiss_store(tmp_path):
    dim = 4
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # faiss_id 0 -> chunk "a"
            [0.0, 1.0, 0.0, 0.0],  # faiss_id 1 -> chunk "b"
            [0.9, 0.1, 0.0, 0.0],  # faiss_id 2 -> chunk "c" (close to query below)
        ],
        dtype=np.float32,
    )
    ids = np.array([0, 1, 2], dtype=np.int64)
    index.add_with_ids(vectors, ids)

    index_path = tmp_path / "faiss.index"
    id_map_path = tmp_path / "faiss_id_map.json"
    faiss.write_index(index, str(index_path))
    id_map_path.write_text(json.dumps({"0": "a", "1": "b", "2": "c"}))

    return FaissVectorStore(str(index_path), str(id_map_path))


def test_faiss_store_search_returns_ranked_hits(small_faiss_store):
    hits = small_faiss_store.search([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert [h["chunk_id"] for h in hits] == ["a", "c"]


def test_faiss_store_total_and_id_translation(small_faiss_store):
    assert small_faiss_store.total == 3
    assert small_faiss_store.chunk_id_for(1) == "b"
    assert small_faiss_store.faiss_id_for("b") == 1
    assert small_faiss_store.chunk_id_for(999) is None


def test_faiss_store_search_subset_filters_to_candidates(small_faiss_store):
    hits = small_faiss_store.search_subset([1.0, 0.0, 0.0, 0.0], candidate_faiss_ids={1, 2}, top_k=5)
    assert [h["chunk_id"] for h in hits] == ["c", "b"]
