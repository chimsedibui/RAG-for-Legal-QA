import pytest

from services.search import DocRefSearchService, SemanticSearchService
from tests.conftest import FakeEmbeddingProvider, FakeReranker, FakeVectorStore


# ---------------------------------------------------------------------------
# SemanticSearchService
# ---------------------------------------------------------------------------

def _make_semantic_service(hits, reranker_scores=None, threshold=0.5, chunk_map=None, chunks_text_map=None):
    vector_store = FakeVectorStore(hits)
    return SemanticSearchService(
        embedder=FakeEmbeddingProvider(),
        vector_store=vector_store,
        reranker=FakeReranker(reranker_scores),
        chunk_map=chunk_map or {},
        chunks_text_map=chunks_text_map or {},
        threshold=threshold,
    )


def test_semantic_search_filters_below_threshold():
    hits = [
        {"chunk_id": "a", "faiss_id": 0, "score": 0.9},
        {"chunk_id": "b", "faiss_id": 1, "score": 0.2},  # below threshold
    ]
    service = _make_semantic_service(hits, threshold=0.5, chunks_text_map={"a": "content a", "b": "content b"})

    results = service.semantic_search("query", top_k=10)
    assert [r["chunk_id"] for r in results] == ["a"]


def test_semantic_search_resorts_by_rerank_score():
    hits = [
        {"chunk_id": "a", "faiss_id": 0, "score": 0.9},
        {"chunk_id": "b", "faiss_id": 1, "score": 0.8},
    ]
    # FAISS ranks "a" first, but reranker flips the order.
    service = _make_semantic_service(
        hits, reranker_scores=[0.1, 0.9], chunks_text_map={"a": "content a", "b": "content b"}
    )

    results = service.semantic_search("query", top_k=10)
    assert [r["chunk_id"] for r in results] == ["b", "a"]
    assert results[0]["score"] == 0.9


def test_semantic_search_returns_empty_when_embedding_fails():
    hits = [{"chunk_id": "a", "faiss_id": 0, "score": 0.9}]
    service = SemanticSearchService(
        embedder=FakeEmbeddingProvider(vector=[]),
        vector_store=FakeVectorStore(hits),
        reranker=FakeReranker(),
        chunk_map={},
        chunks_text_map={},
    )
    assert service.semantic_search("query") == []


def test_semantic_search_truncates_to_top_k():
    hits = [{"chunk_id": str(i), "faiss_id": i, "score": 1.0} for i in range(5)]
    chunks_text_map = {str(i): f"content {i}" for i in range(5)}
    service = _make_semantic_service(hits, chunks_text_map=chunks_text_map)

    results = service.semantic_search("query", top_k=2)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# DocRefSearchService — citation parsing helpers
# ---------------------------------------------------------------------------

def _make_doc_ref_service(chunk_map=None, article_index_map=None, hits=None, threshold=0.5):
    vector_store = FakeVectorStore(hits or [])
    return DocRefSearchService(
        embedder=FakeEmbeddingProvider(),
        vector_store=vector_store,
        chunk_map=chunk_map or {},
        article_index_map=article_index_map or {},
        chunks_text_map={},
        threshold=threshold,
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        # NOTE: the extraction regex's final segment only matches uppercase
        # letters ([A-ZĐƯƠ]+), so a trailing lowercase letter (e.g. the "g" in
        # "TTg") gets dropped. This is a pre-existing quirk carried over
        # unchanged from the original implementation (out of scope for this
        # refactor's approved bugfixes) — asserted here as documented,
        # preserved behavior rather than silently "fixed" by the test.
        ("36/2015/QĐ-TTg", "36/2015/QĐ-TT"),
        ("Theo Quyết định 36 / 2015 / QĐ-TTg về...", "36/2015/QĐ-TT"),
        ("không có số hiệu ở đây", None),
        ("12/2020/NĐ-CP", "12/2020/NĐ-CP"),
    ],
)
def test_extract_doc_num(raw, expected):
    service = _make_doc_ref_service()
    assert service._extract_doc_num(raw) == expected


def test_normalize_doc_ref_collapses_whitespace_and_lowercases():
    service = _make_doc_ref_service()
    assert service._normalize_doc_ref("  Luật   Đất   Đai  ") == "luật đất đai"


def test_normalize_doc_ref_handles_empty():
    service = _make_doc_ref_service()
    assert service._normalize_doc_ref("") == ""


# ---------------------------------------------------------------------------
# DocRefSearchService — doc_ref_search end-to-end against fixture chunk_map
# ---------------------------------------------------------------------------

def test_doc_ref_search_exact_doc_num_match():
    chunk_map = {
        "c1": {"title": "Luật Đất đai", "doc_num": "45/2013/QH-TTg", "doc_id": "doc1", "article": "Điều 10"},
        "c2": {"title": "Luật khác", "doc_num": "1/2020/ND-CP", "doc_id": "doc2"},
    }
    service = _make_doc_ref_service(chunk_map=chunk_map)

    results = service.doc_ref_search(query="q", doc_ref="45/2013/QH-TTg")
    assert [r["chunk_id"] for r in results] == ["c1"]
    assert results[0]["source"] == "doc_ref_search"


def test_doc_ref_search_fuzzy_fallback_on_title():
    chunk_map = {
        "c1": {"title": "Luật Đất đai 2013", "doc_num": "", "doc_id": "doc1"},
    }
    service = _make_doc_ref_service(chunk_map=chunk_map)

    results = service.doc_ref_search(query="q", doc_ref="Luật Đất đai")
    assert [r["chunk_id"] for r in results] == ["c1"]


def test_doc_ref_search_no_match_returns_empty():
    chunk_map = {"c1": {"title": "Luật Đất đai", "doc_num": "45/2013/QH13", "doc_id": "doc1"}}
    service = _make_doc_ref_service(chunk_map=chunk_map)

    assert service.doc_ref_search(query="q", doc_ref="99/9999/XX-YY") == []


def test_doc_ref_search_article_filter_via_article_index_map():
    chunk_map = {
        "c1": {"title": "T", "doc_num": "1/2020/ND-CP", "doc_id": "doc1", "article": "Điều 1"},
        "c2": {"title": "T", "doc_num": "1/2020/ND-CP", "doc_id": "doc1", "article": "Điều 2"},
    }
    article_index_map = {"doc1|Điều 2": [2]}
    hits = [{"chunk_id": "c2", "faiss_id": 2, "score": 1.0}]
    service = _make_doc_ref_service(chunk_map=chunk_map, article_index_map=article_index_map, hits=hits)

    results = service.doc_ref_search(query="q", doc_ref="1/2020/ND-CP", article_filter="Điều 2")
    assert [r["chunk_id"] for r in results] == ["c2"]


def test_doc_ref_search_clause_filter_scans_metadata():
    chunk_map = {
        "c1": {"title": "T", "doc_num": "1/2020/ND-CP", "doc_id": "doc1", "clause": "Khoản 1"},
        "c2": {"title": "T", "doc_num": "1/2020/ND-CP", "doc_id": "doc1", "clause": "Khoản 2"},
    }
    service = _make_doc_ref_service(chunk_map=chunk_map)

    results = service.doc_ref_search(query="q", doc_ref="1/2020/ND-CP", clause_filter="Khoản 2")
    assert [r["chunk_id"] for r in results] == ["c2"]
