"""Retrieval services.

Split from the old services/Search.py "god object" into two single-purpose
classes that depend only on the EmbeddingProvider/VectorStore/Reranker
interfaces (core/interfaces.py) instead of concrete FAISS/ChatService calls:

- SemanticSearchService: global semantic search + rerank (used for the
  initial sub-query retrieval step).
- DocRefSearchService: Vietnamese legal-citation-filtered search (used by the
  search_referenced_document tool).

Both need the same on-disk chunk_map/chunks text, loaded once via
load_search_data() and passed into each constructor.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from core.config import DataSettings
from core.interfaces import EmbeddingProvider, Reranker, VectorStore


def load_search_data(data: DataSettings) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    """Loads chunk_map.json, article_index_map.json, and the embed_text of
    each chunk from chunks.json (used as a fast content lookup)."""
    with open(data.chunk_map_path, "r", encoding="utf-8") as f:
        chunk_map = json.load(f)

    with open(data.article_map_path, "r", encoding="utf-8") as f:
        article_index_map = json.load(f)

    chunks_text_map: Dict[str, str] = {}
    if os.path.exists(data.chunks_path):
        with open(data.chunks_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        for c in all_chunks:
            chunks_text_map[c["chunk_id"]] = c.get("embed_text", "")

    return chunk_map, article_index_map, chunks_text_map


def _get_chunk_content(chunk_id: str, chunk_map: Dict[str, Any], chunks_text_map: Dict[str, str]) -> str:
    """Lấy nội dung full của chunk."""
    if chunk_id in chunks_text_map:
        return chunks_text_map[chunk_id]

    meta = chunk_map.get(chunk_id, {})
    parts = []
    if meta.get("title"):
        parts.append(meta["title"])
    if meta.get("article"):
        parts.append(meta["article"])
    if meta.get("content"):
        parts.append(meta["content"])
    return " | ".join(parts)


class SemanticSearchService:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        reranker: Reranker,
        chunk_map: Dict[str, Any],
        chunks_text_map: Dict[str, str],
        threshold: float = 0.5,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._chunk_map = chunk_map
        self._chunks_text_map = chunks_text_map
        self._threshold = threshold

    def semantic_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Tìm kiếm ngữ nghĩa toàn cục:
        1. Embed query.
        2. Vector search.
        3. Rerank kết quả.
        """
        vec = self._embedder.embed(query)
        if not vec:
            return []

        hits = self._vector_store.search(vec, top_k * 2)

        candidates = []
        docs_text_for_rerank = []

        for hit in hits:
            if hit["score"] < self._threshold:
                continue

            chunk_id = hit["chunk_id"]
            meta = self._chunk_map.get(chunk_id, {})
            content = _get_chunk_content(chunk_id, self._chunk_map, self._chunks_text_map)

            candidates.append({
                "chunk_id": chunk_id,
                "faiss_score": hit["score"],
                "metadata": meta,
                "content": content,
            })
            docs_text_for_rerank.append(content)

        if not candidates:
            return []

        rerank_scores = self._reranker.rerank(query, docs_text_for_rerank)

        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = rerank_scores[i]
            candidate["final_score"] = rerank_scores[i]

        candidates.sort(key=lambda x: x["final_score"], reverse=True)

        return [
            {
                "chunk_id": item["chunk_id"],
                "score": item["final_score"],
                "metadata": item["metadata"],
                "content": item["content"],
            }
            for item in candidates[:top_k]
        ]


class DocRefSearchService:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        chunk_map: Dict[str, Any],
        article_index_map: Dict[str, Any],
        chunks_text_map: Dict[str, str],
        threshold: float = 0.5,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunk_map = chunk_map
        self._article_index_map = article_index_map
        self._chunks_text_map = chunks_text_map
        self._threshold = threshold

    def doc_ref_search(
        self,
        query: str,
        doc_ref: str,
        article_filter: Optional[str] = None,
        clause_filter: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm trong văn bản cụ thể (dùng cho tool).
        Hỗ trợ lọc theo Số hiệu, Điều, Khoản trước khi Semantic Rank.
        """
        extracted_doc_num = self._extract_doc_num(doc_ref)
        ref_norm = self._normalize_doc_ref(extracted_doc_num or doc_ref)

        matched_ids: List[str] = []

        if extracted_doc_num:
            extracted_norm = self._normalize_doc_ref(extracted_doc_num)
            for chunk_id, meta in self._chunk_map.items():
                doc_num_norm = self._normalize_doc_ref(meta.get("doc_num", ""))
                if doc_num_norm == extracted_norm:
                    matched_ids.append(chunk_id)

        if not matched_ids:
            for chunk_id, meta in self._chunk_map.items():
                doc_num_norm = self._normalize_doc_ref(meta.get("doc_num", ""))
                title_norm = self._normalize_doc_ref(meta.get("title", ""))
                if (
                    ref_norm in doc_num_norm
                    or ref_norm in title_norm
                    or doc_num_norm in ref_norm
                    or title_norm in ref_norm
                ):
                    matched_ids.append(chunk_id)

        if not matched_ids:
            return []

        if article_filter:
            dieu_norm = self._normalize_doc_ref(article_filter)
            doc_id = self._chunk_map[matched_ids[0]].get("doc_id")
            article_key = f"{doc_id}|{article_filter.strip()}"

            faiss_ids_for_article = self._article_index_map.get(article_key)
            if faiss_ids_for_article:
                ids_from_article = {
                    self._vector_store.chunk_id_for(fid) for fid in faiss_ids_for_article
                }
                ids_from_article.discard(None)
                matched_ids = [cid for cid in matched_ids if cid in ids_from_article]
            else:
                matched_ids = [
                    cid for cid in matched_ids
                    if dieu_norm in self._normalize_doc_ref(self._chunk_map[cid].get("article", ""))
                ]

        if clause_filter:
            khoan_norm = self._normalize_doc_ref(clause_filter)
            matched_ids = [
                cid for cid in matched_ids
                if khoan_norm in self._normalize_doc_ref(self._chunk_map[cid].get("clause", ""))
            ]

        if not matched_ids:
            return []

        if len(matched_ids) > 1:
            vec = self._embedder.embed(query)
            if vec:
                matched_faiss_ids = {
                    self._vector_store.faiss_id_for(cid) for cid in matched_ids
                }
                matched_faiss_ids.discard(None)

                if matched_faiss_ids:
                    scored_hits = self._vector_store.search_subset(
                        vec, matched_faiss_ids, top_k
                    )
                    matched_ids = [
                        h["chunk_id"] for h in scored_hits if h["score"] >= self._threshold
                    ]

        results = []
        for cid in matched_ids[:top_k]:
            meta = self._chunk_map.get(cid, {})
            results.append({
                "chunk_id": cid,
                "score": 1.0,  # Default score cao vì đã filter thủ công
                "metadata": meta,
                "content": _get_chunk_content(cid, self._chunk_map, self._chunks_text_map),
                "source": "doc_ref_search",
            })

        return results

    def _normalize_doc_ref(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _extract_doc_num(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(
            r"\d+[A-Za-z]*\s*/\s*\d{4}\s*/\s*[A-ZĐƯƠ]+(?:-[A-ZĐƯƠ]+)*",
            text,
            flags=re.UNICODE,
        )
        if not match:
            return None
        raw = match.group(0)
        return re.sub(r"\s*/\s*", "/", raw).strip()
