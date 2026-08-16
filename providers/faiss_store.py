"""VectorStore implementation backed by a FAISS IndexIDMap (inner product).

Owns the FAISS index file, the faiss_id <-> chunk_id translation map, and the
"search broadly, then filter to a candidate set" strategy used by doc-ref
filtered search — logic that used to be duplicated inline across
semantic_search and doc_ref_search in the old services/Search.py.
"""
import json
import os
from typing import List, Optional, Set

import faiss
import numpy as np

from core.interfaces import SearchHit


class FaissVectorStore:
    def __init__(self, index_path: str, id_map_path: str):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Không tìm thấy file index: {index_path}")

        self._index = faiss.read_index(index_path)

        with open(id_map_path, "r") as f:
            self._faiss_id_to_chunk_id = {int(k): v for k, v in json.load(f).items()}
        self._chunk_id_to_faiss_id = {v: k for k, v in self._faiss_id_to_chunk_id.items()}

    @property
    def total(self) -> int:
        return self._index.ntotal

    def chunk_id_for(self, faiss_id: int) -> Optional[str]:
        return self._faiss_id_to_chunk_id.get(faiss_id)

    def faiss_id_for(self, chunk_id: str) -> Optional[int]:
        return self._chunk_id_to_faiss_id.get(chunk_id)

    def search(self, vector: List[float], top_k: int) -> List[SearchHit]:
        vec_np = np.array([vector], dtype=np.float32)
        k = min(top_k, self._index.ntotal)
        if k <= 0:
            return []
        scores, ids = self._index.search(vec_np, k)

        hits: List[SearchHit] = []
        for score, faiss_idx in zip(scores[0], ids[0]):
            if faiss_idx < 0:
                continue
            chunk_id = self._faiss_id_to_chunk_id.get(int(faiss_idx))
            if chunk_id is None:
                continue
            hits.append({"chunk_id": chunk_id, "score": float(score)})
        return hits

    def search_subset(
        self, vector: List[float], candidate_faiss_ids: Set[int], top_k: int
    ) -> List[SearchHit]:
        if not candidate_faiss_ids:
            return []

        vec_np = np.array([vector], dtype=np.float32)
        k_search = min(len(candidate_faiss_ids) + 5, self._index.ntotal)
        if k_search <= 0:
            return []
        scores, ids = self._index.search(vec_np, k_search)

        hits: List[SearchHit] = []
        for score, faiss_idx in zip(scores[0], ids[0]):
            if int(faiss_idx) in candidate_faiss_ids:
                chunk_id = self._faiss_id_to_chunk_id.get(int(faiss_idx))
                if chunk_id is None:
                    continue
                hits.append({"chunk_id": chunk_id, "score": float(score)})

        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:top_k]
