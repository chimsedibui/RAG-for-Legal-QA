"""Reranker implementations.

VLLMReranker talks to a vLLM-hosted cross-encoder reranker (e.g.
Qwen3-Reranker) over its custom (non-OpenAI-SDK) `/rerank` REST endpoint.
NullReranker is the identity/no-op strategy used when no reranker is
configured, replacing the previous `if self.rerank_client is None` branch
that used to live inside ChatService.
"""
from typing import List, Optional

import requests
from pydantic import BaseModel


class _Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class _Document(BaseModel):
    text: str
    multi_modal: Optional[dict] = None


class _RerankerResult(BaseModel):
    index: int
    document: _Document
    relevance_score: float


class _RerankerResponse(BaseModel):
    id: str
    model: str
    usage: _Usage
    results: List[_RerankerResult]


class VLLMReranker:
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self._base_url = base_url if base_url.endswith("/") else base_url + "/"
        self._api_key = api_key
        self._model_name = model_name

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []

        try:
            response = requests.post(
                f"{self._base_url}rerank",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model_name, "query": query, "documents": documents},
            )
            response.raise_for_status()
            parsed = _RerankerResponse.model_validate(response.json())

            score_map = {res.index: res.relevance_score for res in parsed.results}
            return [score_map.get(i, 0.0) for i in range(len(documents))]
        except Exception as e:
            print(f"Lỗi rerank: {e}")
            return [1.0] * len(documents)


class NullReranker:
    """No-op reranker: keeps documents in their original order."""

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        return [1.0] * len(documents)
