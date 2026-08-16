"""Protocols for the swappable components of the RAG pipeline.

These are structural (typing.Protocol) rather than ABCs on purpose: any class
that already implements the right method signatures satisfies the interface
without needing to inherit from anything here.
"""
from typing import Any, Generator, Optional, Protocol, TypedDict


class LLMProvider(Protocol):
    def chat(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
        response_format: Optional[dict] = None,
        stream: bool = False,
    ) -> Generator[Any, None, None]:
        """Streams SDK chunk objects if stream=True, else yields a single message object.

        On error: yields {"error": ...} if stream=True, raises otherwise.
        """
        ...


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]:
        """Returns an embedding vector, or [] on failure."""
        ...


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Returns one relevance score per document, same order as `documents`."""
        ...


class SearchHit(TypedDict):
    chunk_id: str
    score: float


class VectorStore(Protocol):
    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        """Global nearest-neighbor search."""
        ...

    def search_subset(
        self, vector: list[float], candidate_faiss_ids: set[int], top_k: int
    ) -> list[SearchHit]:
        """Nearest-neighbor search restricted to a candidate set of internal ids."""
        ...

    def chunk_id_for(self, faiss_id: int) -> Optional[str]: ...

    def faiss_id_for(self, chunk_id: str) -> Optional[int]: ...

    @property
    def total(self) -> int: ...


class Tool(Protocol):
    name: str
    schema: dict  # OpenAI function-calling "function" sub-dict (name/description/parameters)

    def execute(self, args: dict[str, Any], *, question: str) -> list[dict]:
        """Executes the tool, returning a list of doc dicts in the same shape
        SearchService results use ({chunk_id, score, metadata, content, ...})."""
        ...
