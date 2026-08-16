"""Centralized configuration. Single source of truth for every env var the
API and offline pipeline read — replaces the previously scattered os.getenv()
calls in services/Chat.py and pipeline/chunk_embedding.py.

Required chat/embedding vars have no default: a missing var fails fast with a
clear pydantic ValidationError at startup instead of silently falling back to
a hardcoded (and, previously, internal-network) default.
"""
import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    base_url: str = Field(alias="CHAT_BASE_URL")
    api_key: str = Field(alias="CHAT_API_KEY")
    model_name: str = Field(alias="CHAT_MODEL_NAME")


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    base_url: str = Field(alias="EMBEDDING_BASE_URL")
    api_key: str = Field(alias="EMBEDDING_API_KEY")
    model_name: str = Field(alias="EMBEDDING_MODEL_NAME")


class RerankSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    base_url: Optional[str] = Field(default=None, alias="RERANK_BASE_URL")
    api_key: Optional[str] = Field(default=None, alias="RERANK_API_KEY")
    model_name: Optional[str] = Field(default=None, alias="RERANK_MODEL_NAME")


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    threshold: float = Field(default=0.5, alias="RETRIEVAL_THRESHOLD")
    semantic_top_k: int = Field(default=10, alias="SEMANTIC_TOP_K")
    tool_search_top_k: int = Field(default=5, alias="TOOL_SEARCH_TOP_K")
    max_context_chunks: int = Field(default=50, alias="MAX_CONTEXT_CHUNKS")
    max_tool_iterations: int = Field(default=3, alias="MAX_TOOL_ITERATIONS")


class DataSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: str = Field(default="data", alias="DATA_DIR")

    @property
    def faiss_index_path(self) -> str:
        return os.path.join(self.data_dir, "faiss.index")

    @property
    def faiss_id_map_path(self) -> str:
        return os.path.join(self.data_dir, "faiss_id_map.json")

    @property
    def chunk_map_path(self) -> str:
        return os.path.join(self.data_dir, "chunk_map.json")

    @property
    def article_map_path(self) -> str:
        return os.path.join(self.data_dir, "article_index_map.json")

    @property
    def chunks_path(self) -> str:
        return os.path.join(self.data_dir, "chunks.json")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    port: int = Field(default=8000, alias="PORT")
    chat: ChatSettings = Field(default_factory=ChatSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    data: DataSettings = Field(default_factory=DataSettings)

    @property
    def reranker_enabled(self) -> bool:
        return bool(self.rerank.base_url and self.rerank.api_key and self.rerank.model_name)


@lru_cache
def get_settings() -> Settings:
    return Settings()
