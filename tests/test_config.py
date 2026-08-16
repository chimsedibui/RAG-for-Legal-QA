import pytest
from pydantic import ValidationError

from core.config import Settings


def test_missing_required_chat_var_fails_fast(monkeypatch, required_env):
    monkeypatch.delenv("CHAT_BASE_URL", raising=False)
    with pytest.raises(ValidationError, match="CHAT_BASE_URL"):
        Settings()


def test_missing_required_embedding_var_fails_fast(monkeypatch, required_env):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="EMBEDDING_API_KEY"):
        Settings()


def test_reranker_disabled_when_unset(required_env):
    settings = Settings()
    assert settings.reranker_enabled is False


def test_reranker_enabled_when_rerank_vars_set(monkeypatch, required_env):
    monkeypatch.setenv("RERANK_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("RERANK_API_KEY", "key")
    monkeypatch.setenv("RERANK_MODEL_NAME", "reranker-model")
    settings = Settings()
    assert settings.reranker_enabled is True


def test_reranker_stays_disabled_with_legacy_env_var_name(monkeypatch, required_env):
    """Regression guard for the RERANK_*/RERANKER_* mismatch bug: the old,
    incorrect RERANKER_* names must NOT silently activate the reranker."""
    monkeypatch.setenv("RERANKER_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("RERANKER_API_KEY", "key")
    monkeypatch.setenv("RERANKER_MODEL_NAME", "reranker-model")
    settings = Settings()
    assert settings.reranker_enabled is False


def test_retrieval_defaults_match_previous_hardcoded_values(required_env):
    settings = Settings()
    assert settings.retrieval.threshold == 0.5
    assert settings.retrieval.semantic_top_k == 10
    assert settings.retrieval.tool_search_top_k == 5
    assert settings.retrieval.max_context_chunks == 50
    assert settings.retrieval.max_tool_iterations == 3


def test_retrieval_settings_overridable_via_env(monkeypatch, required_env):
    monkeypatch.setenv("MAX_CONTEXT_CHUNKS", "20")
    monkeypatch.setenv("RETRIEVAL_THRESHOLD", "0.7")
    settings = Settings()
    assert settings.retrieval.max_context_chunks == 20
    assert settings.retrieval.threshold == 0.7


def test_chat_base_url_has_no_unsafe_default(monkeypatch, required_env):
    """Regression guard: previously CHAT_BASE_URL defaulted to an internal
    dev IP (10.9.3.241) if unset. It must now be required with no default."""
    monkeypatch.delenv("CHAT_BASE_URL", raising=False)
    with pytest.raises(ValidationError):
        Settings()
