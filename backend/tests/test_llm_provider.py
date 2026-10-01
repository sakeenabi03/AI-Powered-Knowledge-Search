"""Unit tests for provider-aware LLMService routing (mocked, no downloads)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import get_settings
from app.services.llm_service import (
    LLMConfigurationError,
    LLMService,
    get_llm_service,
)
from app.services.rag_service import get_rag_service


MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is RAG?"},
]


def _reset_llm_singleton() -> None:
    LLMService._instance = None


@pytest.fixture(autouse=True)
def _clear_settings_and_llm(monkeypatch: pytest.MonkeyPatch):
    """Reset settings cache and LLM singleton around each test."""
    get_settings.cache_clear()
    _reset_llm_singleton()
    yield
    get_settings.cache_clear()
    _reset_llm_singleton()


def test_openrouter_provider_uses_openai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter provider should call the AsyncOpenAI chat completions API."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")
    get_settings.cache_clear()

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="OpenRouter answer")),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    service = get_llm_service()
    monkeypatch.setattr(service, "_get_client", lambda: mock_client)

    answer = asyncio.run(service.generate_answer(MESSAGES))

    assert answer == "OpenRouter answer"
    mock_client.chat.completions.create.assert_awaited_once()
    kwargs = mock_client.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "openai/gpt-4o-mini"
    assert kwargs["messages"] == MESSAGES


def test_foundry_local_provider_uses_foundry_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Foundry Local provider should delegate to FoundryLocalService."""
    monkeypatch.setenv("LLM_PROVIDER", "foundry_local")
    monkeypatch.setenv("FOUNDRY_MODEL_ALIAS", "qwen2.5-0.5b")
    get_settings.cache_clear()

    mock_foundry = MagicMock()
    mock_foundry.generate_answer = AsyncMock(return_value="Local Foundry answer")
    monkeypatch.setattr(
        "app.services.foundry_local_service.get_foundry_local_service",
        lambda: mock_foundry,
    )

    service = get_llm_service()
    answer = asyncio.run(service.generate_answer(MESSAGES))

    assert answer == "Local Foundry answer"
    mock_foundry.generate_answer.assert_awaited_once_with(MESSAGES)


def test_unsupported_provider_raises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported providers should fail with a clear configuration error."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    get_settings.cache_clear()

    service = get_llm_service()
    monkeypatch.setattr(
        "app.services.llm_service.get_settings",
        lambda: MagicMock(LLM_PROVIDER="unknown_cloud"),
    )

    with pytest.raises(LLMConfigurationError, match="Unsupported LLM_PROVIDER"):
        asyncio.run(service.generate_answer(MESSAGES))


def test_rag_service_is_provider_agnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAGService should only call LLMService.generate_answer()."""
    monkeypatch.setenv("LLM_PROVIDER", "foundry_local")
    monkeypatch.setenv("FOUNDRY_MODEL_ALIAS", "qwen2.5-0.5b")
    monkeypatch.setenv("MIN_SIMILARITY_SCORE", "0.1")
    get_settings.cache_clear()

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "rank": 1,
            "chunk_id": "c1",
            "filename": "doc.txt",
            "chunk_index": 0,
            "content": "RAG combines retrieval with generation.",
            "character_count": 40,
            "distance": 0.1,
            "similarity_score": 0.9,
        }
    ]
    mock_prompt = MagicMock()
    mock_prompt.build_rag_prompt.return_value = MESSAGES
    mock_llm = MagicMock()
    mock_llm.generate_answer = AsyncMock(return_value="Grounded answer")

    monkeypatch.setattr(
        "app.services.rag_service.get_retriever_service",
        lambda: mock_retriever,
    )
    monkeypatch.setattr(
        "app.services.rag_service.get_prompt_builder_service",
        lambda: mock_prompt,
    )
    monkeypatch.setattr(
        "app.services.rag_service.get_llm_service",
        lambda: mock_llm,
    )

    result = asyncio.run(get_rag_service().answer_question("What is RAG?"))

    mock_llm.generate_answer.assert_awaited_once_with(MESSAGES)
    assert result["answer"] == "Grounded answer"
    assert result["retrieved_count"] == 1


def test_ask_endpoint_model_field_for_foundry_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ask endpoint should return FOUNDRY_MODEL_ALIAS when provider is local."""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("LLM_PROVIDER", "foundry_local")
    monkeypatch.setenv("FOUNDRY_MODEL_ALIAS", "qwen2.5-0.5b")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")
    get_settings.cache_clear()

    mock_rag = MagicMock()
    mock_rag.answer_question = AsyncMock(
        return_value={
            "answer": "Local answer",
            "retrieved_count": 0,
            "sources": [],
        }
    )
    monkeypatch.setattr("app.api.chat.get_rag_service", lambda: mock_rag)

    client = TestClient(app)
    response = client.post("/api/chat/ask", json={"query": "hello world"})

    assert response.status_code == 200
    assert response.json()["model"] == "qwen2.5-0.5b"


def test_ask_endpoint_model_field_for_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ask endpoint should return LLM_MODEL when provider is OpenRouter."""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("FOUNDRY_MODEL_ALIAS", "qwen2.5-0.5b")
    get_settings.cache_clear()

    mock_rag = MagicMock()
    mock_rag.answer_question = AsyncMock(
        return_value={
            "answer": "Cloud answer",
            "retrieved_count": 0,
            "sources": [],
        }
    )
    monkeypatch.setattr("app.api.chat.get_rag_service", lambda: mock_rag)

    client = TestClient(app)
    response = client.post("/api/chat/ask", json={"query": "hello world"})

    assert response.status_code == 200
    assert response.json()["model"] == "openai/gpt-4o-mini"


def test_foundry_service_skips_download_when_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ensure_model should not download when the model is already cached."""
    from app.services.foundry_local_service import (
        FoundryLocalService,
        get_foundry_local_service,
    )

    FoundryLocalService._instance = None
    monkeypatch.setenv("LLM_PROVIDER", "foundry_local")
    monkeypatch.setenv("FOUNDRY_MODEL_ALIAS", "qwen2.5-0.5b")
    get_settings.cache_clear()

    mock_model = MagicMock()
    mock_model.is_cached = True
    mock_model.is_loaded = False
    mock_model.download = MagicMock()
    mock_model.load = MagicMock()

    mock_catalog = MagicMock()
    mock_catalog.get_model.return_value = mock_model
    mock_manager = MagicMock()
    mock_manager.catalog = mock_catalog

    service = get_foundry_local_service()
    service._manager = mock_manager

    resolved = service.ensure_model()
    service.load_model()

    assert resolved is mock_model
    mock_model.download.assert_not_called()
    mock_model.load.assert_called_once()

    FoundryLocalService._instance = None
