"""API tests for ``POST /api/chat/ask`` with mocked RAG service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.services.llm_service import LLMConfigurationError


client = TestClient(app)


def test_ask_endpoint_returns_successful_response(monkeypatch) -> None:
    """Successful RAG answers should return ChatResponse fields."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "chromadb")
    get_settings.cache_clear()

    mock_rag = MagicMock()
    mock_rag.answer_question = AsyncMock(
        return_value={
            "answer": "GRU modeli uzun bagimliliklar icin kullanilir. [Kaynak 1]",
            "retrieved_count": 1,
            "sources": [
                {
                    "source_number": 1,
                    "filename": "plan.txt",
                    "chunk_index": 3,
                    "chunk_id": "chunk-1",
                    "similarity_score": 0.88,
                    "excerpt": "GRU modeli uzun bagimliliklar icin kullanilir.",
                    "content": (
                        "GRU modeli uzun bagimliliklar icin kullanilir. "
                        "Bu pasajin tamamı API yanitinda content alaninda doner."
                    ),
                }
            ],
        }
    )
    monkeypatch.setattr("app.api.chat.get_rag_service", lambda: mock_rag)

    response = client.post(
        "/api/chat/ask",
        json={"query": "  GRU modeli neden kullanilmistir?  ", "top_k": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "GRU modeli neden kullanilmistir?"
    assert body["answer"].startswith("GRU modeli")
    assert body["retrieved_count"] == 1
    assert body["model"] == "openai/gpt-4o-mini"
    assert body["sources"][0]["source_number"] == 1
    assert "content" in body["sources"][0]
    assert len(body["sources"][0]["content"]) >= len(body["sources"][0]["excerpt"])
    mock_rag.answer_question.assert_awaited_once()
    get_settings.cache_clear()


def test_ask_endpoint_returns_404_for_missing_filename(tmp_path, monkeypatch) -> None:
    """Missing filename filters should return HTTP 404."""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    get_settings.cache_clear()

    response = client.post(
        "/api/chat/ask",
        json={
            "query": "izin proseduru nedir?",
            "filename": "missing_document.txt",
        },
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
    get_settings.cache_clear()


def test_ask_endpoint_returns_503_when_api_key_missing(monkeypatch) -> None:
    """Missing LLM configuration should return HTTP 503."""
    mock_rag = MagicMock()
    mock_rag.answer_question = AsyncMock(
        side_effect=LLMConfigurationError("LLM_API_KEY is not configured.")
    )
    monkeypatch.setattr("app.api.chat.get_rag_service", lambda: mock_rag)

    response = client.post(
        "/api/chat/ask",
        json={"query": "GRU modeli neden kullanilmistir?"},
    )

    assert response.status_code == 503
    assert "LLM_API_KEY" in response.json()["detail"]
