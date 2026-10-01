"""Unit tests for ``RAGService`` with mocked dependencies."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.rag_service import (
    EXCERPT_MAX_LENGTH,
    NOT_FOUND_ANSWER,
    RAGService,
    get_rag_service,
)


def _chunk(
    *,
    chunk_id: str,
    filename: str = "doc.txt",
    chunk_index: int = 0,
    content: str = "ornek icerik",
    similarity_score: float = 0.9,
) -> dict[str, object]:
    return {
        "rank": 1,
        "chunk_id": chunk_id,
        "filename": filename,
        "chunk_index": chunk_index,
        "content": content,
        "character_count": len(content),
        "distance": 0.1,
        "similarity_score": similarity_score,
    }


def _build_service(
    monkeypatch: pytest.MonkeyPatch,
    retrieved: list[dict[str, object]],
) -> tuple[RAGService, AsyncMock, MagicMock]:
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = retrieved

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_rag_prompt.return_value = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]

    mock_llm = MagicMock()
    mock_llm.generate_answer = AsyncMock(return_value="Mocked RAG answer [Kaynak 1]")

    monkeypatch.setattr(
        "app.services.rag_service.get_retriever_service",
        lambda: mock_retriever,
    )
    monkeypatch.setattr(
        "app.services.rag_service.get_prompt_builder_service",
        lambda: mock_prompt_builder,
    )
    monkeypatch.setattr(
        "app.services.rag_service.get_llm_service",
        lambda: mock_llm,
    )

    service = get_rag_service()
    return service, mock_llm, mock_prompt_builder


def test_rag_service_calls_llm_when_results_pass_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retriever hits above threshold should trigger an LLM call."""
    service, mock_llm, mock_prompt = _build_service(
        monkeypatch,
        [_chunk(chunk_id="c1", similarity_score=0.9)],
    )

    result = asyncio.run(service.answer_question("GRU modeli nedir?"))

    mock_prompt.build_rag_prompt.assert_called_once()
    mock_llm.generate_answer.assert_awaited_once()
    assert result["answer"] == "Mocked RAG answer [Kaynak 1]"
    assert result["retrieved_count"] == 1
    assert len(result["sources"]) == 1


def test_rag_service_skips_llm_when_no_results_pass_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-similarity results should not call the LLM."""
    service, mock_llm, mock_prompt = _build_service(
        monkeypatch,
        [_chunk(chunk_id="c1", similarity_score=0.1)],
    )

    result = asyncio.run(service.answer_question("bilinmeyen konu"))

    mock_prompt.build_rag_prompt.assert_not_called()
    mock_llm.generate_answer.assert_not_awaited()
    assert result["answer"] == NOT_FOUND_ANSWER
    assert result["sources"] == []
    assert result["retrieved_count"] == 0


def test_source_numbers_start_from_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source numbers should be sequential starting at 1."""
    service, _, _ = _build_service(
        monkeypatch,
        [
            _chunk(chunk_id="c1", chunk_index=2, similarity_score=0.9),
            _chunk(chunk_id="c2", chunk_index=5, similarity_score=0.8),
        ],
    )

    result = asyncio.run(service.answer_question("kaynak sirasi"))
    sources = result["sources"]
    assert isinstance(sources, list)
    assert [source["source_number"] for source in sources] == [1, 2]


def test_excerpt_is_limited_to_220_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Excerpts must not exceed the configured maximum length."""
    long_content = ("A" * 500) + "\n\n" + ("B" * 50)
    service, _, _ = _build_service(
        monkeypatch,
        [_chunk(chunk_id="c1", content=long_content, similarity_score=0.95)],
    )

    result = asyncio.run(service.answer_question("uzun excerpt"))
    sources = result["sources"]
    assert isinstance(sources, list)
    excerpt = str(sources[0]["excerpt"])
    content = str(sources[0]["content"])
    assert len(excerpt) <= EXCERPT_MAX_LENGTH
    assert content == long_content
    assert len(content) > len(excerpt)
    assert excerpt.endswith("...")


def test_source_content_is_full_passage_and_excerpt_is_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sources must expose full content while excerpt stays a short preview."""
    long_content = "Word " * 120
    service, _, _ = _build_service(
        monkeypatch,
        [_chunk(chunk_id="c1", content=long_content, similarity_score=0.91)],
    )

    result = asyncio.run(service.answer_question("full content check"))
    sources = result["sources"]
    assert isinstance(sources, list)
    source = sources[0]
    assert source["content"] == long_content
    assert isinstance(source["excerpt"], str)
    assert len(str(source["excerpt"])) <= EXCERPT_MAX_LENGTH
    assert len(str(source["content"])) > len(str(source["excerpt"]))
    # Truncating excerpt must never mutate the stored full content.
    assert str(source["content"]).endswith("Word ")


def test_duplicate_chunk_ids_are_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate chunk IDs should appear only once in sources."""
    service, _, _ = _build_service(
        monkeypatch,
        [
            _chunk(chunk_id="same", content="birinci", similarity_score=0.9),
            _chunk(chunk_id="same", content="ikinci", similarity_score=0.85),
            _chunk(chunk_id="other", content="ucuncu", similarity_score=0.8),
        ],
    )

    result = asyncio.run(service.answer_question("duplicate chunks"))
    sources = result["sources"]
    assert isinstance(sources, list)
    assert result["retrieved_count"] == 2
    assert [source["chunk_id"] for source in sources] == ["same", "other"]
