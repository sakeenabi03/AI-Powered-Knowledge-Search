"""RAG orchestration: retrieve, filter, prompt, and answer."""

from __future__ import annotations

import logging
import threading
from typing import Final

from app.core.config import get_settings
from app.services.llm_service import LLMService, get_llm_service
from app.services.prompt_builder import PromptBuilderService, get_prompt_builder_service
from app.services.retriever import RetrieverService, get_retriever_service

logger = logging.getLogger(__name__)

NOT_FOUND_ANSWER = "Bu bilgi yüklenen dokümanlarda bulunamadı."
EXCERPT_MAX_LENGTH = 220


class RAGService:
    """Coordinate retrieval-augmented generation for document questions.

    Uses shared retriever and LLM singletons so heavy resources are reused.
    """

    _instance: RAGService | None = None
    _lock: Final[threading.RLock] = threading.RLock()

    def __new__(cls) -> RAGService:
        """Return the shared ``RAGService`` instance.

        Returns:
            RAGService: Process-wide singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._retriever = get_retriever_service()
                    instance._prompt_builder = get_prompt_builder_service()
                    instance._llm_service = get_llm_service()
                    instance._initialized = True
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize instance attributes only once for the singleton."""
        if getattr(self, "_initialized", False):
            return
        self._retriever: RetrieverService = get_retriever_service()
        self._prompt_builder: PromptBuilderService = get_prompt_builder_service()
        self._llm_service: LLMService = get_llm_service()
        self._initialized = True

    @staticmethod
    def _build_excerpt(content: str) -> str:
        """Create a short excerpt for API source citations.

        Args:
            content: Full chunk text.

        Returns:
            str: Whitespace-normalized excerpt of at most 220 characters.
            Ellipsis is appended only when the preview is truncated.
        """
        normalized = " ".join(content.split()).strip()
        if len(normalized) <= EXCERPT_MAX_LENGTH:
            return normalized
        return f"{normalized[: EXCERPT_MAX_LENGTH - 3].rstrip()}..."

    @staticmethod
    def _filter_contexts(
        retrieved: list[dict[str, object]],
        min_similarity_score: float,
    ) -> list[dict[str, object]]:
        """Filter and de-duplicate retrieved chunks by similarity and chunk_id.

        Args:
            retrieved: Raw retriever results in ranked order.
            min_similarity_score: Minimum accepted similarity score.

        Returns:
            list[dict[str, object]]: Filtered contexts preserving retriever order.
        """
        filtered: list[dict[str, object]] = []
        seen_chunk_ids: set[str] = set()

        for item in retrieved:
            score = float(item.get("similarity_score") or 0.0)
            if score < min_similarity_score:
                continue

            chunk_id = str(item.get("chunk_id") or "").strip()
            if chunk_id:
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)

            filtered.append(item)

        return filtered

    async def answer_question(
        self,
        query: str,
        top_k: int | None = None,
        filename: str | None = None,
    ) -> dict[str, object]:
        """Answer a question using retrieved document context and an LLM.

        Args:
            query: End-user question.
            top_k: Optional retrieval depth override.
            filename: Optional source document filter.

        Returns:
            dict[str, object]: Answer payload with ``answer``, ``sources``,
            and ``retrieved_count``.

        Raises:
            ValueError: If the query is invalid.
            RuntimeError: If retrieval or LLM generation fails.
        """
        cleaned_query = query.strip() if query else ""
        if not cleaned_query:
            raise ValueError("query cannot be empty.")

        settings = get_settings()
        resolved_top_k = top_k if top_k is not None else settings.RAG_TOP_K

        logger.info(
            "RAG request started (top_k=%s, filename=%s).",
            resolved_top_k,
            filename,
        )

        retrieved = self._retriever.retrieve(
            query=cleaned_query,
            top_k=resolved_top_k,
            filename=filename,
        )
        logger.info("Retrieval completed with %s raw results.", len(retrieved))

        filtered = self._filter_contexts(
            retrieved=retrieved,
            min_similarity_score=settings.MIN_SIMILARITY_SCORE,
        )
        logger.info(
            "Results remaining after similarity threshold: %s.",
            len(filtered),
        )

        if not filtered:
            return {
                "answer": NOT_FOUND_ANSWER,
                "sources": [],
                "retrieved_count": 0,
            }

        messages = self._prompt_builder.build_rag_prompt(
            query=cleaned_query,
            contexts=filtered,
        )
        logger.info("Prompt created.")

        logger.info("LLM call started.")
        answer = await self._llm_service.generate_answer(messages)
        logger.info("LLM call completed.")

        sources: list[dict[str, object]] = []
        for index, item in enumerate(filtered, start=1):
            content = str(item.get("content") or "")
            sources.append(
                {
                    "source_number": index,
                    "filename": str(item.get("filename") or ""),
                    "chunk_index": int(item.get("chunk_index") or 0),
                    "chunk_id": str(item.get("chunk_id") or ""),
                    "similarity_score": float(item.get("similarity_score") or 0.0),
                    "excerpt": self._build_excerpt(content),
                    "content": content,
                }
            )

        logger.info("RAG answer prepared with %s sources.", len(sources))
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_count": len(sources),
        }


def get_rag_service() -> RAGService:
    """Return the shared ``RAGService`` singleton.

    Returns:
        RAGService: Process-wide RAG service instance.
    """
    return RAGService()
