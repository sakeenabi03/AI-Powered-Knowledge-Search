"""Semantic retrieval over indexed AI-Powered Knowledge Search document chunks."""

from __future__ import annotations

import logging
import threading
from typing import Any, Final

from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.vector_store import get_vector_store_service

logger = logging.getLogger(__name__)


class RetrieverService:
    """Retrieve the most relevant document chunks for a natural-language query.

    Uses shared singleton embedding and vector-store services so heavy
    resources are not recreated on every request. The active store is selected
    by ``VECTOR_STORE_PROVIDER``.
    """

    _instance: RetrieverService | None = None
    _lock: Final[threading.RLock] = threading.RLock()

    def __new__(cls) -> RetrieverService:
        """Return the shared ``RetrieverService`` instance.

        Returns:
            RetrieverService: Process-wide singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._embedding_service = get_embedding_service()
                    instance._vector_store = get_vector_store_service()
                    instance._initialized = True
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize instance attributes only once for the singleton."""
        if getattr(self, "_initialized", False):
            return
        self._embedding_service: EmbeddingService = get_embedding_service()
        self._vector_store: Any = get_vector_store_service()
        self._initialized = True

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filename: str | None = None,
    ) -> list[dict[str, object]]:
        """Embed a query and return the most similar indexed chunks.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of chunks to return (1-20).
            filename: Optional source file filter.

        Returns:
            list[dict[str, object]]: Ranked retrieval results in ChromaDB order.

        Raises:
            ValueError: If ``query`` or ``top_k`` is invalid.
            RuntimeError: If embedding or vector search fails.
        """
        cleaned_query = query.strip() if query else ""
        if not cleaned_query:
            raise ValueError("query cannot be empty.")

        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20.")

        where: dict[str, Any] | None = None
        if filename is not None and filename.strip():
            where = {"filename": filename.strip()}

        logger.info(
            "Retrieval started (top_k=%s, filename=%s).",
            top_k,
            where["filename"] if where else None,
        )

        try:
            query_embedding = self._embedding_service.embed_text(cleaned_query)
            logger.info("Query embedding generated.")

            if where is not None:
                logger.info(
                    "Filename filter applied: '%s'.",
                    where["filename"],
                )

            raw_results = self._vector_store.similarity_search(
                query_embedding=query_embedding,
                n_results=top_k,
                where=where,
            )
            logger.info("Vector similarity search completed.")
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected retrieval failure.")
            raise RuntimeError("Failed to retrieve documents for query.") from exc

        results: list[dict[str, object]] = []
        for index, item in enumerate(raw_results, start=1):
            metadata = item.get("metadata") or {}
            distance_raw = item.get("distance")
            distance = float(distance_raw) if distance_raw is not None else 0.0
            similarity_score = max(0.0, min(1.0, 1.0 - distance))

            chunk_id = str(metadata.get("chunk_id") or item.get("id") or "")
            source_filename = str(metadata.get("filename") or "")
            chunk_index = int(metadata.get("chunk_index") or 0)
            content = str(item.get("document") or "")
            character_count = int(
                metadata.get("character_count") or len(content)
            )

            results.append(
                {
                    "rank": index,
                    "chunk_id": chunk_id,
                    "filename": source_filename,
                    "chunk_index": chunk_index,
                    "content": content,
                    "character_count": character_count,
                    "distance": distance,
                    "similarity_score": similarity_score,
                }
            )

        logger.info("Retrieval returned %s results.", len(results))
        return results


def get_retriever_service() -> RetrieverService:
    """Return the shared ``RetrieverService`` singleton.

    Returns:
        RetrieverService: Process-wide retriever instance.
    """
    return RetrieverService()
