"""Document indexing pipeline: extract, chunk, embed, and store."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Final

from app.core.config import get_settings
from app.services.document_loader import extract_text
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.text_splitter import split_text
from app.services.vector_store import get_vector_store_service

logger = logging.getLogger(__name__)


class DocumentIndexerService:
    """Orchestrate document extraction, chunking, embedding, and indexing.

    Uses shared singleton embedding and vector-store services so heavy models
    and store clients are not recreated per request. The active vector store is
    selected by ``VECTOR_STORE_PROVIDER`` (chromadb or sqlite).
    """

    _instance: DocumentIndexerService | None = None
    _lock: Final[threading.RLock] = threading.RLock()

    def __new__(cls) -> DocumentIndexerService:
        """Return the shared ``DocumentIndexerService`` instance.

        Returns:
            DocumentIndexerService: Process-wide singleton instance.
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

    def count_indexed_chunks(self, filename: str) -> int:
        """Count vectors stored for a given source filename.

        Args:
            filename: Stored source file name.

        Returns:
            int: Number of indexed chunk vectors for the file.

        Raises:
            ValueError: If ``filename`` is empty.
            RuntimeError: If the count query fails.
        """
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty.")

        safe_filename = filename.strip()
        try:
            return int(self._vector_store.count_by_filename(safe_filename))
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to count indexed chunks for '%s'.",
                safe_filename,
            )
            raise RuntimeError(
                f"Failed to count indexed chunks for '{safe_filename}'."
            ) from exc

    def delete_document_index(self, filename: str) -> int:
        """Delete all vector records associated with a document.

        Args:
            filename: Stored source file name.

        Returns:
            int: Number of deleted vector records.

        Raises:
            ValueError: If ``filename`` is empty.
            RuntimeError: If deletion fails.
        """
        deleted = self._vector_store.delete_document(filename)
        logger.info(
            "Deleted %s indexed vectors for document '%s'.",
            deleted,
            filename,
        )
        return deleted

    def index_document(self, file_path: Path) -> dict[str, int]:
        """Extract, chunk, embed, and index a document into the vector store.

        Args:
            file_path: Absolute or relative path to the stored document.

        Returns:
            dict[str, int]: Indexing stats with ``character_count``,
            ``total_chunks``, and ``vector_count``.

        Raises:
            ValueError: If extraction or chunking fails validation.
            RuntimeError: If embedding or vector storage fails.
            OSError: If the file cannot be read.
            UnicodeDecodeError: If a text file is not valid UTF-8.
        """
        filename = file_path.name
        logger.info("Document processing started for '%s'.", filename)
        indexed = False

        try:
            text = extract_text(file_path)
            character_count = len(text)
            logger.info(
                "Text extracted for '%s' (%s characters).",
                filename,
                character_count,
            )

            settings = get_settings()
            chunks = split_text(
                text=text,
                source_filename=filename,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            total_chunks = len(chunks)
            logger.info(
                "Chunks created for '%s' (total_chunks=%s).",
                filename,
                total_chunks,
            )

            documents = [str(chunk["content"]) for chunk in chunks]
            embeddings = self._embedding_service.embed_documents(documents)
            logger.info("Embedding completed for '%s'.", filename)

            metadatas: list[dict[str, Any]] = [
                {
                    "filename": filename,
                    "chunk_index": int(chunk["chunk_index"]),
                    "chunk_id": str(chunk["chunk_id"]),
                    "character_count": int(chunk["character_count"]),
                }
                for chunk in chunks
            ]
            ids = [str(chunk["chunk_id"]) for chunk in chunks]

            vector_count = self._vector_store.add_documents(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            indexed = True
            logger.info(
                "Vector indexing completed for '%s' (vector_count=%s).",
                filename,
                vector_count,
            )

            return {
                "character_count": character_count,
                "total_chunks": total_chunks,
                "vector_count": vector_count,
            }
        except Exception:
            if indexed:
                try:
                    self.delete_document_index(filename)
                    logger.warning(
                        "Rollback applied for '%s' after indexing failure.",
                        filename,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Rollback failed while cleaning vectors for '%s'.",
                        filename,
                    )
            else:
                # Partial writes may still exist if add_documents failed mid-way.
                try:
                    deleted = self.delete_document_index(filename)
                    if deleted:
                        logger.warning(
                            "Rollback applied for '%s' (%s partial vectors).",
                            filename,
                            deleted,
                        )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Rollback failed while cleaning vectors for '%s'.",
                        filename,
                    )
            raise

    def reindex_document(self, file_path: Path) -> dict[str, int]:
        """Rebuild the vector index for an existing document.

        Args:
            file_path: Path to the already stored document file.

        Returns:
            dict[str, int]: Fresh indexing stats.

        Raises:
            ValueError: If extraction or chunking fails validation.
            RuntimeError: If embedding or vector storage fails.
        """
        filename = file_path.name
        logger.info("Reindex started for '%s'.", filename)
        self.delete_document_index(filename)
        result = self.index_document(file_path)
        logger.info("Reindex completed for '%s'.", filename)
        return result


def get_document_indexer() -> DocumentIndexerService:
    """Return the shared ``DocumentIndexerService`` singleton.

    Returns:
        DocumentIndexerService: Process-wide indexer instance.
    """
    return DocumentIndexerService()
