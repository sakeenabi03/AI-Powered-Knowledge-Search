"""ChromaDB persistent vector store service."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Final

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import get_settings

logger = logging.getLogger(__name__)

COLLECTION_NAME: Final[str] = "corporate_documents"
REQUIRED_METADATA_KEYS: Final[tuple[str, ...]] = (
    "filename",
    "chunk_index",
    "chunk_id",
    "character_count",
)


class VectorStoreService:
    """Manage document chunk embeddings in a persistent ChromaDB collection.

    The service uses ``chromadb.PersistentClient`` so vectors survive process
    restarts. Collection access is lazy and thread-safe.
    """

    _instance: VectorStoreService | None = None
    _lock: Final[threading.RLock] = threading.RLock()

    def __new__(cls) -> VectorStoreService:
        """Return the shared ``VectorStoreService`` instance.

        Returns:
            VectorStoreService: Process-wide singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._client = None
                    instance._collection = None
                    instance._persist_directory = Path(
                        get_settings().VECTOR_STORE_DIR
                    ).resolve()
                    instance._initialized = True
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize instance attributes only once for the singleton."""
        if getattr(self, "_initialized", False):
            return
        self._client: chromadb.ClientAPI | None = None
        self._collection: Collection | None = None
        self._persist_directory: Path = Path(
            get_settings().VECTOR_STORE_DIR
        ).resolve()
        self._initialized = True

    def _get_client(self) -> chromadb.ClientAPI:
        """Create or return the persistent ChromaDB client.

        Returns:
            chromadb.ClientAPI: Persistent ChromaDB client.

        Raises:
            RuntimeError: If the client cannot be initialized.
        """
        if self._client is not None:
            return self._client

        with self._lock:
            if self._client is not None:
                return self._client

            try:
                self._persist_directory.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Initializing ChromaDB PersistentClient at '%s'.",
                    self._persist_directory,
                )
                self._client = chromadb.PersistentClient(
                    path=str(self._persist_directory)
                )
                return self._client
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to initialize ChromaDB client.")
                raise RuntimeError(
                    "Failed to initialize ChromaDB PersistentClient."
                ) from exc

    def create_collection(self) -> Collection:
        """Create or retrieve the ``AI-Powered Knowledge Search_documents`` collection.

        Returns:
            Collection: Ready-to-use ChromaDB collection.

        Raises:
            RuntimeError: If the collection cannot be created or retrieved.
        """
        if self._collection is not None:
            return self._collection

        with self._lock:
            if self._collection is not None:
                return self._collection

            try:
                client = self._get_client()
                self._collection = client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    "ChromaDB collection '%s' is ready (count=%s).",
                    COLLECTION_NAME,
                    self._collection.count(),
                )
                return self._collection
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Failed to create or get collection '%s'.",
                    COLLECTION_NAME,
                )
                raise RuntimeError(
                    f"Failed to create collection '{COLLECTION_NAME}'."
                ) from exc

    def _validate_metadatas(self, metadatas: list[dict[str, Any]]) -> None:
        """Validate that each metadata payload contains required keys.

        Args:
            metadatas: Metadata dictionaries aligned with documents.

        Raises:
            ValueError: If metadata is missing required fields.
        """
        for index, metadata in enumerate(metadatas):
            missing = [
                key for key in REQUIRED_METADATA_KEYS if key not in metadata
            ]
            if missing:
                raise ValueError(
                    "Metadata at index "
                    f"{index} is missing required keys: {', '.join(missing)}."
                )

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> int:
        """Add document chunks and their embeddings to the vector store.

        Args:
            documents: Chunk text contents.
            embeddings: Embedding vectors matching ``documents``.
            metadatas: Metadata for each chunk. Must include ``filename``,
                ``chunk_index``, ``chunk_id``, and ``character_count``.
            ids: Optional Chroma IDs. Defaults to each metadata ``chunk_id``.

        Returns:
            int: Number of documents added.

        Raises:
            ValueError: If inputs are empty or length-mismatched.
            RuntimeError: If ChromaDB insertion fails.
        """
        if not documents:
            raise ValueError("Documents list cannot be empty.")

        if not (len(documents) == len(embeddings) == len(metadatas)):
            raise ValueError(
                "documents, embeddings, and metadatas must have the same length."
            )

        if ids is not None and len(ids) != len(documents):
            raise ValueError("ids length must match documents length.")

        self._validate_metadatas(metadatas)

        document_ids = ids or [str(item["chunk_id"]) for item in metadatas]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("Document ids must be unique.")

        try:
            collection = self.create_collection()
            collection.add(
                ids=document_ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(
                "Added %s documents to collection '%s'.",
                len(documents),
                COLLECTION_NAME,
            )
            return len(documents)
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to add documents to vector store.")
            raise RuntimeError("Failed to add documents to vector store.") from exc

    def delete_document(self, filename: str) -> int:
        """Delete all chunks belonging to a source document.

        Args:
            filename: Stored source file name used in metadata.

        Returns:
            int: Number of deleted chunk records.

        Raises:
            ValueError: If ``filename`` is empty.
            RuntimeError: If deletion fails.
        """
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty.")

        safe_filename = filename.strip()

        try:
            collection = self.create_collection()
            existing = collection.get(where={"filename": safe_filename})
            existing_ids = existing.get("ids") or []

            if not existing_ids:
                logger.info(
                    "No vector records found for filename '%s'.",
                    safe_filename,
                )
                return 0

            collection.delete(ids=existing_ids)
            deleted_count = len(existing_ids)
            logger.info(
                "Deleted %s vector records for filename '%s'.",
                deleted_count,
                safe_filename,
            )
            return deleted_count
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to delete vectors for filename '%s'.",
                safe_filename,
            )
            raise RuntimeError(
                f"Failed to delete vectors for filename '{safe_filename}'."
            ) from exc

    def similarity_search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a cosine similarity search against stored embeddings.

        Args:
            query_embedding: Query vector produced by ``EmbeddingService``.
            n_results: Maximum number of matches to return.
            where: Optional Chroma metadata filter.

        Returns:
            list[dict[str, Any]]: Ranked matches with ``id``, ``document``,
            ``metadata``, and ``distance``.

        Raises:
            ValueError: If the query embedding or ``n_results`` is invalid.
            RuntimeError: If the search operation fails.
        """
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty.")

        if n_results <= 0:
            raise ValueError("n_results must be greater than zero.")

        try:
            collection = self.create_collection()
            total = collection.count()
            if total == 0:
                logger.info("Similarity search skipped; collection is empty.")
                return []

            effective_n_results = min(n_results, total)
            if where is not None:
                matching = collection.get(where=where)
                matching_count = len(matching.get("ids") or [])
                if matching_count == 0:
                    logger.info(
                        "Similarity search skipped; no vectors match filter %s.",
                        where,
                    )
                    return []
                effective_n_results = min(n_results, matching_count)

            request: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": effective_n_results,
                "include": ["documents", "metadatas", "distances"],
            }
            if where is not None:
                request["where"] = where

            raw = collection.query(**request)

            ids = (raw.get("ids") or [[]])[0]
            documents = (raw.get("documents") or [[]])[0]
            metadatas = (raw.get("metadatas") or [[]])[0]
            distances = (raw.get("distances") or [[]])[0]

            results: list[dict[str, Any]] = []
            for index, item_id in enumerate(ids):
                results.append(
                    {
                        "id": item_id,
                        "document": documents[index] if index < len(documents) else None,
                        "metadata": metadatas[index] if index < len(metadatas) else None,
                        "distance": distances[index] if index < len(distances) else None,
                    }
                )

            logger.info("Similarity search returned %s results.", len(results))
            return results
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to run similarity search.")
            raise RuntimeError("Failed to run similarity search.") from exc

    def count(self) -> int:
        """Return the number of vectors stored in the collection.

        Returns:
            int: Total embedded chunk count.

        Raises:
            RuntimeError: If the count operation fails.
        """
        try:
            collection = self.create_collection()
            total = collection.count()
            logger.debug("Vector store count=%s.", total)
            return total
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to count vectors in collection.")
            raise RuntimeError("Failed to count vectors in collection.") from exc

    def count_by_filename(self, filename: str) -> int:
        """Count vectors stored for a source filename.

        Args:
            filename: Stored source file name.

        Returns:
            int: Number of chunk vectors for the file.

        Raises:
            ValueError: If ``filename`` is empty.
            RuntimeError: If the count query fails.
        """
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty.")

        safe_filename = filename.strip()
        try:
            collection = self.create_collection()
            existing = collection.get(where={"filename": safe_filename})
            return len(existing.get("ids") or [])
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to count vectors for filename '%s'.",
                safe_filename,
            )
            raise RuntimeError(
                f"Failed to count vectors for filename '{safe_filename}'."
            ) from exc

    @property
    def collection_name(self) -> str:
        """Display name used by document stats endpoints."""
        return COLLECTION_NAME


_factory_lock: Final[threading.RLock] = threading.RLock()
_active_provider: str | None = None
_active_vector_store: Any | None = None


def reset_vector_store_factory() -> None:
    """Reset the provider factory cache (used by tests)."""
    global _active_provider, _active_vector_store
    with _factory_lock:
        _active_provider = None
        _active_vector_store = None


def get_vector_store_service() -> Any:
    """Return the configured vector store provider singleton.

    Selects ChromaDB or SQLite based on ``VECTOR_STORE_PROVIDER``.

    Returns:
        Any: Provider service exposing indexer/retriever-compatible methods.

    Raises:
        RuntimeError: If the configured provider is unsupported.
    """
    global _active_provider, _active_vector_store

    settings = get_settings()
    provider = settings.VECTOR_STORE_PROVIDER

    with _factory_lock:
        if _active_vector_store is not None and _active_provider == provider:
            return _active_vector_store

        logger.info("Vector store provider selected: %s", provider)

        if provider == "chromadb":
            _active_vector_store = VectorStoreService()
        elif provider == "sqlite":
            from app.services.sqlite_vector_store import (
                get_sqlite_vector_store_service,
            )

            store = get_sqlite_vector_store_service()
            store.initialize()
            _active_vector_store = store
        else:
            raise RuntimeError(
                f"Unsupported VECTOR_STORE_PROVIDER '{provider}'. "
                "Valid values: chromadb, sqlite."
            )

        _active_provider = provider
        return _active_vector_store
