"""SQLite-backed local document/chunk knowledge store with cosine retrieval."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from app.core.config import get_settings
from app.services.vector_store import REQUIRED_METADATA_KEYS

logger = logging.getLogger(__name__)

SQLITE_COLLECTION_NAME: Final[str] = "sqlite_knowledge_base"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    Args:
        vector_a: First embedding.
        vector_b: Second embedding.

    Returns:
        float: Cosine similarity in roughly ``[-1, 1]``.
    """
    if len(vector_a) != len(vector_b) or not vector_a:
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for left, right in zip(vector_a, vector_b, strict=True):
        dot += left * right
        norm_a += left * left
        norm_b += right * right

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class SQLiteVectorStoreService:
    """Persist documents/chunks in SQLite and search with Python cosine similarity.

    Embeddings are stored as JSON text. This service also exposes the same
    public methods used by the indexer/retriever for ChromaDB compatibility.
    """

    _instance: SQLiteVectorStoreService | None = None
    _lock: Final[threading.RLock] = threading.RLock()

    def __new__(cls) -> SQLiteVectorStoreService:
        """Return the shared ``SQLiteVectorStoreService`` instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._db_path = Path(
                        get_settings().SQLITE_DB_PATH
                    ).resolve()
                    instance._connection: sqlite3.Connection | None = None
                    instance._schema_ready = False
                    instance._initialized = True
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize instance attributes only once for the singleton."""
        if getattr(self, "_initialized", False):
            return
        self._db_path: Path = Path(get_settings().SQLITE_DB_PATH).resolve()
        self._connection: sqlite3.Connection | None = None
        self._schema_ready = False
        self._initialized = True

    @property
    def collection_name(self) -> str:
        """Display name used by document stats endpoints."""
        return SQLITE_COLLECTION_NAME

    def initialize(self) -> None:
        """Create the database file/schema if needed and open a connection."""
        with self._lock:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            if self._connection is None:
                self._connection = sqlite3.connect(
                    str(self._db_path),
                    check_same_thread=False,
                )
                self._connection.row_factory = sqlite3.Row
                self._connection.execute("PRAGMA foreign_keys = ON")

            if not self._schema_ready:
                self._connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL UNIQUE,
                        file_type TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id INTEGER NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        chunk_id TEXT NOT NULL UNIQUE,
                        content TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        character_count INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(document_id) REFERENCES documents(id)
                            ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
                        ON document_chunks(document_id);
                    CREATE INDEX IF NOT EXISTS idx_documents_filename
                        ON documents(filename);
                    """
                )
                self._connection.commit()
                self._schema_ready = True
                logger.info(
                    "SQLite knowledge base ready at '%s'.",
                    self._db_path,
                )

    def _conn(self) -> sqlite3.Connection:
        self.initialize()
        assert self._connection is not None
        return self._connection

    def create_collection(self) -> SQLiteVectorStoreService:
        """Compatibility helper mirroring Chroma collection bootstrap."""
        self.initialize()
        return self

    def document_exists(self, filename: str) -> bool:
        """Return whether a document row exists for ``filename``."""
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty.")
        row = self._conn().execute(
            "SELECT 1 FROM documents WHERE filename = ? LIMIT 1",
            (filename.strip(),),
        ).fetchone()
        return row is not None

    def add_document(
        self,
        filename: str,
        file_type: str = "",
        *,
        replace_existing: bool = True,
    ) -> int:
        """Insert a document row and return its id.

        Args:
            filename: Unique source file name.
            file_type: Optional file extension/type label.
            replace_existing: When true, delete an existing document first.

        Returns:
            int: Document primary key.
        """
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty.")

        safe_filename = filename.strip()
        extension = file_type.strip() or Path(safe_filename).suffix.lstrip(".")

        with self._lock:
            conn = self._conn()
            if self.document_exists(safe_filename):
                if not replace_existing:
                    raise ValueError(
                        f"Document '{safe_filename}' is already indexed."
                    )
                self.delete_document(safe_filename)

            cursor = conn.execute(
                """
                INSERT INTO documents (filename, file_type, created_at)
                VALUES (?, ?, ?)
                """,
                (safe_filename, extension, _utc_now()),
            )
            conn.commit()
            document_id = int(cursor.lastrowid)
            logger.info(
                "SQLite document inserted (id=%s, filename=%s).",
                document_id,
                safe_filename,
            )
            return document_id

    def add_chunks(
        self,
        document_id: int,
        chunks: list[dict[str, Any]],
    ) -> int:
        """Insert chunk rows for an existing document.

        Each chunk dict must include:
        ``chunk_index``, ``content``, ``embedding``, ``character_count``,
        and preferably ``chunk_id``.
        """
        if not chunks:
            raise ValueError("chunks list cannot be empty.")

        with self._lock:
            conn = self._conn()
            document = conn.execute(
                "SELECT id FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if document is None:
                raise ValueError(f"Document id {document_id} does not exist.")

            created_at = _utc_now()
            rows: list[tuple[Any, ...]] = []
            for chunk in chunks:
                embedding = chunk.get("embedding")
                if not isinstance(embedding, list) or not embedding:
                    raise ValueError("Each chunk must include an embedding list.")
                chunk_index = int(chunk["chunk_index"])
                content = str(chunk["content"])
                character_count = int(
                    chunk.get("character_count") or len(content)
                )
                chunk_id = str(
                    chunk.get("chunk_id")
                    or f"doc{document_id}-chunk-{chunk_index}"
                )
                rows.append(
                    (
                        document_id,
                        chunk_index,
                        chunk_id,
                        content,
                        json.dumps(embedding),
                        character_count,
                        created_at,
                    )
                )

            conn.executemany(
                """
                INSERT INTO document_chunks (
                    document_id,
                    chunk_index,
                    chunk_id,
                    content,
                    embedding,
                    character_count,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            logger.info(
                "SQLite inserted %s chunks for document_id=%s.",
                len(rows),
                document_id,
            )
            return len(rows)

    def delete_document(self, filename: str) -> int:
        """Delete a document and cascade-delete its chunks.

        Returns:
            int: Number of deleted chunk rows.
        """
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty.")

        safe_filename = filename.strip()
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT id FROM documents WHERE filename = ?",
                (safe_filename,),
            ).fetchone()
            if row is None:
                return 0

            document_id = int(row["id"])
            count_row = conn.execute(
                "SELECT COUNT(*) AS total FROM document_chunks WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            deleted_chunks = int(count_row["total"] if count_row else 0)
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            conn.commit()
            logger.info(
                "SQLite deleted document '%s' (%s chunks).",
                safe_filename,
                deleted_chunks,
            )
            return deleted_chunks

    def list_documents(self) -> list[dict[str, Any]]:
        """List indexed documents with chunk counts."""
        rows = self._conn().execute(
            """
            SELECT
                d.id,
                d.filename,
                d.file_type,
                d.created_at,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN document_chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.filename COLLATE NOCASE
            """
        ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "filename": str(row["filename"]),
                "file_type": str(row["file_type"]),
                "created_at": str(row["created_at"]),
                "chunk_count": int(row["chunk_count"]),
            }
            for row in rows
        ]

    def count_documents(self) -> int:
        """Return the number of indexed documents."""
        row = self._conn().execute(
            "SELECT COUNT(*) AS total FROM documents"
        ).fetchone()
        return int(row["total"] if row else 0)

    def count_chunks(self) -> int:
        """Return the number of indexed chunks."""
        row = self._conn().execute(
            "SELECT COUNT(*) AS total FROM document_chunks"
        ).fetchone()
        return int(row["total"] if row else 0)

    def count(self) -> int:
        """Compatibility alias used by document stats endpoints."""
        return self.count_chunks()

    def count_by_filename(self, filename: str) -> int:
        """Count chunks stored for a source filename."""
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty.")
        row = self._conn().execute(
            """
            SELECT COUNT(c.id) AS total
            FROM document_chunks c
            INNER JOIN documents d ON d.id = c.document_id
            WHERE d.filename = ?
            """,
            (filename.strip(),),
        ).fetchone()
        return int(row["total"] if row else 0)

    def get_document_chunks(self, filename: str) -> list[dict[str, Any]]:
        """Return stored chunks for a document (without embeddings)."""
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty.")

        rows = self._conn().execute(
            """
            SELECT
                c.chunk_id,
                c.chunk_index,
                c.content,
                c.character_count,
                d.filename
            FROM document_chunks c
            INNER JOIN documents d ON d.id = c.document_id
            WHERE d.filename = ?
            ORDER BY c.chunk_index ASC
            """,
            (filename.strip(),),
        ).fetchall()
        return [
            {
                "chunk_id": str(row["chunk_id"]),
                "chunk_index": int(row["chunk_index"]),
                "content": str(row["content"]),
                "character_count": int(row["character_count"]),
                "filename": str(row["filename"]),
            }
            for row in rows
        ]

    def _iter_candidate_chunks(
        self,
        filename: str | None = None,
    ) -> list[sqlite3.Row]:
        if filename:
            return self._conn().execute(
                """
                SELECT
                    c.chunk_id,
                    c.chunk_index,
                    c.content,
                    c.character_count,
                    c.embedding,
                    d.filename
                FROM document_chunks c
                INNER JOIN documents d ON d.id = c.document_id
                WHERE d.filename = ?
                """,
                (filename,),
            ).fetchall()

        return self._conn().execute(
            """
            SELECT
                c.chunk_id,
                c.chunk_index,
                c.content,
                c.character_count,
                c.embedding,
                d.filename
            FROM document_chunks c
            INNER JOIN documents d ON d.id = c.document_id
            """
        ).fetchall()

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filename: str | None = None,
        *,
        n_results: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search stored embeddings with cosine similarity.

        Returns Chroma-compatible items:
        ``{id, document, metadata, distance}`` where
        ``distance = 1 - cosine_similarity``.
        """
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty.")

        limit = n_results if n_results is not None else top_k
        if limit <= 0:
            raise ValueError("top_k/n_results must be greater than zero.")

        filter_filename = filename
        if filter_filename is None and where and "filename" in where:
            filter_filename = str(where["filename"])

        if filter_filename is not None:
            filter_filename = filter_filename.strip() or None

        rows = self._iter_candidate_chunks(filter_filename)
        if not rows:
            return []

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            try:
                embedding = json.loads(str(row["embedding"]))
            except json.JSONDecodeError:
                continue
            if not isinstance(embedding, list):
                continue
            similarity = _cosine_similarity(
                query_embedding,
                [float(value) for value in embedding],
            )
            scored.append((similarity, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:limit]

        results: list[dict[str, Any]] = []
        for similarity, row in top:
            similarity_clamped = max(0.0, min(1.0, float(similarity)))
            distance = 1.0 - similarity_clamped
            results.append(
                {
                    "id": str(row["chunk_id"]),
                    "document": str(row["content"]),
                    "metadata": {
                        "filename": str(row["filename"]),
                        "chunk_index": int(row["chunk_index"]),
                        "chunk_id": str(row["chunk_id"]),
                        "character_count": int(row["character_count"]),
                    },
                    "distance": distance,
                    # Convenience field for SQLite-native consumers/tests.
                    "similarity_score": similarity_clamped,
                    "chunk_id": str(row["chunk_id"]),
                    "filename": str(row["filename"]),
                    "chunk_index": int(row["chunk_index"]),
                    "content": str(row["content"]),
                }
            )
        return results

    def _validate_metadatas(self, metadatas: list[dict[str, Any]]) -> None:
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
        """Indexer-compatible bulk insert into SQLite."""
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

        filenames = {str(item["filename"]) for item in metadatas}
        if len(filenames) != 1:
            raise ValueError(
                "SQLite add_documents currently supports one filename per batch."
            )
        filename = next(iter(filenames))

        chunks = [
            {
                "chunk_index": int(metadatas[index]["chunk_index"]),
                "chunk_id": str(document_ids[index]),
                "content": documents[index],
                "embedding": embeddings[index],
                "character_count": int(metadatas[index]["character_count"]),
            }
            for index in range(len(documents))
        ]

        document_id = self.add_document(
            filename=filename,
            file_type=Path(filename).suffix.lstrip("."),
            replace_existing=True,
        )
        return self.add_chunks(document_id=document_id, chunks=chunks)


def get_sqlite_vector_store_service() -> SQLiteVectorStoreService:
    """Return the shared ``SQLiteVectorStoreService`` singleton."""
    return SQLiteVectorStoreService()
