"""Unit tests for SQLite vector store and provider selection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.core.config import get_settings
from app.services.retriever import RetrieverService, get_retriever_service
from app.services.sqlite_vector_store import (
    SQLiteVectorStoreService,
    get_sqlite_vector_store_service,
)
from app.services.vector_store import (
    VectorStoreService,
    get_vector_store_service,
    reset_vector_store_factory,
)


def _store(tmp_path: Path) -> SQLiteVectorStoreService:
    db_path = tmp_path / "knowledge_base.db"
    # Configure before singleton creation.
    import os

    os.environ["VECTOR_STORE_PROVIDER"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = str(db_path)
    get_settings.cache_clear()
    reset_vector_store_factory()
    SQLiteVectorStoreService._instance = None

    store = get_sqlite_vector_store_service()
    # Force path from current settings (fresh singleton).
    store._db_path = db_path.resolve()
    store._connection = None
    store._schema_ready = False
    store.initialize()
    return store


def test_sqlite_db_creation_and_schema(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store._db_path.exists()
    assert store.count_documents() == 0
    assert store.count_chunks() == 0


def test_document_and_chunk_insertion(tmp_path: Path) -> None:
    store = _store(tmp_path)
    document_id = store.add_document("policy.txt", file_type="txt")
    inserted = store.add_chunks(
        document_id,
        [
            {
                "chunk_index": 0,
                "chunk_id": "c0",
                "content": "Alpha content",
                "embedding": [1.0, 0.0, 0.0],
                "character_count": 13,
            },
            {
                "chunk_index": 1,
                "chunk_id": "c1",
                "content": "Beta content",
                "embedding": [0.0, 1.0, 0.0],
                "character_count": 12,
            },
        ],
    )
    assert inserted == 2
    assert store.count_documents() == 1
    assert store.count_chunks() == 2
    assert store.document_exists("policy.txt")
    chunks = store.get_document_chunks("policy.txt")
    assert [item["chunk_id"] for item in chunks] == ["c0", "c1"]


def test_duplicate_document_protection(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_document("dup.txt", file_type="txt", replace_existing=False)
    with pytest.raises(ValueError, match="already indexed"):
        store.add_document("dup.txt", file_type="txt", replace_existing=False)

    # Compatibility path replaces existing document instead of duplicating.
    store.add_documents(
        documents=["one", "two"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[
            {
                "filename": "dup.txt",
                "chunk_index": 0,
                "chunk_id": "n0",
                "character_count": 3,
            },
            {
                "filename": "dup.txt",
                "chunk_index": 1,
                "chunk_id": "n1",
                "character_count": 3,
            },
        ],
    )
    assert store.count_documents() == 1
    assert store.count_chunks() == 2


def test_document_listing_and_deletion_cascade(tmp_path: Path) -> None:
    store = _store(tmp_path)
    document_id = store.add_document("delete-me.txt", file_type="txt")
    store.add_chunks(
        document_id,
        [
            {
                "chunk_index": 0,
                "chunk_id": "d0",
                "content": "to delete",
                "embedding": [0.1, 0.2],
                "character_count": 9,
            }
        ],
    )
    listed = store.list_documents()
    assert listed[0]["filename"] == "delete-me.txt"
    assert listed[0]["chunk_count"] == 1

    deleted = store.delete_document("delete-me.txt")
    assert deleted == 1
    assert store.count_documents() == 0
    assert store.count_chunks() == 0
    assert store.get_document_chunks("delete-me.txt") == []


def test_cosine_similarity_ordering_top_k_and_filename_filter(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.add_documents(
        documents=["exact match text", "orthogonal text"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        metadatas=[
            {
                "filename": "a.txt",
                "chunk_index": 0,
                "chunk_id": "a0",
                "character_count": 16,
            },
            {
                "filename": "a.txt",
                "chunk_index": 1,
                "chunk_id": "a1",
                "character_count": 15,
            },
        ],
    )
    store.add_documents(
        documents=["other file text"],
        embeddings=[[0.8, 0.2, 0.0]],
        metadatas=[
            {
                "filename": "b.txt",
                "chunk_index": 0,
                "chunk_id": "b0",
                "character_count": 14,
            }
        ],
    )

    results = store.similarity_search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "a0"
    assert results[0]["similarity_score"] >= results[1]["similarity_score"]
    assert results[0]["distance"] == pytest.approx(
        1.0 - results[0]["similarity_score"]
    )

    filtered = store.similarity_search(
        [1.0, 0.0, 0.0],
        top_k=5,
        filename="b.txt",
    )
    assert len(filtered) == 1
    assert filtered[0]["filename"] == "b.txt"

    filtered_where = store.similarity_search(
        [1.0, 0.0, 0.0],
        n_results=5,
        where={"filename": "a.txt"},
    )
    assert {item["chunk_id"] for item in filtered_where} == {"a0", "a1"}


def test_provider_selection_chromadb_and_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "chromadb")
    get_settings.cache_clear()
    reset_vector_store_factory()
    VectorStoreService._instance = None
    store = get_vector_store_service()
    assert isinstance(store, VectorStoreService)

    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "provider.db"))
    get_settings.cache_clear()
    reset_vector_store_factory()
    SQLiteVectorStoreService._instance = None
    store = get_vector_store_service()
    assert isinstance(store, SQLiteVectorStoreService)


def test_unsupported_vector_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "chromadb")
    get_settings.cache_clear()
    reset_vector_store_factory()

    monkeypatch.setattr(
        "app.services.vector_store.get_settings",
        lambda: MagicMock(VECTOR_STORE_PROVIDER="pinecone"),
    )
    with pytest.raises(RuntimeError, match="Unsupported VECTOR_STORE_PROVIDER"):
        get_vector_store_service()


def test_retriever_is_provider_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "retriever.db"))
    get_settings.cache_clear()
    reset_vector_store_factory()
    SQLiteVectorStoreService._instance = None
    RetrieverService._instance = None

    store = get_sqlite_vector_store_service()
    store._db_path = (tmp_path / "retriever.db").resolve()
    store._connection = None
    store._schema_ready = False
    store.initialize()
    store.add_documents(
        documents=["Foundry local retrieval works"],
        embeddings=[[1.0, 0.0]],
        metadatas=[
            {
                "filename": "local.txt",
                "chunk_index": 0,
                "chunk_id": "l0",
                "character_count": 30,
            }
        ],
    )

    mock_embedding = MagicMock()
    mock_embedding.embed_text.return_value = [1.0, 0.0]
    monkeypatch.setattr(
        "app.services.retriever.get_embedding_service",
        lambda: mock_embedding,
    )
    monkeypatch.setattr(
        "app.services.retriever.get_vector_store_service",
        lambda: store,
    )
    RetrieverService._instance = None

    results = get_retriever_service().retrieve("Foundry", top_k=1)
    assert len(results) == 1
    assert results[0]["filename"] == "local.txt"
    assert results[0]["chunk_id"] == "l0"
    assert results[0]["similarity_score"] == pytest.approx(1.0)
