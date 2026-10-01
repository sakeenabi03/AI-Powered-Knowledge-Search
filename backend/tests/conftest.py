"""Shared pytest fixtures for backend tests."""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.services.document_indexer import DocumentIndexerService
from app.services.embedding_service import EmbeddingService
from app.services.foundry_local_service import FoundryLocalService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.retriever import RetrieverService
from app.services.sqlite_vector_store import SQLiteVectorStoreService
from app.services.vector_store import VectorStoreService, reset_vector_store_factory


@pytest.fixture(autouse=True)
def reset_singletons() -> None:
    """Reset service singletons and settings cache between tests."""
    RAGService._instance = None
    LLMService._instance = None
    FoundryLocalService._instance = None
    EmbeddingService._instance = None
    DocumentIndexerService._instance = None
    RetrieverService._instance = None
    VectorStoreService._instance = None
    SQLiteVectorStoreService._instance = None
    reset_vector_store_factory()
    get_settings.cache_clear()
    yield
    RAGService._instance = None
    LLMService._instance = None
    FoundryLocalService._instance = None
    EmbeddingService._instance = None
    DocumentIndexerService._instance = None
    RetrieverService._instance = None
    VectorStoreService._instance = None
    SQLiteVectorStoreService._instance = None
    reset_vector_store_factory()
    get_settings.cache_clear()
