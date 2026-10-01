"""Sentence Transformers based embedding service."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Final

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

logger = logging.getLogger(__name__)

LOCAL_MODEL_MARKERS: Final[tuple[str, ...]] = (
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
)


def _resolve_local_model_path(raw_path: str) -> Path:
    """Resolve a configured local model path against the backend root.

    Args:
        raw_path: Relative or absolute path from settings.

    Returns:
        Path: Absolute path to the candidate model directory.
    """
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    backend_root = Path(__file__).resolve().parents[2]
    return (backend_root / path).resolve()


def _looks_like_sentence_transformer_model(directory: Path) -> bool:
    """Return whether ``directory`` appears to contain a saved ST model."""
    if not directory.is_dir():
        return False
    return any((directory / marker).is_file() for marker in LOCAL_MODEL_MARKERS)


def _apply_offline_environment_flags() -> None:
    """Set Hugging Face / Transformers offline flags for this process."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


class EmbeddingService:
    """Generate dense vector embeddings with a lazily loaded model.

    Prefers a locally saved Sentence Transformers directory so document upload
    and retrieval can run without internet access.
    """

    _instance: EmbeddingService | None = None
    _lock: Final[threading.Lock] = threading.Lock()

    def __new__(cls) -> EmbeddingService:
        """Return the shared ``EmbeddingService`` instance.

        Returns:
            EmbeddingService: Process-wide singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._model = None
                    instance._model_source = None
                    instance._initialized = True
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize instance attributes only once for the singleton."""
        if getattr(self, "_initialized", False):
            return
        self._model: SentenceTransformer | None = None
        self._model_source: str | None = None
        self._initialized = True

    def _load_from_local_path(self, local_path: Path) -> SentenceTransformer:
        """Load a Sentence Transformers model from a local directory.

        Args:
            local_path: Absolute path to a saved model directory.

        Returns:
            SentenceTransformer: Loaded local model.
        """
        logger.info("Loading embedding model from local path: %s", local_path)
        model = SentenceTransformer(
            str(local_path),
            local_files_only=True,
        )
        self._model_source = str(local_path)
        return model

    def _load_from_remote_id(self, model_name: str) -> SentenceTransformer:
        """Load a Sentence Transformers model from a Hugging Face model id.

        Args:
            model_name: Remote model identifier.

        Returns:
            SentenceTransformer: Loaded model (may download if online).
        """
        logger.info(
            "Loading embedding model from remote identifier '%s'...",
            model_name,
        )
        model = SentenceTransformer(model_name, local_files_only=False)
        self._model_source = model_name
        return model

    def load_model(self) -> SentenceTransformer:
        """Load the Sentence Transformers model if it is not already loaded.

        Preference order:
        1. Local directory from ``EMBEDDING_MODEL_LOCAL_PATH`` when valid
        2. Remote/hub id from ``EMBEDDING_MODEL`` when offline mode is disabled

        Returns:
            SentenceTransformer: Loaded embedding model.

        Raises:
            RuntimeError: If the model cannot be loaded, especially offline.
        """
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            settings = get_settings()
            local_path = _resolve_local_model_path(
                settings.EMBEDDING_MODEL_LOCAL_PATH
            )
            local_available = _looks_like_sentence_transformer_model(local_path)

            if settings.OFFLINE_MODE:
                _apply_offline_environment_flags()
                if not local_available:
                    raise RuntimeError(
                        "Local embedding model is not available. "
                        "Run scripts/download_embedding_model.py while online first."
                    )
                try:
                    self._model = self._load_from_local_path(local_path)
                    logger.info(
                        "Embedding model loaded successfully from local path "
                        "(offline mode)."
                    )
                    return self._model
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Failed to load local embedding model from '%s'.",
                        local_path,
                    )
                    raise RuntimeError(
                        "Local embedding model is not available. "
                        "Run scripts/download_embedding_model.py while online first."
                    ) from exc

            if local_available:
                try:
                    self._model = self._load_from_local_path(local_path)
                    logger.info(
                        "Embedding model loaded successfully from local path."
                    )
                    return self._model
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Failed to load local embedding model from '%s'; "
                        "falling back to remote model id.",
                        local_path,
                    )
                    # Fall through to remote load only when not in offline mode.

            try:
                self._model = self._load_from_remote_id(settings.EMBEDDING_MODEL)
                logger.info(
                    "Embedding model '%s' loaded successfully.",
                    settings.EMBEDDING_MODEL,
                )
                return self._model
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Failed to load embedding model '%s'.",
                    settings.EMBEDDING_MODEL,
                )
                hint = ""
                if not local_available:
                    hint = (
                        " Local model directory was not found at "
                        f"'{local_path}'. Run scripts/download_embedding_model.py "
                        "while online, or enable network access."
                    )
                raise RuntimeError(
                    f"Failed to load embedding model '{settings.EMBEDDING_MODEL}'.{hint}"
                ) from exc

    def embed_text(self, text: str) -> list[float]:
        """Create an embedding vector for a single text string.

        Args:
            text: Input text to embed.

        Returns:
            list[float]: Embedding vector as a Python list of floats.

        Raises:
            ValueError: If the text is empty or whitespace-only.
            RuntimeError: If embedding generation fails.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty for embedding.")

        try:
            model = self.load_model()
            vector = model.encode(
                text.strip(),
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embedding = np.asarray(vector, dtype=np.float32).tolist()
            logger.debug("Generated embedding with dimension %s.", len(embedding))
            return embedding
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to embed text.")
            raise RuntimeError("Failed to generate embedding for text.") from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Create embedding vectors for multiple documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            list[list[float]]: Embedding vectors as nested Python float lists.

        Raises:
            ValueError: If the input list is empty or contains blank texts.
            RuntimeError: If embedding generation fails.
        """
        if not texts:
            raise ValueError("Document list cannot be empty for embedding.")

        cleaned_texts = [text.strip() for text in texts]
        if any(not text for text in cleaned_texts):
            raise ValueError("Document list contains empty text entries.")

        try:
            model = self.load_model()
            vectors = model.encode(
                cleaned_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings = np.asarray(vectors, dtype=np.float32).tolist()
            logger.info("Generated embeddings for %s documents.", len(embeddings))
            return embeddings
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to embed documents.")
            raise RuntimeError("Failed to generate embeddings for documents.") from exc


def get_embedding_service() -> EmbeddingService:
    """Return the shared ``EmbeddingService`` singleton.

    Returns:
        EmbeddingService: Process-wide embedding service instance.
    """
    return EmbeddingService()
