"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["openrouter", "foundry_local"]
VectorStoreProvider = Literal["chromadb", "sqlite"]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and `.env`."""

    APP_NAME: str = "corporate"
    APP_VERSION: str = "1.0.0"
    UPLOAD_DIR: str = "data/uploads"
    VECTOR_STORE_DIR: str = "data/vector_store"
    VECTOR_STORE_PROVIDER: VectorStoreProvider = "chromadb"
    SQLITE_DB_PATH: str = "data/knowledge_base.db"
    EMBEDDING_MODEL: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    EMBEDDING_MODEL_LOCAL_PATH: str = (
        "models/paraphrase-multilingual-MiniLM-L12-v2"
    )
    OFFLINE_MODE: bool = False
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE_MB: int = 10

    LLM_PROVIDER: LLMProvider = "openrouter"
    LLM_MODEL: str = "openai/gpt-4o-mini"
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_API_KEY: str = ""
    LLM_TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=700, gt=0)
    FOUNDRY_MODEL_ALIAS: str = "qwen2.5-0.5b"
    RAG_TOP_K: int = Field(default=5, ge=1, le=20)
    MIN_SIMILARITY_SCORE: float = Field(default=0.25, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("LLM_PROVIDER", mode="before")
    @classmethod
    def normalize_llm_provider(cls, value: object) -> str:
        """Normalize and validate the configured LLM provider.

        Args:
            value: Raw provider value from the environment.

        Returns:
            str: Normalized provider identifier.

        Raises:
            ValueError: If the provider is unsupported.
        """
        provider = str(value or "openrouter").strip().lower()
        if provider not in {"openrouter", "foundry_local"}:
            raise ValueError(
                "LLM_PROVIDER must be one of: openrouter, foundry_local."
            )
        return provider

    @field_validator("FOUNDRY_MODEL_ALIAS", mode="before")
    @classmethod
    def normalize_foundry_alias(cls, value: object) -> str:
        """Normalize the Foundry Local model alias.

        Args:
            value: Raw alias value from the environment.

        Returns:
            str: Non-empty model alias.
        """
        alias = str(value or "").strip()
        return alias or "qwen2.5-0.5b"

    @field_validator("VECTOR_STORE_PROVIDER", mode="before")
    @classmethod
    def normalize_vector_store_provider(cls, value: object) -> str:
        """Normalize and validate the vector store provider.

        Args:
            value: Raw provider value from the environment.

        Returns:
            str: Normalized provider identifier.

        Raises:
            ValueError: If the provider is unsupported.
        """
        provider = str(value or "chromadb").strip().lower()
        if provider not in {"chromadb", "sqlite"}:
            raise ValueError(
                "VECTOR_STORE_PROVIDER must be one of: chromadb, sqlite."
            )
        return provider

    @field_validator("SQLITE_DB_PATH", mode="before")
    @classmethod
    def normalize_sqlite_db_path(cls, value: object) -> str:
        """Normalize the SQLite database path.

        Args:
            value: Raw path value from the environment.

        Returns:
            str: Non-empty SQLite database path.
        """
        path = str(value or "").strip()
        return path or "data/knowledge_base.db"

    @field_validator("EMBEDDING_MODEL_LOCAL_PATH", mode="before")
    @classmethod
    def normalize_embedding_local_path(cls, value: object) -> str:
        """Normalize the local embedding model directory path.

        Args:
            value: Raw path value from the environment.

        Returns:
            str: Non-empty relative/absolute path for the local model.
        """
        path = str(value or "").strip()
        return path or "models/paraphrase-multilingual-MiniLM-L12-v2"

    @field_validator("OFFLINE_MODE", mode="before")
    @classmethod
    def normalize_offline_mode(cls, value: object) -> bool:
        """Normalize boolean offline mode from environment strings.

        Args:
            value: Raw offline mode value.

        Returns:
            bool: Whether fully offline mode is enabled.
        """
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @property
    def active_llm_model_name(self) -> str:
        """Return the model name that should appear in API responses.

        Returns:
            str: OpenRouter model id or Foundry Local alias.
        """
        if self.LLM_PROVIDER == "foundry_local":
            return self.FOUNDRY_MODEL_ALIAS
        return self.LLM_MODEL


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Returns:
        Settings: Application configuration values.
    """
    return Settings()
