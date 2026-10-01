"""Pydantic schemas for chat and semantic search APIs."""

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """Request body for semantic document search."""

    query: str = Field(..., description="Natural-language search query.")
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of chunks to return.",
    )
    filename: str | None = Field(
        default=None,
        description="Optional source filename filter.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Normalize and validate the search query.

        Args:
            value: Raw query string.

        Returns:
            str: Stripped query string.

        Raises:
            ValueError: If the query is shorter than 2 characters.
        """
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("query must be at least 2 characters long.")
        return cleaned

    @field_validator("filename")
    @classmethod
    def normalize_filename(cls, value: str | None) -> str | None:
        """Normalize optional filename filter.

        Args:
            value: Optional filename value.

        Returns:
            str | None: Stripped filename or ``None``.
        """
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class SearchResult(BaseModel):
    """A single ranked retrieval hit."""

    rank: int = Field(..., description="1-based result rank.")
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    filename: str = Field(..., description="Source document file name.")
    chunk_index: int = Field(..., description="Zero-based chunk index.")
    content: str = Field(..., description="Retrieved chunk text.")
    character_count: int = Field(..., description="Chunk character count.")
    distance: float = Field(..., description="Vector distance from the query.")
    similarity_score: float = Field(
        ...,
        description=(
            "Similarity score derived as max(0.0, min(1.0, 1.0 - distance))."
        ),
    )


class SearchResponse(BaseModel):
    """Response payload for semantic document search."""

    query: str = Field(..., description="Original search query.")
    total_results: int = Field(..., description="Number of returned results.")
    results: list[SearchResult] = Field(
        default_factory=list,
        description="Ranked retrieval results.",
    )


class ChatRequest(BaseModel):
    """Request body for RAG question answering."""

    query: str = Field(..., description="Natural-language question.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional retrieval depth override.",
    )
    filename: str | None = Field(
        default=None,
        description="Optional source filename filter.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Normalize and validate the chat query.

        Args:
            value: Raw query string.

        Returns:
            str: Stripped query string.

        Raises:
            ValueError: If the query is shorter than 2 characters.
        """
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("query must be at least 2 characters long.")
        return cleaned

    @field_validator("filename")
    @classmethod
    def normalize_filename(cls, value: str | None) -> str | None:
        """Normalize optional filename filter.

        Args:
            value: Optional filename value.

        Returns:
            str | None: Stripped filename or ``None``.
        """
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class ChatSource(BaseModel):
    """A cited source chunk used to ground a RAG answer."""

    source_number: int = Field(..., description="1-based source number.")
    filename: str = Field(..., description="Source document file name.")
    chunk_index: int = Field(..., description="Zero-based chunk index.")
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    similarity_score: float = Field(..., description="Chunk similarity score.")
    excerpt: str = Field(
        ...,
        description="Short preview of the retrieved passage (max 220 chars).",
    )
    content: str = Field(
        ...,
        description="Full retrieved passage text without truncation.",
    )


class ChatResponse(BaseModel):
    """Response payload for RAG question answering."""

    query: str = Field(..., description="Original user question.")
    answer: str = Field(..., description="Grounded natural-language answer.")
    retrieved_count: int = Field(
        ...,
        description="Number of context chunks used after filtering.",
    )
    sources: list[ChatSource] = Field(
        default_factory=list,
        description="Cited source chunks.",
    )
    model: str = Field(..., description="LLM model identifier.")
