"""Pydantic schemas for document API responses."""

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response returned after a successful document upload and indexing."""

    filename: str = Field(..., description="Stored unique file name.")
    file_type: str = Field(..., description="File extension, e.g. '.pdf'.")
    file_size: int = Field(..., description="File size in bytes.")
    character_count: int = Field(..., description="Number of extracted characters.")
    total_chunks: int = Field(..., description="Number of text chunks created.")
    vector_count: int = Field(..., description="Number of vectors stored in ChromaDB.")
    message: str = Field(..., description="Human-readable status message.")


class DocumentListItem(BaseModel):
    """Metadata for a single uploaded document."""

    filename: str = Field(..., description="Stored file name.")
    file_type: str = Field(..., description="File extension, e.g. '.pdf'.")
    file_size: int = Field(..., description="File size in bytes.")
    indexed_chunks: int = Field(
        ...,
        description="Number of ChromaDB chunk vectors for this file.",
    )


class DocumentDeleteResponse(BaseModel):
    """Response returned after deleting a document and its vectors."""

    message: str = Field(..., description="Human-readable status message.")
    filename: str = Field(..., description="Deleted file name.")
    deleted_vectors: int = Field(
        ...,
        description="Number of ChromaDB vectors removed for the file.",
    )


class DocumentStatsResponse(BaseModel):
    """Aggregate document and vector store statistics."""

    total_documents: int = Field(
        ...,
        description="Number of supported files in the uploads directory.",
    )
    total_vectors: int = Field(
        ...,
        description="Total vectors stored in the ChromaDB collection.",
    )
    collection_name: str = Field(
        ...,
        description="ChromaDB collection name.",
    )


class DocumentReindexResponse(BaseModel):
    """Response returned after reindexing a stored document."""

    filename: str = Field(..., description="Reindexed file name.")
    total_chunks: int = Field(..., description="Number of chunks after reindex.")
    message: str = Field(..., description="Human-readable status message.")


class DocumentChunk(BaseModel):
    """A single text chunk with source metadata."""

    chunk_id: str = Field(..., description="Unique chunk identifier.")
    source: str = Field(..., description="Source document file name.")
    chunk_index: int = Field(..., description="Zero-based chunk order index.")
    content: str = Field(..., description="Chunk text content.")
    character_count: int = Field(..., description="Number of characters in the chunk.")


class DocumentChunkPreviewResponse(BaseModel):
    """Preview response for document chunking."""

    filename: str = Field(..., description="Stored document file name.")
    total_chunks: int = Field(..., description="Total number of produced chunks.")
    chunk_size: int = Field(..., description="Chunk size used for splitting.")
    chunk_overlap: int = Field(..., description="Chunk overlap used for splitting.")
    chunks: list[DocumentChunk] = Field(
        ...,
        description="Preview of the first N chunks.",
    )
