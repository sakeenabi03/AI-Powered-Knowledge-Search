"""Document management API routes."""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.core.config import get_settings
from app.schemas.document import (
    DocumentChunk,
    DocumentChunkPreviewResponse,
    DocumentDeleteResponse,
    DocumentListItem,
    DocumentReindexResponse,
    DocumentStatsResponse,
    DocumentUploadResponse,
)
from app.services.document_indexer import get_document_indexer
from app.services.document_loader import (
    SUPPORTED_EXTENSIONS,
    extract_text,
    validate_file_extension,
)
from app.services.text_splitter import split_text
from app.services.vector_store import get_vector_store_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_upload_dir() -> Path:
    """Resolve and ensure the upload directory exists.

    Returns:
        Path: Absolute path to the upload directory.
    """
    settings = get_settings()
    upload_dir = Path(settings.UPLOAD_DIR).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _safe_join_upload(upload_dir: Path, filename: str) -> Path:
    """Join a filename under the upload directory safely.

    Args:
        upload_dir: Absolute upload directory path.
        filename: Candidate file name (may contain traversal segments).

    Returns:
        Path: Resolved absolute path inside the upload directory.

    Raises:
        HTTPException: If the resolved path escapes the upload directory.
    """
    safe_name = Path(filename).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    destination = (upload_dir / safe_name).resolve()
    try:
        destination.relative_to(upload_dir)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename path.",
        ) from exc

    return destination


def _build_unique_filename(original_filename: str, extension: str) -> str:
    """Build a unique storage name from the original file name.

    Args:
        original_filename: Sanitized original file name.
        extension: Normalized lowercase extension.

    Returns:
        str: Unique file name such as ``report_ab12cd34.pdf``.
    """
    stem = Path(original_filename).stem or "document"
    return f"{stem}_{uuid.uuid4().hex[:8]}{extension}"


def _rollback_upload(destination: Path | None) -> None:
    """Remove a partially uploaded file and any indexed vectors.

    Args:
        destination: Path to the uploaded file, if it was created.
    """
    indexer = get_document_indexer()

    if destination is not None and destination.exists():
        filename = destination.name
        try:
            deleted_vectors = indexer.delete_document_index(filename)
            if deleted_vectors:
                logger.warning(
                    "Rollback removed %s vectors for '%s'.",
                    deleted_vectors,
                    filename,
                )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Rollback failed while deleting vectors for '%s'.",
                filename,
            )

        destination.unlink(missing_ok=True)
        logger.warning("Rollback deleted uploaded file '%s'.", filename)


@router.get("/")
async def documents_root() -> dict[str, str]:
    """Return a readiness message for the documents API.

    Returns:
        dict[str, str]: Documents API readiness payload.
    """
    return {"message": "Document API is ready"}


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """Upload a document, index it into ChromaDB, and return indexing stats.

    Args:
        file: Uploaded PDF, DOCX, or TXT file.

    Returns:
        DocumentUploadResponse: Metadata for the stored and indexed document.

    Raises:
        HTTPException: For invalid type, size limit, processing, or server errors.
    """
    settings = get_settings()
    upload_dir = _get_upload_dir()
    destination: Path | None = None

    try:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required.",
            )

        original_name = Path(file.filename).name
        try:
            extension = validate_file_extension(original_name)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        unique_name = _build_unique_filename(original_name, extension)
        destination = _safe_join_upload(upload_dir, unique_name)

        if destination.exists():
            unique_name = _build_unique_filename(original_name, extension)
            destination = _safe_join_upload(upload_dir, unique_name)

        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        file_size = 0

        try:
            with destination.open("wb") as buffer:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    file_size += len(chunk)
                    if file_size > max_bytes:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=(
                                f"File exceeds the maximum size of "
                                f"{settings.MAX_FILE_SIZE_MB} MB."
                            ),
                        )
                    buffer.write(chunk)
        except HTTPException:
            if destination.exists():
                destination.unlink(missing_ok=True)
            raise

        if file_size == 0:
            destination.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        try:
            index_result = get_document_indexer().index_document(destination)
        except ValueError as exc:
            _rollback_upload(destination)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document could not be processed: {exc}",
            ) from exc
        except (UnicodeDecodeError, OSError) as exc:
            _rollback_upload(destination)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document could not be processed: {exc}",
            ) from exc
        except RuntimeError as exc:
            _rollback_upload(destination)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        return DocumentUploadResponse(
            filename=unique_name,
            file_type=extension,
            file_size=file_size,
            character_count=index_result["character_count"],
            total_chunks=index_result["total_chunks"],
            vector_count=index_result["vector_count"],
            message="Document uploaded, processed and indexed successfully",
        )

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while uploading document.")
        _rollback_upload(destination)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error while uploading document.",
        ) from exc
    finally:
        await file.close()


@router.get("/list", response_model=list[DocumentListItem])
async def list_documents() -> list[DocumentListItem]:
    """List supported documents and their indexed chunk counts.

    Returns:
        list[DocumentListItem]: Documents sorted by file name.
    """
    try:
        upload_dir = _get_upload_dir()
        indexer = get_document_indexer()
        items: list[DocumentListItem] = []

        for path in sorted(upload_dir.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file():
                continue
            extension = path.suffix.lower()
            if extension not in SUPPORTED_EXTENSIONS:
                continue
            items.append(
                DocumentListItem(
                    filename=path.name,
                    file_type=extension,
                    file_size=path.stat().st_size,
                    indexed_chunks=indexer.count_indexed_chunks(path.name),
                )
            )

        return items
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while listing documents.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error while listing documents.",
        ) from exc


@router.get("/stats", response_model=DocumentStatsResponse)
async def get_document_stats() -> DocumentStatsResponse:
    """Return upload directory and vector store aggregate statistics.

    Returns:
        DocumentStatsResponse: Document and vector totals.
    """
    try:
        upload_dir = _get_upload_dir()
        total_documents = sum(
            1
            for path in upload_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        vector_store = get_vector_store_service()
        total_vectors = vector_store.count()
        collection_name = getattr(
            vector_store,
            "collection_name",
            "corporate_documents",
        )
        return DocumentStatsResponse(
            total_documents=total_documents,
            total_vectors=total_vectors,
            collection_name=str(collection_name),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while collecting document stats.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error while collecting document stats.",
        ) from exc


@router.get("/{filename}/chunks", response_model=DocumentChunkPreviewResponse)
async def preview_document_chunks(
    filename: str,
    limit: int = Query(default=5, ge=1, le=50),
    chunk_size: int | None = Query(default=None, gt=0),
    chunk_overlap: int | None = Query(default=None, ge=0),
) -> DocumentChunkPreviewResponse:
    """Extract text from a stored document and return a chunk preview.

    Args:
        filename: Stored document file name.
        limit: Maximum number of chunks to include in the response (1-50).
        chunk_size: Optional chunk size override. Defaults to settings.
        chunk_overlap: Optional chunk overlap override. Defaults to settings.

    Returns:
        DocumentChunkPreviewResponse: Chunk preview with totals and settings.

    Raises:
        HTTPException: If the file is missing or chunking fails.
    """
    settings = get_settings()
    resolved_chunk_size = (
        chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    )
    resolved_chunk_overlap = (
        chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
    )

    try:
        upload_dir = _get_upload_dir()
        target = _safe_join_upload(upload_dir, filename)

        try:
            validate_file_extension(target.name)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        if not target.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{Path(filename).name}' not found.",
            )

        try:
            text = extract_text(target)
            chunk_dicts = split_text(
                text=text,
                source_filename=target.name,
                chunk_size=resolved_chunk_size,
                chunk_overlap=resolved_chunk_overlap,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except (UnicodeDecodeError, OSError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document could not be processed: {exc}",
            ) from exc

        preview_chunks = [
            DocumentChunk(**chunk) for chunk in chunk_dicts[:limit]
        ]

        return DocumentChunkPreviewResponse(
            filename=target.name,
            total_chunks=len(chunk_dicts),
            chunk_size=resolved_chunk_size,
            chunk_overlap=resolved_chunk_overlap,
            chunks=preview_chunks,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while previewing document chunks.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error while previewing document chunks.",
        ) from exc


@router.post("/{filename}/reindex", response_model=DocumentReindexResponse)
async def reindex_document(filename: str) -> DocumentReindexResponse:
    """Rebuild ChromaDB vectors for an already uploaded document.

    Args:
        filename: Stored document file name.

    Returns:
        DocumentReindexResponse: Reindex confirmation payload.

    Raises:
        HTTPException: If the file is missing or reindexing fails.
    """
    try:
        upload_dir = _get_upload_dir()
        target = _safe_join_upload(upload_dir, filename)

        try:
            validate_file_extension(target.name)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        if not target.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{Path(filename).name}' not found.",
            )

        try:
            index_result = get_document_indexer().reindex_document(target)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document could not be processed: {exc}",
            ) from exc
        except (UnicodeDecodeError, OSError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document could not be processed: {exc}",
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        return DocumentReindexResponse(
            filename=target.name,
            total_chunks=index_result["total_chunks"],
            message="Document reindexed successfully",
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while reindexing document.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error while reindexing document.",
        ) from exc


@router.delete("/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str) -> DocumentDeleteResponse:
    """Delete a stored document and its ChromaDB vector records.

    Args:
        filename: Stored file name to delete.

    Returns:
        DocumentDeleteResponse: Deletion confirmation payload.

    Raises:
        HTTPException: If the file is invalid or not found.
    """
    try:
        upload_dir = _get_upload_dir()
        target = _safe_join_upload(upload_dir, filename)

        try:
            validate_file_extension(target.name)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        if not target.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{Path(filename).name}' not found.",
            )

        try:
            deleted_vectors = get_document_indexer().delete_document_index(
                target.name
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        target.unlink()
        return DocumentDeleteResponse(
            message="Document and vector records deleted successfully",
            filename=target.name,
            deleted_vectors=deleted_vectors,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while deleting document.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error while deleting document.",
        ) from exc
