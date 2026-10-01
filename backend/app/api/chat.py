"""Chat / Q&A API routes."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatSource,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.llm_service import LLMConfigurationError, LLMGenerationError
from app.services.rag_service import get_rag_service
from app.services.retriever import get_retriever_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_upload_file(filename: str) -> Path:
    """Resolve a filename safely under the uploads directory.

    Args:
        filename: Candidate source file name.

    Returns:
        Path: Absolute path to the file inside the uploads directory.

    Raises:
        HTTPException: If the filename is invalid or the file does not exist.
    """
    safe_name = Path(filename).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename filter.",
        )

    upload_dir = Path(get_settings().UPLOAD_DIR).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = (upload_dir / safe_name).resolve()

    try:
        target.relative_to(upload_dir)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename filter.",
        ) from exc

    if not target.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{safe_name}' not found.",
        )

    return target


@router.get("/")
async def chat_root() -> dict[str, str]:
    """Return a readiness message for the chat API.

    Returns:
        dict[str, str]: Chat API readiness payload.
    """
    return {"message": "Chat API is ready"}


@router.post("/search", response_model=SearchResponse)
async def search_documents(payload: SearchRequest) -> SearchResponse:
    """Search indexed document chunks by semantic similarity.

    Args:
        payload: Search query, result limit, and optional filename filter.

    Returns:
        SearchResponse: Ranked retrieval results without LLM generation.

    Raises:
        HTTPException: For invalid input, missing files, or retrieval failures.
    """
    safe_filename: str | None = None
    if payload.filename is not None:
        safe_filename = _resolve_upload_file(payload.filename).name

    try:
        results = get_retriever_service().retrieve(
            query=payload.query,
            top_k=payload.top_k,
            filename=safe_filename,
        )
        return SearchResponse(
            query=payload.query,
            total_results=len(results),
            results=[SearchResult(**item) for item in results],
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.exception("Retrieval failed due to embedding or vector store error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while searching documents.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error while searching documents.",
        ) from exc


@router.post("/ask", response_model=ChatResponse)
async def ask_question(payload: ChatRequest) -> ChatResponse:
    """Answer a question with retrieval-augmented generation.

    Args:
        payload: Question, optional top_k, and optional filename filter.

    Returns:
        ChatResponse: Grounded answer with cited sources.

    Raises:
        HTTPException: For invalid input, missing files, missing API key,
            or upstream LLM/service failures.
    """
    settings = get_settings()
    safe_filename: str | None = None
    if payload.filename is not None:
        safe_filename = _resolve_upload_file(payload.filename).name

    try:
        result = await get_rag_service().answer_question(
            query=payload.query,
            top_k=payload.top_k,
            filename=safe_filename,
        )
        return ChatResponse(
            query=payload.query,
            answer=str(result["answer"]),
            retrieved_count=int(result["retrieved_count"]),
            sources=[
                ChatSource(**source)
                for source in result["sources"]  # type: ignore[arg-type]
            ],
            model=settings.active_llm_model_name,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except LLMGenerationError as exc:
        logger.exception("LLM generation failed while answering question.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.exception("RAG service runtime error.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while answering question.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error while answering question.",
        ) from exc
