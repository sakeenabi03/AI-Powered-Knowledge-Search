"""FastAPI application entry point for AI-Powered Knowledge Search."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, documents

app = FastAPI(
    title="AI-Powered Knowledge Search API",
    version="1.0.0",
    description="AI-Powered Knowledge Search document querying API powered by RAG",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])


@app.get("/")
async def root() -> dict[str, str]:
    """Return a simple API status payload.

    Returns:
        dict[str, str]: Welcome message and running status.
    """
    return {
        "message": "AI-Powered Knowledge Search API",
        "status": "running",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Return application health and version information.

    Returns:
        dict[str, str]: Health status and application version.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
    }
