"""Text splitting utilities for document chunking."""

import re
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(
    text: str,
    source_filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, object]]:
    """Split document text into overlapping chunks with metadata.

    Uses LangChain ``RecursiveCharacterTextSplitter`` to produce retrieval-ready
    segments and attaches source metadata to each chunk.

    Args:
        text: Full document text to split.
        source_filename: Original or stored source file name.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        list[dict[str, object]]: Chunk dictionaries containing ``chunk_id``,
        ``source``, ``chunk_index``, ``content``, and ``character_count``.

    Raises:
        ValueError: If the text is empty or chunk settings are invalid.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    cleaned_text = re.sub(r"[ \t]+", " ", text.strip())
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

    if not cleaned_text:
        raise ValueError("Text cannot be empty after cleaning.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    raw_chunks = splitter.split_text(cleaned_text)

    chunks: list[dict[str, object]] = []
    for content in raw_chunks:
        normalized = content.strip()
        if not normalized:
            continue

        chunks.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "source": source_filename,
                "chunk_index": len(chunks),
                "content": normalized,
                "character_count": len(normalized),
            }
        )

    if not chunks:
        raise ValueError("No non-empty chunks could be produced from the text.")

    return chunks
