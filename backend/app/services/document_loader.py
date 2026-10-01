"""Document loading and text extraction utilities."""

from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt"}


def validate_file_extension(filename: str) -> str:
    """Validate and normalize a document file extension.

    Args:
        filename: Original or stored file name.

    Returns:
        str: Lowercase file extension including the leading dot.

    Raises:
        ValueError: If the extension is missing or unsupported.
    """
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{extension or 'unknown'}'. "
            f"Supported types: {supported}."
        )
    return extension


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text content from a PDF file.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        str: Extracted plain text.

    Raises:
        ValueError: If no usable text can be extracted.
        OSError: If the file cannot be read.
    """
    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("No extractable text found in the PDF document.")
    return text


def extract_text_from_docx(file_path: Path) -> str:
    """Extract text content from a DOCX file.

    Args:
        file_path: Absolute or relative path to the DOCX file.

    Returns:
        str: Extracted plain text.

    Raises:
        ValueError: If no usable text can be extracted.
        OSError: If the file cannot be read.
    """
    document = Document(str(file_path))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    text = "\n".join(paragraphs).strip()
    if not text:
        raise ValueError("No extractable text found in the DOCX document.")
    return text


def extract_text_from_txt(file_path: Path) -> str:
    """Extract text content from a UTF-8 TXT file.

    Args:
        file_path: Absolute or relative path to the TXT file.

    Returns:
        str: Extracted plain text.

    Raises:
        ValueError: If the file is empty or contains only whitespace.
        UnicodeDecodeError: If the file is not valid UTF-8.
        OSError: If the file cannot be read.
    """
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("No extractable text found in the TXT document.")
    return text


def extract_text(file_path: Path) -> str:
    """Extract text from a supported document based on its extension.

    Args:
        file_path: Path to the document file.

    Returns:
        str: Extracted plain text.

    Raises:
        ValueError: If the file type is unsupported or text is empty.
    """
    extension = validate_file_extension(file_path.name)

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)
    if extension == ".docx":
        return extract_text_from_docx(file_path)
    if extension == ".txt":
        return extract_text_from_txt(file_path)

    raise ValueError(f"Unsupported file type '{extension}'.")
