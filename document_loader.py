"""Load and parse documents from various formats."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Union

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def load_text_file(file_path: Path) -> str:
    """Load content from a text or markdown file."""
    return file_path.read_text(encoding="utf-8")


def load_pdf_file(file_path: Path) -> str:
    """Load content from a PDF file."""
    from PyPDF2 import PdfReader

    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


def load_docx_file(file_path: Path) -> str:
    """Load content from a DOCX file."""
    from docx import Document

    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def load_document(file_path: Path) -> Optional[Tuple[str, str]]:
    """
    Load a document from the given path.

    Returns:
        Tuple of (filename, content) or None if unsupported format.
    """
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        logger.warning(f"Unsupported file format: {file_path}")
        return None

    try:
        if suffix in {".txt", ".md"}:
            content = load_text_file(file_path)
        elif suffix == ".pdf":
            content = load_pdf_file(file_path)
        elif suffix == ".docx":
            content = load_docx_file(file_path)
        else:
            return None

        logger.info(f"Loaded document: {file_path.name} ({len(content)} chars)")
        return (file_path.name, content)

    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def load_documents_from_directory(
    directory: Union[str, Path],
) -> List[Tuple[str, str]]:
    """
    Load all supported documents from a directory.

    Args:
        directory: Path to the directory containing documents.

    Returns:
        List of (filename, content) tuples.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    documents = []
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            result = load_document(file_path)
            if result:
                documents.append(result)

    logger.info(f"Loaded {len(documents)} documents from {directory}")
    return documents
