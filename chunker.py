"""Fixed-size text chunking with overlap."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    text: str
    source_file: str
    chunk_index: int


def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """
    Split text into fixed-size chunks with overlap.

    Args:
        text: The text to chunk.
        source_file: Name of the source file.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of Chunk objects.
    """
    if not text or not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source_file=source_file,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

        start += chunk_size - chunk_overlap

    logger.debug(f"Created {len(chunks)} chunks from {source_file}")
    return chunks


def chunk_documents(
    documents: list[tuple[str, str]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """
    Chunk multiple documents.

    Args:
        documents: List of (filename, content) tuples.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of all Chunk objects from all documents.
    """
    all_chunks = []

    for filename, content in documents:
        chunks = chunk_text(content, filename, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents")
    return all_chunks
