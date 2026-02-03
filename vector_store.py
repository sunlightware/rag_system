"""Pinecone vector store operations."""

from __future__ import annotations

import logging
import time

from pinecone import Pinecone, ServerlessSpec

from .chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Pinecone vector store for RAG system."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int = 1536,
    ):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self._index = None

    def _ensure_index_exists(self) -> None:
        """Create the index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            logger.info(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
            logger.info(f"Index {self.index_name} is ready")

    @property
    def index(self):
        """Get or create the Pinecone index."""
        if self._index is None:
            self._ensure_index_exists()
            self._index = self.pc.Index(self.index_name)
        return self._index

    def upsert_chunks(
        self,
        chunks_with_embeddings: list[tuple[Chunk, list[float]]],
        batch_size: int = 100,
    ) -> int:
        """
        Upsert chunks with their embeddings to the index.

        Args:
            chunks_with_embeddings: List of (chunk, embedding) tuples.
            batch_size: Number of vectors to upsert per batch.

        Returns:
            Number of vectors upserted.
        """
        vectors = []
        for chunk, embedding in chunks_with_embeddings:
            vector_id = f"{chunk.source_file}_{chunk.chunk_index}"
            vectors.append(
                {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "source_file": chunk.source_file,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    },
                }
            )

        # Upsert in batches
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch)
            total_upserted += len(batch)
            logger.debug(f"Upserted batch: {total_upserted}/{len(vectors)}")

        logger.info(f"Upserted {total_upserted} vectors to index")
        return total_upserted

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[dict]:
        """
        Query the index for similar chunks.

        Args:
            query_embedding: The embedding of the query text.
            top_k: Number of results to return.

        Returns:
            List of dicts with 'text', 'source_file', 'chunk_index', and 'score'.
        """
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )

        matches = []
        for match in results.matches:
            matches.append(
                {
                    "text": match.metadata["text"],
                    "source_file": match.metadata["source_file"],
                    "chunk_index": match.metadata["chunk_index"],
                    "score": match.score,
                }
            )

        logger.debug(f"Found {len(matches)} matches")
        return matches

    def delete_by_source(self, source_file: str) -> None:
        """
        Delete all vectors from a specific source file.

        Args:
            source_file: Name of the source file to delete vectors for.
        """
        self.index.delete(filter={"source_file": {"$eq": source_file}})
        logger.info(f"Deleted vectors for source: {source_file}")

    def delete_all(self) -> None:
        """Delete all vectors from the index."""
        self.index.delete(delete_all=True)
        logger.info("Deleted all vectors from index")

    def get_stats(self) -> dict:
        """Get index statistics."""
        return self.index.describe_index_stats()
