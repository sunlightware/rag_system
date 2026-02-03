"""OpenAI embedding operations."""

from __future__ import annotations

import logging

from openai import OpenAI

from .chunker import Chunk

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for generating embeddings using OpenAI."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def get_embeddings_batch(
        self, texts: list[str], batch_size: int = 100
    ) -> list[list[float]]:
        """
        Get embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process per API call.

        Returns:
            List of embeddings in the same order as input texts.
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(f"Embedding batch {i // batch_size + 1}")

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_chunks(
        self, chunks: list[Chunk], batch_size: int = 100
    ) -> list[tuple[Chunk, list[float]]]:
        """
        Embed a list of chunks.

        Args:
            chunks: List of Chunk objects.
            batch_size: Number of chunks to process per API call.

        Returns:
            List of (chunk, embedding) tuples.
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.get_embeddings_batch(texts, batch_size)

        logger.info(f"Embedded {len(chunks)} chunks")
        return list(zip(chunks, embeddings))
