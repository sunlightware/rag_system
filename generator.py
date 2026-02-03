"""GPT-4 response generation."""

from __future__ import annotations

import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer the question based on the context provided. If the context doesn't contain relevant information, say so."""


class Generator:
    """Generate responses using GPT-4 with retrieved context."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        question: str,
        context_chunks: list[dict],
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a response based on the question and retrieved context.

        Args:
            question: The user's question.
            context_chunks: List of chunk dicts with 'text' and 'source_file'.
            max_tokens: Maximum tokens in the response.

        Returns:
            Generated response string.
        """
        # Format context
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source_file", "unknown")
            text = chunk.get("text", "")
            context_parts.append(f"[{i}] (Source: {source})\n{text}")

        context = "\n\n".join(context_parts)

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        logger.debug(f"Generating response for: {question[:50]}...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content
        logger.info("Generated response")
        return answer
