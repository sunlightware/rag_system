"""Configuration and environment variable loading."""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Configuration for the RAG system."""

    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    chat_model: str = "gpt-4"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 3
    log_level: str = "INFO"


def load_config() -> Config:
    """Load configuration from environment variables."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "rag-index")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable is required")

    return Config(
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
