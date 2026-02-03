"""CLI interface for the RAG system."""

from __future__ import annotations

import argparse
import logging
import sys

if __name__ == "__main__" or not __package__:
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rag_system.chunker import chunk_documents
    from rag_system.config import load_config, setup_logging
    from rag_system.document_loader import load_documents_from_directory
    from rag_system.embeddings import EmbeddingClient
    from rag_system.generator import Generator
    from rag_system.vector_store import VectorStore
else:
    from .chunker import chunk_documents
    from .config import load_config, setup_logging
    from .document_loader import load_documents_from_directory
    from .embeddings import EmbeddingClient
    from .generator import Generator
    from .vector_store import VectorStore

logger = logging.getLogger(__name__)


def cmd_index(args: argparse.Namespace) -> int:
    """Index documents from a directory."""
    config = load_config()
    setup_logging(config.log_level)

    logger.info(f"Indexing documents from: {args.dir}")

    # Load documents
    documents = load_documents_from_directory(args.dir)
    if not documents:
        logger.warning("No documents found to index")
        return 1

    # Chunk documents
    chunks = chunk_documents(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    # Generate embeddings
    embedding_client = EmbeddingClient(
        api_key=config.openai_api_key,
        model=config.embedding_model,
    )
    chunks_with_embeddings = embedding_client.embed_chunks(chunks)

    # Upsert to vector store
    vector_store = VectorStore(
        api_key=config.pinecone_api_key,
        index_name=config.pinecone_index_name,
        dimension=config.embedding_dimension,
    )
    count = vector_store.upsert_chunks(chunks_with_embeddings)

    print(f"Indexed {count} chunks from {len(documents)} documents")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Query the RAG system."""
    config = load_config()
    setup_logging(config.log_level)

    query = args.query
    logger.info(f"Processing query: {query}")

    # Generate query embedding
    embedding_client = EmbeddingClient(
        api_key=config.openai_api_key,
        model=config.embedding_model,
    )
    query_embedding = embedding_client.get_embedding(query)

    # Query vector store
    vector_store = VectorStore(
        api_key=config.pinecone_api_key,
        index_name=config.pinecone_index_name,
        dimension=config.embedding_dimension,
    )
    matches = vector_store.query(query_embedding, top_k=config.top_k)

    if not matches:
        print("No relevant documents found.")
        return 0

    # Generate response
    generator = Generator(
        api_key=config.openai_api_key,
        model=config.chat_model,
    )
    response = generator.generate(query, matches)

    print("\n" + "=" * 50)
    print("ANSWER:")
    print("=" * 50)
    print(response)
    print("\n" + "-" * 50)
    print("SOURCES:")
    print("-" * 50)
    for match in matches:
        print(f"  - {match['source_file']} (score: {match['score']:.3f})")

    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete vectors from the index."""
    config = load_config()
    setup_logging(config.log_level)

    vector_store = VectorStore(
        api_key=config.pinecone_api_key,
        index_name=config.pinecone_index_name,
        dimension=config.embedding_dimension,
    )

    if args.all:
        logger.info("Deleting all vectors from index")
        vector_store.delete_all()
        print("Deleted all vectors from index")
    elif args.file:
        logger.info(f"Deleting vectors for file: {args.file}")
        vector_store.delete_by_source(args.file)
        print(f"Deleted vectors for: {args.file}")
    else:
        print("Please specify --all or --file")
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="rag_system",
        description="RAG System - Retrieval-Augmented Generation with Pinecone and OpenAI",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    index_parser = subparsers.add_parser(
        "index", help="Index documents from a directory"
    )
    index_parser.add_argument(
        "--dir", "-d", required=True, help="Directory containing documents to index"
    )
    index_parser.set_defaults(func=cmd_index)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query", help="The question to ask")
    query_parser.set_defaults(func=cmd_query)

    # Delete command
    delete_parser = subparsers.add_parser(
        "delete", help="Delete vectors from the index"
    )
    delete_parser.add_argument("--all", action="store_true", help="Delete all vectors")
    delete_parser.add_argument("--file", "-f", help="Delete vectors for a specific file")
    delete_parser.set_defaults(func=cmd_delete)

    args = parser.parse_args(argv)

    # Set log level from CLI arg
    if args.log_level:
        import os
        os.environ["LOG_LEVEL"] = args.log_level

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
