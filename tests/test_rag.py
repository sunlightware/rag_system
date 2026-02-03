"""Integration tests for RAG pipeline with mocked APIs."""

from unittest.mock import MagicMock, patch

import pytest

from rag_system.chunker import Chunk, chunk_text
from rag_system.embeddings import EmbeddingClient
from rag_system.generator import Generator
from rag_system.vector_store import VectorStore


class TestEmbeddingClient:
    """Tests for the embedding client."""

    @patch("rag_system.embeddings.OpenAI")
    def test_get_embedding(self, mock_openai_class):
        """Test single text embedding."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        client = EmbeddingClient(api_key="test-key")
        embedding = client.get_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="test text",
        )

    @patch("rag_system.embeddings.OpenAI")
    def test_get_embeddings_batch(self, mock_openai_class):
        """Test batch embedding."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        client = EmbeddingClient(api_key="test-key")
        embeddings = client.get_embeddings_batch(["text1", "text2"])

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[1] == [0.3, 0.4]

    @patch("rag_system.embeddings.OpenAI")
    def test_embed_chunks(self, mock_openai_class):
        """Test embedding chunks."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        chunks = [
            Chunk(text="chunk1", source_file="test.txt", chunk_index=0),
            Chunk(text="chunk2", source_file="test.txt", chunk_index=1),
        ]

        client = EmbeddingClient(api_key="test-key")
        results = client.embed_chunks(chunks)

        assert len(results) == 2
        assert results[0][0].text == "chunk1"
        assert results[0][1] == [0.1, 0.2]
        assert results[1][0].text == "chunk2"
        assert results[1][1] == [0.3, 0.4]


class TestVectorStore:
    """Tests for the vector store."""

    @patch("rag_system.vector_store.Pinecone")
    def test_upsert_chunks(self, mock_pinecone_class):
        """Test upserting chunks with embeddings."""
        mock_pc = MagicMock()
        mock_pinecone_class.return_value = mock_pc

        mock_index = MagicMock()
        mock_pc.list_indexes.return_value = [MagicMock(name="test-index")]
        mock_pc.Index.return_value = mock_index

        store = VectorStore(api_key="test-key", index_name="test-index")

        chunks_with_embeddings = [
            (Chunk(text="chunk1", source_file="doc.txt", chunk_index=0), [0.1, 0.2]),
            (Chunk(text="chunk2", source_file="doc.txt", chunk_index=1), [0.3, 0.4]),
        ]

        count = store.upsert_chunks(chunks_with_embeddings)

        assert count == 2
        mock_index.upsert.assert_called_once()

        # Check the vectors were formatted correctly
        call_args = mock_index.upsert.call_args
        vectors = call_args.kwargs["vectors"]
        assert len(vectors) == 2
        assert vectors[0]["id"] == "doc.txt_0"
        assert vectors[0]["values"] == [0.1, 0.2]
        assert vectors[0]["metadata"]["text"] == "chunk1"

    @patch("rag_system.vector_store.Pinecone")
    def test_query(self, mock_pinecone_class):
        """Test querying for similar chunks."""
        mock_pc = MagicMock()
        mock_pinecone_class.return_value = mock_pc

        mock_index = MagicMock()
        mock_pc.list_indexes.return_value = [MagicMock(name="test-index")]
        mock_pc.Index.return_value = mock_index

        mock_match = MagicMock()
        mock_match.metadata = {
            "text": "relevant chunk",
            "source_file": "doc.txt",
            "chunk_index": 0,
        }
        mock_match.score = 0.95

        mock_results = MagicMock()
        mock_results.matches = [mock_match]
        mock_index.query.return_value = mock_results

        store = VectorStore(api_key="test-key", index_name="test-index")
        matches = store.query([0.1, 0.2, 0.3], top_k=3)

        assert len(matches) == 1
        assert matches[0]["text"] == "relevant chunk"
        assert matches[0]["source_file"] == "doc.txt"
        assert matches[0]["score"] == 0.95

    @patch("rag_system.vector_store.Pinecone")
    def test_delete_by_source(self, mock_pinecone_class):
        """Test deleting vectors by source file."""
        mock_pc = MagicMock()
        mock_pinecone_class.return_value = mock_pc

        mock_index = MagicMock()
        mock_pc.list_indexes.return_value = [MagicMock(name="test-index")]
        mock_pc.Index.return_value = mock_index

        store = VectorStore(api_key="test-key", index_name="test-index")
        store.delete_by_source("doc.txt")

        mock_index.delete.assert_called_once_with(
            filter={"source_file": {"$eq": "doc.txt"}}
        )

    @patch("rag_system.vector_store.Pinecone")
    def test_delete_all(self, mock_pinecone_class):
        """Test deleting all vectors."""
        mock_pc = MagicMock()
        mock_pinecone_class.return_value = mock_pc

        mock_index = MagicMock()
        mock_pc.list_indexes.return_value = [MagicMock(name="test-index")]
        mock_pc.Index.return_value = mock_index

        store = VectorStore(api_key="test-key", index_name="test-index")
        store.delete_all()

        mock_index.delete.assert_called_once_with(delete_all=True)


class TestGenerator:
    """Tests for the response generator."""

    @patch("rag_system.generator.OpenAI")
    def test_generate(self, mock_openai_class):
        """Test generating a response."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="This is the answer."))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        generator = Generator(api_key="test-key")
        context_chunks = [
            {"text": "Context 1", "source_file": "doc1.txt"},
            {"text": "Context 2", "source_file": "doc2.txt"},
        ]

        response = generator.generate("What is the answer?", context_chunks)

        assert response == "This is the answer."
        mock_client.chat.completions.create.assert_called_once()

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4"
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert "Context 1" in messages[1]["content"]
        assert "Context 2" in messages[1]["content"]


class TestRAGPipeline:
    """Integration tests for the full RAG pipeline."""

    @patch("rag_system.embeddings.OpenAI")
    @patch("rag_system.vector_store.Pinecone")
    @patch("rag_system.generator.OpenAI")
    def test_full_pipeline(
        self, mock_gen_openai, mock_pinecone, mock_embed_openai
    ):
        """Test the complete RAG pipeline from chunks to response."""
        # Setup embedding mock
        embed_client = MagicMock()
        mock_embed_openai.return_value = embed_client

        embed_response = MagicMock()
        embed_response.data = [MagicMock(embedding=[0.1] * 1536)]
        embed_client.embeddings.create.return_value = embed_response

        # Setup Pinecone mock
        mock_pc = MagicMock()
        mock_pinecone.return_value = mock_pc

        mock_index = MagicMock()
        mock_pc.list_indexes.return_value = [MagicMock(name="test-index")]
        mock_pc.Index.return_value = mock_index

        mock_match = MagicMock()
        mock_match.metadata = {
            "text": "Machine learning is a subset of AI.",
            "source_file": "ml_intro.txt",
            "chunk_index": 0,
        }
        mock_match.score = 0.92

        mock_results = MagicMock()
        mock_results.matches = [mock_match]
        mock_index.query.return_value = mock_results

        # Setup generator mock
        gen_client = MagicMock()
        mock_gen_openai.return_value = gen_client

        gen_response = MagicMock()
        gen_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="Machine learning is a branch of artificial intelligence."
                )
            )
        ]
        gen_client.chat.completions.create.return_value = gen_response

        # Run the pipeline
        embedding_client = EmbeddingClient(api_key="test-key")
        vector_store = VectorStore(api_key="test-key", index_name="test-index")
        generator = Generator(api_key="test-key")

        # 1. Get query embedding
        query = "What is machine learning?"
        query_embedding = embedding_client.get_embedding(query)

        # 2. Query vector store
        matches = vector_store.query(query_embedding, top_k=3)

        # 3. Generate response
        response = generator.generate(query, matches)

        assert "machine learning" in response.lower()
        assert len(matches) == 1
        assert matches[0]["source_file"] == "ml_intro.txt"
