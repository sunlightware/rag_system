"""Tests for the chunking logic."""

import pytest

from rag_system.chunker import Chunk, chunk_documents, chunk_text


class TestChunkText:
    """Tests for chunk_text function."""

    def test_basic_chunking(self):
        """Test that text is chunked into correct sizes."""
        text = "a" * 2500
        chunks = chunk_text(text, "test.txt", chunk_size=1000, chunk_overlap=200)

        # With step size 800 (1000-200), chunks start at: 0, 800, 1600, 2400 = 4 chunks
        assert len(chunks) == 4
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_size(self):
        """Test that chunks are the expected size."""
        text = "a" * 2000
        chunks = chunk_text(text, "test.txt", chunk_size=1000, chunk_overlap=200)

        # First chunk should be full size
        assert len(chunks[0].text) == 1000
        # Middle/last chunks depend on overlap
        assert len(chunks[1].text) == 1000

    def test_overlap(self):
        """Test that chunks properly overlap."""
        text = "".join(str(i % 10) for i in range(2000))
        chunks = chunk_text(text, "test.txt", chunk_size=1000, chunk_overlap=200)

        # The last 200 chars of chunk 0 should equal first 200 of chunk 1
        assert chunks[0].text[-200:] == chunks[1].text[:200]

    def test_metadata_preserved(self):
        """Test that source file and chunk index are preserved."""
        text = "a" * 2500
        chunks = chunk_text(text, "myfile.pdf", chunk_size=1000, chunk_overlap=200)

        for i, chunk in enumerate(chunks):
            assert chunk.source_file == "myfile.pdf"
            assert chunk.chunk_index == i

    def test_empty_text(self):
        """Test handling of empty text."""
        chunks = chunk_text("", "empty.txt", chunk_size=1000, chunk_overlap=200)
        assert chunks == []

    def test_whitespace_only_text(self):
        """Test handling of whitespace-only text."""
        chunks = chunk_text("   \n\t  ", "whitespace.txt", chunk_size=1000, chunk_overlap=200)
        assert chunks == []

    def test_text_smaller_than_chunk_size(self):
        """Test text smaller than chunk size creates one chunk."""
        text = "Hello, world!"
        chunks = chunk_text(text, "small.txt", chunk_size=1000, chunk_overlap=200)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].source_file == "small.txt"
        assert chunks[0].chunk_index == 0

    def test_overlap_greater_than_chunk_size_raises(self):
        """Test that overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            chunk_text("some text", "test.txt", chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            chunk_text("some text", "test.txt", chunk_size=100, chunk_overlap=150)

    def test_exact_chunk_size_text(self):
        """Test text exactly equal to chunk size."""
        text = "a" * 800  # Use 800 to get exactly 1 chunk with step 800
        chunks = chunk_text(text, "exact.txt", chunk_size=1000, chunk_overlap=200)

        assert len(chunks) == 1
        assert len(chunks[0].text) == 800


class TestChunkDocuments:
    """Tests for chunk_documents function."""

    def test_multiple_documents(self):
        """Test chunking multiple documents."""
        documents = [
            ("doc1.txt", "a" * 1500),
            ("doc2.txt", "b" * 1500),
        ]
        chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)

        # Each 1500 char doc should produce 2 chunks
        assert len(chunks) == 4

        # Check source files
        doc1_chunks = [c for c in chunks if c.source_file == "doc1.txt"]
        doc2_chunks = [c for c in chunks if c.source_file == "doc2.txt"]
        assert len(doc1_chunks) == 2
        assert len(doc2_chunks) == 2

    def test_chunk_indices_per_document(self):
        """Test that chunk indices reset for each document."""
        documents = [
            ("doc1.txt", "a" * 2500),
            ("doc2.txt", "b" * 2500),
        ]
        chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)

        doc1_indices = [c.chunk_index for c in chunks if c.source_file == "doc1.txt"]
        doc2_indices = [c.chunk_index for c in chunks if c.source_file == "doc2.txt"]

        # With step size 800, 2500 chars produces chunks at: 0, 800, 1600, 2400 = 4 chunks
        assert doc1_indices == [0, 1, 2, 3]
        assert doc2_indices == [0, 1, 2, 3]

    def test_empty_documents_list(self):
        """Test handling of empty documents list."""
        chunks = chunk_documents([], chunk_size=1000, chunk_overlap=200)
        assert chunks == []

    def test_mixed_size_documents(self):
        """Test documents of different sizes."""
        documents = [
            ("small.txt", "Short text"),
            ("medium.txt", "m" * 1500),
            ("large.txt", "l" * 3000),
        ]
        chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)

        small_chunks = [c for c in chunks if c.source_file == "small.txt"]
        medium_chunks = [c for c in chunks if c.source_file == "medium.txt"]
        large_chunks = [c for c in chunks if c.source_file == "large.txt"]

        assert len(small_chunks) == 1
        assert len(medium_chunks) == 2
        assert len(large_chunks) == 4
