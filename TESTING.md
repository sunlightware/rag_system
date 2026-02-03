# Testing Guide

## Running Tests

Run all tests from the parent directory:

```bash
python -m pytest rag_system/tests/ -v
```

Run with coverage (requires pytest-cov):

```bash
pip install pytest-cov
python -m pytest rag_system/tests/ --cov=rag_system --cov-report=term-missing
```

## Test Structure

```
tests/
├── __init__.py
├── test_chunker.py    # Unit tests for chunking logic
└── test_rag.py        # Integration tests with mocked APIs
```

### test_chunker.py

Tests the text chunking functionality:

- `TestChunkText`: Tests for single text chunking
  - Basic chunking behavior
  - Chunk size validation
  - Overlap verification
  - Metadata preservation
  - Edge cases (empty text, whitespace, exact chunk size)
  - Error handling (invalid overlap)

- `TestChunkDocuments`: Tests for multiple document chunking
  - Multiple documents processing
  - Chunk index reset per document
  - Mixed document sizes

### test_rag.py

Integration tests with mocked OpenAI and Pinecone APIs:

- `TestEmbeddingClient`: Tests OpenAI embedding operations
  - Single text embedding
  - Batch embedding
  - Chunk embedding

- `TestVectorStore`: Tests Pinecone operations
  - Upserting chunks
  - Querying similar chunks
  - Deleting by source file
  - Deleting all vectors

- `TestGenerator`: Tests GPT-4 response generation
  - Response generation with context

- `TestRAGPipeline`: End-to-end pipeline test
  - Full flow from query to response

## Writing New Tests

### Adding a Chunker Test

```python
def test_my_new_case(self):
    """Description of what this tests."""
    text = "sample text"
    chunks = chunk_text(text, "test.txt", chunk_size=100, chunk_overlap=20)

    assert len(chunks) == expected_count
    assert chunks[0].text == expected_text
```

### Adding a Mocked API Test

```python
from unittest.mock import MagicMock, patch

@patch("rag_system.embeddings.OpenAI")
def test_my_api_call(self, mock_openai_class):
    # Setup mock
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2])]
    mock_client.embeddings.create.return_value = mock_response

    # Run test
    client = EmbeddingClient(api_key="test-key")
    result = client.get_embedding("test")

    # Assert
    assert result == [0.1, 0.2]
    mock_client.embeddings.create.assert_called_once()
```

## Test Configuration

Tests use mocked APIs by default. No real API calls are made during testing.

To run integration tests against real APIs (not recommended for CI):

```bash
# Set environment variables
export OPENAI_API_KEY=your-real-key
export PINECONE_API_KEY=your-real-key

# Create a separate test file for live tests
```

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r rag_system/requirements.txt
      - run: python -m pytest rag_system/tests/ -v
```
