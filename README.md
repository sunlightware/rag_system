# RAG System

A Python Retrieval-Augmented Generation (RAG) system that indexes documents into Pinecone and generates responses using GPT-4.

## Features

- **Multi-format document support**: txt, md, pdf, docx
- **Vector storage**: Pinecone with cosine similarity
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **Generation**: GPT-4 with retrieved context
- **CLI interface**: Simple commands for indexing, querying, and managing vectors

## Project Structure

```
rag_system/
├── __init__.py          # Package initialization
├── config.py            # Configuration and environment loading
├── document_loader.py   # Multi-format document parsing
├── chunker.py           # Fixed-size text chunking with overlap
├── embeddings.py        # OpenAI embedding operations
├── vector_store.py      # Pinecone vector database operations
├── generator.py         # GPT-4 response generation
├── cli.py               # Command-line interface
├── main.py              # Entry point
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── tests/               # Test suite
    ├── test_chunker.py
    └── test_rag.py
```

## Installation

1. Clone or copy the `rag_system` directory

2. Install dependencies:
   ```bash
   pip install -r rag_system/requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp rag_system/.env.example .env
   ```

4. Edit `.env` with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_INDEX_NAME=rag-index
   ```

## Usage

### Index Documents

Index all supported documents from a directory:

```bash
python -m rag_system.cli index --dir ./documents
```

### Query

Ask a question and get a response based on indexed documents:

```bash
python -m rag_system.cli query "What is machine learning?"
```

### Delete Vectors

Delete all vectors from the index:

```bash
python -m rag_system.cli delete --all
```

Delete vectors for a specific file:

```bash
python -m rag_system.cli delete --file document.pdf
```

### Logging

Set log level for debugging:

```bash
python -m rag_system.cli --log-level DEBUG index --dir ./documents
```

## Configuration

Default settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `chunk_size` | 1000 | Characters per chunk |
| `chunk_overlap` | 200 | Overlap between chunks |
| `top_k` | 3 | Number of chunks to retrieve |
| `embedding_model` | text-embedding-3-small | OpenAI embedding model |
| `chat_model` | gpt-4 | OpenAI chat model |

## How It Works

1. **Indexing**: Documents are loaded, split into overlapping chunks, embedded using OpenAI, and stored in Pinecone with metadata.

2. **Querying**: The query is embedded, similar chunks are retrieved from Pinecone, and GPT-4 generates a response using the retrieved context.

## Dependencies

- openai >= 1.0.0
- pinecone >= 3.0.0
- python-dotenv >= 1.0.0
- PyPDF2 >= 3.0.0
- python-docx >= 0.8.11
- pytest >= 7.0.0

## License

MIT
