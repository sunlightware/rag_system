# Quick Start Guide

Get the RAG system running in 5 minutes.

## 1. Install Dependencies

```bash
pip install -r rag_system/requirements.txt
```

## 2. Configure API Keys

Create a `.env` file in your working directory:

```bash
OPENAI_API_KEY=sk-your-openai-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=rag-index
```

## 3. Prepare Documents

Create a directory with your documents:

```bash
mkdir documents
# Add .txt, .md, .pdf, or .docx files to this directory
```

## 4. Index Documents

```bash
python -m rag_system.cli index --dir ./documents
```

Expected output:
```
Indexed 15 chunks from 3 documents
```

## 5. Query

```bash
python -m rag_system.cli query "Your question here"
```

Expected output:
```
==================================================
ANSWER:
==================================================
[Generated response based on your documents]

--------------------------------------------------
SOURCES:
--------------------------------------------------
  - document1.pdf (score: 0.892)
  - document2.txt (score: 0.845)
  - document1.pdf (score: 0.823)
```

## Common Issues

### "OPENAI_API_KEY environment variable is required"

Make sure your `.env` file is in the directory where you run the command, or export the variables:

```bash
export OPENAI_API_KEY=sk-your-key
export PINECONE_API_KEY=your-key
```

### "No documents found to index"

Check that your directory contains supported file types (.txt, .md, .pdf, .docx).

### "No relevant documents found"

The index may be empty. Run the index command first.

## Next Steps

- See [README.md](README.md) for full documentation
- See [TESTING.md](TESTING.md) for running tests
