# Imperial Physics Notes RAG

A Retrieval-Augmented Generation (RAG) project built over my Imperial College BSc Physics lecture notes.

## Features
- PDF text extraction and cleaning
- Chunking with metadata
- Embedding-based retrieval using SentenceTransformers
- Lexical subject routing
- Optional year-level filtering
- LLM answer generation with Llama 3 via Hugging Face Inference API
- Streamlit interface

## Main files
- `clean_text.py` – cleans extracted PDF text
- `chunk_text.py` – creates chunks with metadata
- `embed_chunks.py` – builds embeddings
- `search_chunks.py` – retrieval-only testing
- `rag_answer.py` – retrieval + generation
- `app.py` – Streamlit UI

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN="your_token_here"