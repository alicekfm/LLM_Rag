# Imperial Physics Notes RAG

A Retrieval-Augmented Generation (RAG) system built over my Imperial College London BSc Physics lecture notes.
The project focuses on transforming raw academic material into a structured, searchable knowledge base and generating answers grounded in the source content.

## Overview

The pipeline extracts text from lecture notes, cleans and structures it, splits it into semantically meaningful chunks, and embeds these for retrieval. At query time, the most relevant chunks are selected and passed to a language model to generate context-aware responses.

## Features

* PDF text extraction and cleaning
* Chunking with structured metadata (subject, source, page)
* Embedding-based semantic retrieval (SentenceTransformers)
* Lexical subject routing for improved relevance
* Optional year-level filtering
* LLM-based answer generation (Llama 3 via Hugging Face Inference API)
* Streamlit interface for interactive querying

## Project Structure

* `extract_text.py` – extracts raw text from PDF lecture notes
* `clean_text.py` – cleans and normalises extracted text
* `chunk_text.py` – splits text into chunks and attaches metadata
* `embed_chunks.py` – generates embeddings for retrieval
* `search_chunks.py` – retrieval-only testing
* `rag_answer.py` – retrieval + generation pipeline
* `app.py` – Streamlit interface

## Data

The original lecture notes are not included in this repository, as they are private course materials.

This repository is designed to showcase the RAG pipeline and system architecture.
If desired, a small sample dataset can be added under `data/sample/` for demonstration purposes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN="your_token_here"
```

## Workflow

Run the pipeline in the following order:

```bash
python extract_text.py
python clean_text.py
python chunk_text.py
python embed_chunks.py
```

Test retrieval only:

```bash
python search_chunks.py
```

Run full RAG (retrieval + generation):

```bash
python rag_answer.py
```

Launch the Streamlit interface:

```bash
streamlit run app.py
```

## Notes

* Source documents are excluded from version control
* Generated files (embeddings, intermediate JSON outputs) are not tracked
* The focus of this repository is the pipeline design and implementation

## Future Improvements

* Add vector database integration (e.g. Qdrant / FAISS)
* Improve chunking strategy (adaptive or semantic chunking)
* Enhance retrieval with hybrid search (dense + lexical)
* Expand evaluation and benchmarking of answer quality
