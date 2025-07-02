"""
README: Hybrid RAG - Data Ingestion and Retrieval
=================================================

This project demonstrates a Hybrid Retrieval-Augmented Generation (RAG) pipeline with two main components:

1. **Data Ingestion**
   - Supports PDF, DOCX, TXT, and image files (with OCR).
   - Uses `document_loader.py` to extract text from documents.
   - Uses `chunker.py` to split text into manageable chunks for retrieval.
   - OCR for images is handled by `ocr.py` using Tesseract.

2. **Retrieval**
   - **Keyword Index**: `keyword_index.py` uses Weaviate's BM25 for fast keyword-based search.
   - **Vector Store**: `vector_store.py` uses dense embeddings (HuggingFace) and Weaviate for semantic search.
   - **Hybrid Retrieval**: `hybrid.py` combines BM25 and vector search for improved results.

**How to Explain to Team Lead:**
- The ingestion pipeline loads and preprocesses documents, including OCR for images.
- The retrieval system supports both keyword (BM25) and semantic (vector) search, and can combine both for hybrid retrieval.
- All retrieval is powered by Weaviate, with schema and class management handled in `vector_store.py`.
- The FastAPI app (`main.py`) provides an entry point for serving the system as an API.

**Key Files:**
- `ingestion/document_loader.py`: Loads and extracts text from various file types.
- `ingestion/chunker.py`: Splits text into chunks for retrieval.
- `ingestion/ocr.py`: OCR for images.
- `retrieval/keyword_index.py`: BM25 keyword search.
- `retrieval/vector_store.py`: Dense vector search with embeddings.
- `retrieval/hybrid.py`: Combines both retrieval methods.
- `main.py`: FastAPI app entry point.

**Typical Flow:**
1. Ingest documents (PDF, DOCX, TXT, images).
2. Extract and chunk text.
3. Store in Weaviate with both keyword and vector indices.
4. Query using keyword, vector, or hybrid search.

---
This summary can be shared with your team lead to explain the ingestion and retrieval architecture.
