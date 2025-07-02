from fastapi import FastAPI, HTTPException
# from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
from ingestion.chunker import chunk_text
from ingestion.document_loader import extract_text

app = FastAPI(title="Hybrid RAG API", version="1.0.0")

# --- Retrieval Endpoints ---
from retrieval.keyword_index import bm25_search
from retrieval.hybrid import hybrid_search

# Endpoint to test BM25 keyword search
@app.post("/bm25-search")
def bm25_search_endpoint(query: str, k: int = 10):
    """Run BM25 keyword search using Weaviate."""
    try:
        docs = bm25_search(query, k)
        return {"results": [d.page_content for d in docs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to test hybrid search (BM25 + dense)
@app.post("/hybrid-search")
def hybrid_search_endpoint(query: str, k_dense: int = 10, k_bm25: int = 10, top_k: int = 8):
    """Run hybrid search (BM25 + dense similarity) using Weaviate."""
    try:
        docs = hybrid_search(query, k_dense, k_bm25, top_k)
        return {"results": [d.page_content for d in docs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# main.py
"""
Entry point for running the FastAPI app for Hybrid RAG.
"""





# Endpoint to extract and chunk any supported file (pdf, docx, txt, image)
@app.post("/chunk-file")
def chunk_file(filepath: str):
    """Extract text from a file (pdf, docx, txt, image) and return its chunks."""
    try:
        text = extract_text(Path(filepath))
        chunks = chunk_text(text)
        return {"chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Import your API routers here (example: from .api import router as api_router)
# from app.api import router as api_router



# CORS middleware (adjust origins as needed)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Include routers here
# app.include_router(api_router)


@app.get("/")
def root():
    return {"message": "Hybrid RAG FastAPI is running!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
