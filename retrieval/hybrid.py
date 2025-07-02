# hybrid.py
"""
Hybrid retrieval = BM25 + dense similarity.
Weaviate supports *native* hybrid queries, but we can combine
scores manually for clarity.
"""
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
import numpy as np
from .vector_store import get_vector_store
from .keyword_index import bm25_search

def hybrid_search(query: str, k_dense: int = 10, k_bm25: int = 10, top_k: int = 8) -> list[Document]:
    vs = get_vector_store()  # Remove type hint since WeaviateVectorStore inherits from VectorStore
    dense_docs = vs.similarity_search(query, k_dense)
    bm25_docs = bm25_search(query, k_bm25)

    # --- naive re‑rank: combine lists, favour docs that appear in both --- #
    scored: dict[str, tuple[Document, float]] = {}
    for rank, doc in enumerate(dense_docs):
        scored[doc.page_content] = (doc, 1.0 / (rank + 1))

    for rank, doc in enumerate(bm25_docs):
        if doc.page_content in scored:
            scored[doc.page_content] = (
                doc,
                scored[doc.page_content][1] + 1.0 / (rank + 1),
            )
        else:
            scored[doc.page_content] = (doc, 0.5 / (rank + 1))

    # sort by combined score
    sorted_docs = sorted(scored.values(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in sorted_docs[:top_k]]
