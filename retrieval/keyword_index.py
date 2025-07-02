# keyword_index.py
"""
Keyword search piggybacks on Weaviate's BM25,
so we only need a small helper to query it.
"""
from langchain_core.documents import Document
from app.core.config import settings
from .vector_store import _client

def bm25_search(query: str, k: int = 10) -> list[Document]:
    client = _client()
    res = (
        client.query
        .get(settings.WEAVIATE_CLASS, ["text", "doc_id"])
        .with_bm25(query, properties=["text"])
        .with_limit(k)
        .do()
    )
    return [
        Document(page_content=hit["text"], metadata={"doc_id": hit["doc_id"]})
        for hit in res["data"]["Get"][settings.WEAVIATE_CLASS]
    ]
