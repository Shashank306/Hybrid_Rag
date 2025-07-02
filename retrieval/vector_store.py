# vector_store.py
"""
Weaviate vector store initialisation & accessors.
Uses LangChain's Weaviate wrapper + HuggingFaceEmbeddings
so no paid APIs are required.
"""
import functools
import weaviate
from langchain_community.vectorstores import Weaviate
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

@functools.lru_cache
def _embeddings() -> HuggingFaceEmbeddings:
    """Initialize embeddings with proper error handling"""
    try:
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': False
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 1  # Process one at a time to avoid memory issues
            }
        )
    except Exception as e:
        error(f"Failed to load embedding model: {e}")
        raise

@functools.lru_cache
def _client() -> weaviate.Client:
    """Initialize Weaviate client using v3 API"""
    try:
        client = weaviate.Client(
            url=settings.WEAVIATE_URL,
            auth_client_secret=weaviate.auth.AuthApiKey(settings.WEAVIATE_API_KEY) if settings.WEAVIATE_API_KEY else None,
            additional_headers={"X-OpenAI-Api-Key": "none"},
        )
        return client
    except Exception as e:
        error(f"Failed to connect to Weaviate: {e}")
        raise

@functools.lru_cache
def get_vector_store() -> Weaviate:
    """Initialize vector store with proper error handling"""
    client = _client()
    class_name = settings.WEAVIATE_CLASS
    # Only ensure class exists if needed (skip delete/recreate for now)
    try:
        schema = client.schema.get()
        existing_classes = {cls['class'] for cls in schema.get('classes', [])}
        if class_name not in existing_classes:
            client.schema.create_class({
                "class": class_name,
                "vectorizer": "none",
                "properties": [
                    {"name": "text", "dataType": ["text"], "description": "The text content"},
                    {"name": "doc_id", "dataType": ["text"], "description": "Document ID"},
                ],
                "vectorIndexConfig": {"distance": "cosine"}
            })
    except Exception as schema_error:
        error(f"Schema creation failed: {schema_error}")
        pass
    embeddings = _embeddings()
    return Weaviate(
        client=client,
        index_name=class_name,
        text_key="text",
        embedding=embeddings,
        by_text=False
    )
