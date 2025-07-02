# config.py
from pathlib import Path
from pydantic import Field, validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # --------------------------------------------------------------------- #
    #   App                                                               #
    # --------------------------------------------------------------------- #
    APP_NAME: str = "Hybrid RAG"
    APP_VERSION: str = "1.0.0"  # Version of the app
    CHUNK_SIZE: int = 768
    CHUNK_OVERLAP: int = 64

    # --------------------------------------------------------------------- #
    #   Vector store / embeddings                                          #
    # --------------------------------------------------------------------- #
    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_API_KEY: str | None = None
    WEAVIATE_CLASS: str = "DocumentChunk"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Smaller, faster model

    # --------------------------------------------------------------------- #
    #   LLM                                                                #
    # --------------------------------------------------------------------- #
    OLLAMA_MODEL: str = "llama3.2"


    # class Config:
    #     env_file = ".env"
    #     case_sensitive = True


settings = Settings()
