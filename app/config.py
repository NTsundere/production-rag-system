from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Google Gemini
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_documents"

    # Retrieval
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    bm25_weight: float = 0.3
    dense_weight: float = 0.7


settings = Settings()