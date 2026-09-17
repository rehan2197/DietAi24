"""
Central configuration for the DietAI24 Nutrition Estimator backend.
All values are overridable via environment variables or a `.env` file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Application ---
    app_name: str = "DietAI24 Nutrition Estimator"
    api_v1_prefix: str = "/api/v1"
    debug: bool = True

    # --- Database (SQLite for local dev / PostgreSQL for production) ---
    database_url: str = "sqlite:///./nutrition.db"

    # --- Vector DB (Chroma) ---
    chroma_persist_dir: str = "./chroma_store"
    chroma_collection_name: str = "ifct_foods"

    # --- Multimodal Vision & Generation (Gemini / GPT) ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "text-embedding-004"

    # --- RAG Retrieval Settings ---
    retrieval_top_k: int = 5
    retrieval_multiquery_count: int = 3
    low_confidence_threshold: float = 0.55
    use_cloud_embeddings: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
