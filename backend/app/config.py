"""
Central configuration for the Indian Food Nutrition Estimator backend.
All values are overridable via environment variables or a `.env` file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "Indian Food Nutrition Estimator"
    api_v1_prefix: str = "/api/v1"
    debug: bool = True

    # --- Database (PostgreSQL in production; defaults to local SQLite for easy dev/demo) ---
    # Example production value:
    #   postgresql+psycopg2://user:password@localhost:5432/food_nutrition
    database_url: str = "sqlite:///./nutrition.db"

    # --- Vector DB (Chroma) ---
    chroma_persist_dir: str = "./chroma_store"
    chroma_collection_name: str = "ifct_foods"

    # --- Gemini (Vision AI) ---
    # If left empty, the vision service runs in MOCK MODE (returns a stub recognition)
    # so the rest of the pipeline is fully testable without a live API key.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # --- Retrieval ---
    retrieval_top_k: int = 5
    low_confidence_threshold: float = 0.55


@lru_cache
def get_settings() -> Settings:
    return Settings()
