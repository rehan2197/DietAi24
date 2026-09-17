"""
Retrieval service adapter for the DietAI24 application.
Connects routers and ingestion scripts to the app.rag retrieval engine.
"""
from __future__ import annotations

from typing import Any

from app.rag.retriever import RAGRetriever


class RetrievalService:
    def __init__(self):
        self._retriever = RAGRetriever()

    def is_indexed(self) -> bool:
        return self._retriever.is_indexed()

    def index_foods(self, foods: list[dict[str, Any]]) -> None:
        self._retriever.index_foods(foods)

    def search(self, query_text: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self._retriever.search(query_text, top_k=top_k)
