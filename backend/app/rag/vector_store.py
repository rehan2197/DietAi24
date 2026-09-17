"""
ChromaDB Vector Store integration for DietAI24 RAG.
Stores embeddings and metadata of standardized food database items (IFCT / FNDDS).
"""
from __future__ import annotations

import logging
import os
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    def __init__(self, persist_directory: str = "./chroma_store", collection_name: str = "ifct_foods"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_directory)

    @property
    def collection(self) -> Collection:
        """Always return the active collection, creating it if it does not exist."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def is_indexed(self) -> bool:
        """Returns True if the collection exists and has at least one document indexed."""
        try:
            return self.collection.count() > 0
        except Exception:
            return False

    def count(self) -> int:
        return self.collection.count()

    def reset_collection(self) -> None:
        """Deletes the existing collection to allow clean re-indexing."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}' for clean rebuild.")
        except Exception as e:
            logger.debug(f"Collection delete skipped or not found: {e}")

    def add_food_entries(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add batch of food entries and their embeddings to the vector store."""
        coll = self.collection
        coll.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Successfully added {len(ids)} items to collection '{self.collection_name}'.")

    def query(
        self,
        query_embeddings: list[list[float]],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Query the collection using dense query embeddings."""
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where=where,
        )
