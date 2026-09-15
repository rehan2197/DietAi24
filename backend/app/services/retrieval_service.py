"""
RAG retrieval layer: given a free-text food description (from the vision
service), finds the closest-matching food entries in the vector index built
from the IFCT dataset. This is the "look it up, don't guess" step —
retrieval returns candidates from real data; it never invents a food.
"""
from __future__ import annotations

import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb

from app.config import get_settings
from app.services.embedding import TextEmbedder

settings = get_settings()

_EMBEDDER_PATH = "./chroma_store/embedder.pkl"


class RetrievalService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._embedder: TextEmbedder | None = None

    @property
    def collection(self):
        # Always fetched fresh (not cached on self) — index_foods() may delete and
        # recreate the underlying collection (e.g. during re-ingest), which would
        # otherwise leave long-lived instances (like the router-level singleton)
        # holding a stale, deleted collection reference.
        return self.client.get_or_create_collection(
            name=settings.chroma_collection_name, metadata={"hnsw:space": "cosine"}
        )

    @property
    def embedder(self) -> TextEmbedder:
        if self._embedder is None:
            self._embedder = TextEmbedder.load(_EMBEDDER_PATH)
        return self._embedder

    def is_indexed(self) -> bool:
        return self.collection.count() > 0

    def index_foods(self, foods: list[dict]) -> None:
        """
        Build the vector index from scratch. `foods` is a list of dicts with
        at least: food_code, food_name, region_variant, description, food_group.
        """
        corpus = [
            f"{f['food_name']} {f['region_variant']} {f['food_group']} {f.get('description', '')}"
            for f in foods
        ]
        embedder = TextEmbedder(n_components=32)
        vectors = embedder.fit_transform(corpus)
        embedder.save(_EMBEDDER_PATH)
        self._embedder = embedder

        # Reset collection to avoid duplicate/stale entries on re-index.
        # `self.collection` is fetched fresh each access (see property above),
        # so recreating it here is immediately visible to every other holder
        # of this service, including ones created before this call.
        try:
            self.client.delete_collection(settings.chroma_collection_name)
        except Exception:
            pass  # fine if it didn't exist yet
        self.collection.add(
            ids=[f["food_code"] for f in foods],
            embeddings=vectors,
            documents=corpus,
            metadatas=[
                {"food_name": f["food_name"], "region_variant": f["region_variant"], "food_group": f["food_group"]}
                for f in foods
            ],
        )

    def search(self, query_text: str, top_k: int | None = None) -> list[dict]:
        """
        Returns candidates ranked by similarity, each as:
        {food_code, food_name, region_variant, food_group, similarity}
        similarity is in [0, 1], higher is better (converted from cosine distance).
        """
        top_k = top_k or settings.retrieval_top_k
        query_vec = self.embedder.transform([query_text])
        results = self.collection.query(query_embeddings=query_vec, n_results=top_k)

        candidates = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        for food_code, distance, meta in zip(ids, distances, metadatas):
            similarity = max(0.0, 1.0 - distance / 2.0)  # cosine distance -> rough similarity
            candidates.append(
                {
                    "food_code": food_code,
                    "food_name": meta["food_name"],
                    "region_variant": meta["region_variant"],
                    "food_group": meta["food_group"],
                    "similarity": round(similarity, 4),
                }
            )
        return candidates
