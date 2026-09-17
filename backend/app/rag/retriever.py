"""
DietAI24 RAG Retriever implementation.
Orchestrates multi-query expansion, vector embedding, similarity search in ChromaDB,
candidate deduplication, and confidence scoring.
"""
from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.rag.embedder import BaseEmbedder, LocalEmbedder, get_default_embedder
from app.rag.multiquery import MultiQueryGenerator
from app.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        embedder: BaseEmbedder | None = None,
        multi_query_gen: MultiQueryGenerator | None = None,
    ):
        settings = get_settings()
        self.vector_store = vector_store or ChromaVectorStore(
            persist_directory=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
        self.embedder_path = f"{settings.chroma_persist_dir}/embedder.pkl"
        self._embedder = embedder
        self.multi_query_gen = multi_query_gen or MultiQueryGenerator(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )
        self.default_top_k = settings.retrieval_top_k
        self.low_confidence_threshold = settings.low_confidence_threshold

    @property
    def embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            settings = get_settings()
            self._embedder = get_default_embedder(
                gemini_api_key=settings.gemini_api_key,
                use_cloud_embeddings=False,
                local_embedder_path=self.embedder_path,
            )
        return self._embedder

    def is_indexed(self) -> bool:
        """Returns True if the underlying vector store has documents indexed."""
        return self.vector_store.is_indexed()

    def index_foods(self, foods: list[dict[str, Any]]) -> None:
        """
        Builds the vector store index from a list of food records.
        Each record should contain at least: food_code, food_name, region_variant, food_group, description.
        """
        if not foods:
            logger.warning("Empty food list provided for indexing.")
            return

        corpus = [
            f"{f.get('food_name', '')} {f.get('region_variant', '')} {f.get('food_group', '')} {f.get('description', '')}".strip()
            for f in foods
        ]

        # Use and fit local embedder for fast reliable retrieval
        local_embedder = LocalEmbedder(n_components=min(32, len(corpus)))
        vectors = local_embedder.fit_transform(corpus)
        local_embedder.save(self.embedder_path)
        self._embedder = local_embedder

        # Reset existing collection and re-index
        self.vector_store.reset_collection()
        self.vector_store.add_food_entries(
            ids=[str(f["food_code"]) for f in foods],
            embeddings=vectors,
            documents=corpus,
            metadatas=[
                {
                    "food_name": str(f.get("food_name", "")),
                    "region_variant": str(f.get("region_variant", "")),
                    "food_group": str(f.get("food_group", "")),
                    "typical_portion_unit": str(f.get("typical_portion_unit", "plate")),
                    "typical_portion_grams": float(f.get("typical_portion_grams", 100.0)),
                }
                for f in foods
            ],
        )
        logger.info(f"RAGRetriever successfully indexed {len(foods)} foods.")

    def search(
        self,
        food_description: str,
        top_k: int | None = None,
        use_multiquery: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Executes DietAI24 RAG retrieval.
        Expands the food description into multiple queries, embeds and searches the vector store,
        fuses candidate scores, and returns ranked candidates.
        """
        limit = top_k or self.default_top_k
        if not food_description or not food_description.strip():
            return []

        # Step 1: Multi-query generation
        queries = (
            self.multi_query_gen.generate_queries(food_description, count=3)
            if use_multiquery
            else [food_description.strip()]
        )

        candidate_scores: dict[str, dict[str, Any]] = {}

        # Step 2: Search for each query and aggregate results
        for q in queries:
            try:
                q_vec = self.embedder.embed_texts([q])
                results = self.vector_store.query(query_embeddings=q_vec, top_k=limit)
            except Exception as e:
                logger.error(f"Vector search failed for query '{q}': {e}")
                continue

            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            for food_code, distance, meta in zip(ids, distances, metadatas):
                # Cosine distance to similarity: [0, 2] -> [1, 0]
                similarity = max(0.0, 1.0 - (distance / 2.0))

                if food_code not in candidate_scores:
                    candidate_scores[food_code] = {
                        "food_code": food_code,
                        "food_name": meta.get("food_name", ""),
                        "region_variant": meta.get("region_variant", ""),
                        "food_group": meta.get("food_group", ""),
                        "typical_portion_unit": meta.get("typical_portion_unit", "plate"),
                        "typical_portion_grams": meta.get("typical_portion_grams", 100.0),
                        "similarity": similarity,
                        "matched_queries": [q],
                        "hit_count": 1,
                    }
                else:
                    # Multi-query fusion: boost similarity when matched across multiple queries
                    existing = candidate_scores[food_code]
                    existing["hit_count"] += 1
                    existing["matched_queries"].append(q)
                    # Slightly boost items found by multiple query formulations
                    boosted = max(existing["similarity"], similarity) * 1.05
                    existing["similarity"] = min(1.0, boosted)

        # Step 3: Sort by similarity and filter
        candidates = sorted(candidate_scores.values(), key=lambda x: x["similarity"], reverse=True)

        for c in candidates:
            c["similarity"] = round(c["similarity"], 4)

        return candidates[:limit]
