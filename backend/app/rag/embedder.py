"""
Embedder interfaces and implementations for the DietAI24 RAG pipeline.
Provides support for:
1. Local offline embedding via TF-IDF + TruncatedSVD (or local models).
2. Cloud GenAI embedding via Google GenAI / Gemini when an API key is available.
"""
from __future__ import annotations

import abc
import logging
import pickle
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BaseEmbedder(abc.ABC):
    """Abstract base class for all RAG embedders."""

    @abc.abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents into dense vectors."""
        pass

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        return self.embed_texts([query])[0]


class LocalEmbedder(BaseEmbedder):
    """
    Offline-capable vector embedder.
    Uses TF-IDF feature extraction combined with Latent Semantic Analysis (TruncatedSVD)
    to generate dense semantic vectors without external API calls or GPU requirements.
    """

    def __init__(self, n_components: int = 32):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
        self.svd: Optional[TruncatedSVD] = None
        self._fitted = False

    def fit(self, corpus: list[str]) -> "LocalEmbedder":
        from sklearn.decomposition import TruncatedSVD

        tfidf = self.vectorizer.fit_transform(corpus)
        n_comp = max(2, min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf)
        self._fitted = True
        return self

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted or self.svd is None:
            raise RuntimeError("LocalEmbedder must be fit on a corpus before generating embeddings.")
        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf)
        return dense.tolist()

    def fit_transform(self, corpus: list[str]) -> list[list[float]]:
        self.fit(corpus)
        return self.embed_texts(corpus)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "svd": self.svd,
                    "n_components": self.n_components,
                    "_fitted": self._fitted,
                },
                f,
            )
        logger.info(f"LocalEmbedder successfully saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LocalEmbedder":
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = cls(n_components=data.get("n_components", 32))
        inst.vectorizer = data["vectorizer"]
        inst.svd = data["svd"]
        inst._fitted = data.get("_fitted", True)
        return inst


class GenAIEmbedder(BaseEmbedder):
    """
    Cloud-based semantic embedder using Google GenAI (e.g. text-embedding-004).
    Provides higher semantic precision when API credentials are configured.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        self.api_key = api_key
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            result = client.models.embed_content(
                model=self.model,
                contents=texts,
            )
            # Support both single and batch results
            if hasattr(result, "embeddings"):
                return [e.values for e in result.embeddings]
            elif hasattr(result, "embedding"):
                return [result.embedding.values]
            raise ValueError(f"Unexpected response structure from GenAI embedding: {result}")
        except Exception as e:
            logger.error(f"GenAI embedding request failed: {e}")
            raise


def get_default_embedder(
    gemini_api_key: str = "",
    use_cloud_embeddings: bool = False,
    local_embedder_path: str | Path = "./chroma_store/embedder.pkl",
) -> BaseEmbedder:
    """
    Factory function returning the configured embedder.
    Falls back to LocalEmbedder if no cloud key is provided or if file exists.
    """
    path = Path(local_embedder_path)
    if use_cloud_embeddings and gemini_api_key:
        return GenAIEmbedder(api_key=gemini_api_key)

    if path.exists():
        try:
            return LocalEmbedder.load(path)
        except Exception as e:
            logger.warning(f"Could not load saved LocalEmbedder from {path}: {e}")

    # Return a fresh un-fitted LocalEmbedder
    return LocalEmbedder()
