"""
Local, offline-capable text embedder used to populate and query the Chroma
vector store.

Why not call an embedding API directly here?
Using TF-IDF + SVD (LSA) means the retrieval layer works immediately with
zero external dependencies or API keys — useful for development, testing,
and offline demos. It is a genuine vector-similarity search, just backed by
a lighter-weight model than a hosted embedding API.

Swap-in upgrade path: replace `TextEmbedder` with a wrapper around Gemini's
or OpenAI's embedding endpoint (same `.transform(texts) -> list[list[float]]`
interface) once API access is available — no other code needs to change.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class TextEmbedder:
    def __init__(self, n_components: int = 64):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        self.svd: TruncatedSVD | None = None
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        tfidf = self.vectorizer.fit_transform(corpus)
        # n_components must be < number of samples/features; clamp for small demo corpora
        n_components = max(2, min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf)
        self._fitted = True

    def transform(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted or self.svd is None:
            raise RuntimeError("TextEmbedder must be fit() before transform().")
        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf)
        return dense.tolist()

    def fit_transform(self, corpus: list[str]) -> list[list[float]]:
        self.fit(corpus)
        return self.transform(corpus)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd, "n_components": self.n_components}, f)

    @classmethod
    def load(cls, path: str) -> "TextEmbedder":
        with open(path, "rb") as f:
            data = pickle.load(f)
        embedder = cls(n_components=data["n_components"])
        embedder.vectorizer = data["vectorizer"]
        embedder.svd = data["svd"]
        embedder._fitted = True
        return embedder
