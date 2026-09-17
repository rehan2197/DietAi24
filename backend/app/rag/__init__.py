"""
DietAI24 Retrieval-Augmented Generation (RAG) module.
Handles semantic embedding, multi-query expansion, vector storage in ChromaDB,
and candidate retrieval matching authoritative nutrition data.
"""
from app.rag.embedder import BaseEmbedder, LocalEmbedder, GenAIEmbedder, get_default_embedder
from app.rag.multiquery import MultiQueryGenerator
from app.rag.vector_store import ChromaVectorStore
from app.rag.retriever import RAGRetriever

__all__ = [
    "BaseEmbedder",
    "LocalEmbedder",
    "GenAIEmbedder",
    "get_default_embedder",
    "MultiQueryGenerator",
    "ChromaVectorStore",
    "RAGRetriever",
]
