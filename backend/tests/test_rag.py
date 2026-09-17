"""
Unit and integration tests for the DietAI24 RAG model pipeline:
- Embedder (LocalEmbedder)
- MultiQueryGenerator
- ChromaVectorStore
- RAGRetriever
"""
import pytest
from app.rag.embedder import LocalEmbedder
from app.rag.multiquery import MultiQueryGenerator
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import ChromaVectorStore


def test_local_embedder_fit_and_embed():
    corpus = [
        "Chicken Biryani Hyderabadi Rice Dish spicy layered chicken",
        "Steamed White Rice Polished Cereals boiled rice",
        "Masala Dosa South Indian fermented crepe with potato",
    ]
    embedder = LocalEmbedder(n_components=8)
    vectors = embedder.fit_transform(corpus)
    assert len(vectors) == len(corpus)
    assert len(vectors[0]) == 8

    # Query embedding
    q_vec = embedder.embed_query("biryani with chicken")
    assert len(q_vec) == 8


def test_multiquery_generator_rule_based():
    generator = MultiQueryGenerator()
    queries = generator.generate_queries("a plate of chicken biryani with cucumber raita", count=3)
    assert len(queries) >= 2
    assert "a plate of chicken biryani with cucumber raita" in queries
    # Check that it split components
    assert any("biryani" in q.lower() or "raita" in q.lower() for q in queries)


def test_rag_retriever_search_accuracy():
    retriever = RAGRetriever()
    assert retriever.is_indexed() is True

    # Search for idli
    results = retriever.search("steamed fermented idli cakes", top_k=3)
    assert len(results) > 0
    top = results[0]
    assert top["food_name"] == "Idli"
    assert top["similarity"] > 0.5
    assert len(top["matched_queries"]) > 0


def test_rag_retriever_multi_query_fusion():
    retriever = RAGRetriever()
    results = retriever.search("biryani with boiled egg and chicken", top_k=5)
    assert len(results) > 0
    # Top match should be Chicken Biryani
    top_names = [r["food_name"] for r in results]
    assert "Chicken Biryani" in top_names
