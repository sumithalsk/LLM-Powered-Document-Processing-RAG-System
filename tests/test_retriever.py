"""
Unit tests for retriever module.
"""
import pytest
import numpy as np

from src.rag_system.embeddings import EmbeddingGenerator
from src.rag_system.vector_store import VectorStore
from src.rag_system.retriever import Retriever


@pytest.fixture
def embedding_generator():
    """Create an EmbeddingGenerator instance."""
    return EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def vector_store(embedding_generator):
    """Create a VectorStore with sample data."""
    dimension = embedding_generator.get_dimension()
    store = VectorStore(dimension=dimension, index_type="FlatIP")
    
    # Add sample documents
    texts = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks.",
        "Natural language processing handles text data.",
        "Computer vision processes images."
    ]
    
    embeddings = embedding_generator.generate_embeddings(texts, show_progress=False)
    documents = [
        {'id': i, 'text': text, 'document': f'doc_{i}'}
        for i, text in enumerate(texts)
    ]
    
    store.add_documents(embeddings, documents)
    return store


@pytest.fixture
def retriever(embedding_generator, vector_store):
    """Create a Retriever instance."""
    return Retriever(
        embedding_generator=embedding_generator,
        vector_store=vector_store,
        top_k=3,
        similarity_threshold=0.5
    )


def test_retriever_initialization(retriever):
    """Test Retriever initialization."""
    assert retriever.top_k == 3
    assert retriever.similarity_threshold == 0.5
    assert retriever.embedding_generator is not None
    assert retriever.vector_store is not None


def test_retrieve(retriever):
    """Test document retrieval."""
    query = "What is machine learning?"
    results = retriever.retrieve(query)
    
    assert isinstance(results, list)
    assert len(results) > 0
    assert all('document' in r for r in results)
    assert all('score' in r for r in results)
    assert all('text' in r for r in results)


def test_retrieve_with_custom_params(retriever):
    """Test retrieval with custom parameters."""
    query = "neural networks"
    results = retriever.retrieve(query, top_k=2, threshold=0.3)
    
    assert len(results) <= 2
    assert all(r['score'] >= 0.3 for r in results)


def test_retrieve_context(retriever):
    """Test context retrieval and combination."""
    query = "What is Python?"
    context = retriever.retrieve_context(query)
    
    assert isinstance(context, str)
    assert len(context) > 0
    assert "Source" in context


def test_retrieve_no_results_with_high_threshold(retriever):
    """Test retrieval with very high threshold returns no results."""
    query = "completely unrelated topic xyz123"
    results = retriever.retrieve(query, threshold=0.99)
    
    # Should return empty or very few results
    assert len(results) <= 1


def test_retrieve_relevance(retriever):
    """Test that retrieved documents are relevant to query."""
    query = "neural networks and deep learning"
    results = retriever.retrieve(query, top_k=2)
    
    # Should retrieve documents about deep learning/neural networks
    assert len(results) > 0
    top_result = results[0]
    assert "neural" in top_result['text'].lower() or "deep" in top_result['text'].lower()
