"""
Unit tests for vector store.
"""
import tempfile
import os
import pytest
import numpy as np

from src.rag_system.vector_store import VectorStore


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings."""
    np.random.seed(42)
    return np.random.rand(10, 384).astype('float32')


@pytest.fixture
def sample_documents():
    """Create sample documents."""
    return [
        {'id': i, 'text': f'Document {i}', 'metadata': f'meta_{i}'}
        for i in range(10)
    ]


@pytest.fixture
def vector_store():
    """Create a VectorStore instance."""
    return VectorStore(dimension=384, index_type="FlatIP")


def test_vector_store_initialization():
    """Test VectorStore initialization."""
    store = VectorStore(dimension=384, index_type="FlatL2")
    assert store.dimension == 384
    assert store.index_type == "FlatL2"
    assert store.get_total_documents() == 0


def test_add_documents(vector_store, sample_embeddings, sample_documents):
    """Test adding documents to vector store."""
    vector_store.add_documents(sample_embeddings, sample_documents)
    assert vector_store.get_total_documents() == len(sample_documents)


def test_add_documents_mismatch(vector_store, sample_embeddings, sample_documents):
    """Test error handling for mismatched embeddings and documents."""
    with pytest.raises(ValueError):
        vector_store.add_documents(sample_embeddings[:5], sample_documents)


def test_search(vector_store, sample_embeddings, sample_documents):
    """Test searching for similar documents."""
    vector_store.add_documents(sample_embeddings, sample_documents)
    
    # Search with first embedding
    query = sample_embeddings[0]
    results = vector_store.search(query, top_k=3)
    
    assert len(results) <= 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    
    # First result should be the same document (highest similarity)
    doc, score = results[0]
    assert doc['id'] == 0


def test_save_and_load(vector_store, sample_embeddings, sample_documents):
    """Test saving and loading vector store."""
    # Add documents
    vector_store.add_documents(sample_embeddings, sample_documents)
    
    # Save to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store.save(tmpdir)
        
        # Load from directory
        loaded_store = VectorStore.load(tmpdir)
        
        assert loaded_store.dimension == vector_store.dimension
        assert loaded_store.get_total_documents() == vector_store.get_total_documents()
        
        # Test search on loaded store
        query = sample_embeddings[0]
        results = loaded_store.search(query, top_k=3)
        assert len(results) > 0


def test_clear(vector_store, sample_embeddings, sample_documents):
    """Test clearing the vector store."""
    vector_store.add_documents(sample_embeddings, sample_documents)
    assert vector_store.get_total_documents() > 0
    
    vector_store.clear()
    assert vector_store.get_total_documents() == 0


def test_different_index_types():
    """Test creating vector stores with different index types."""
    dimension = 384
    
    store_l2 = VectorStore(dimension, "FlatL2")
    assert store_l2.index is not None
    
    store_ip = VectorStore(dimension, "FlatIP")
    assert store_ip.index is not None


def test_invalid_index_type():
    """Test error handling for invalid index type."""
    with pytest.raises(ValueError):
        VectorStore(384, "InvalidType")
