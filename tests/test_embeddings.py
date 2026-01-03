"""
Unit tests for embeddings module.
"""
import pytest
import numpy as np

from src.rag_system.embeddings import EmbeddingGenerator


@pytest.fixture
def embedding_generator():
    """Create an EmbeddingGenerator instance."""
    return EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")


def test_embedding_generator_initialization(embedding_generator):
    """Test EmbeddingGenerator initialization."""
    assert embedding_generator.model is not None
    assert embedding_generator.embedding_dimension > 0


def test_generate_single_embedding(embedding_generator):
    """Test generating embedding for a single text."""
    text = "This is a test sentence."
    embedding = embedding_generator.generate_embedding(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == embedding_generator.embedding_dimension
    assert not np.isnan(embedding).any()


def test_generate_multiple_embeddings(embedding_generator):
    """Test generating embeddings for multiple texts."""
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    embeddings = embedding_generator.generate_embeddings(texts, show_progress=False)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == embedding_generator.embedding_dimension
    assert not np.isnan(embeddings).any()


def test_embedding_similarity(embedding_generator):
    """Test that similar texts have similar embeddings."""
    text1 = "The cat sits on the mat."
    text2 = "A cat is sitting on a mat."
    text3 = "Python programming is fun."
    
    emb1 = embedding_generator.generate_embedding(text1)
    emb2 = embedding_generator.generate_embedding(text2)
    emb3 = embedding_generator.generate_embedding(text3)
    
    # Cosine similarity
    sim_1_2 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    sim_1_3 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
    
    # Similar texts should have higher similarity
    assert sim_1_2 > sim_1_3


def test_get_dimension(embedding_generator):
    """Test getting embedding dimension."""
    dimension = embedding_generator.get_dimension()
    assert isinstance(dimension, int)
    assert dimension > 0
