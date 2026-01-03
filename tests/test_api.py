"""
Integration tests for the API endpoints.
"""
import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Set a dummy API key for testing
os.environ['OPENAI_API_KEY'] = 'test-key-123'

from src.rag_system.api import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_pdf():
    """Create a sample PDF for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as tmp:
        c = canvas.Canvas(tmp.name, pagesize=letter)
        c.drawString(100, 750, "Artificial Intelligence Overview")
        c.drawString(100, 730, "AI is the simulation of human intelligence by machines.")
        c.drawString(100, 710, "Machine learning is a key component of AI.")
        c.save()
        yield tmp.name
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "endpoints" in data


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_stats_endpoint(client):
    """Test stats endpoint."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "embedding_dimension" in data
    assert "embedding_model" in data


def test_upload_endpoint(client, sample_pdf):
    """Test document upload endpoint."""
    with open(sample_pdf, 'rb') as f:
        response = client.post(
            "/upload",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_processed"] > 0


def test_upload_invalid_file_type(client):
    """Test upload with invalid file type."""
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"content", "text/plain")}
    )
    assert response.status_code == 400


def test_query_endpoint_without_documents(client):
    """Test query endpoint without documents."""
    # Clear documents first
    client.delete("/documents")
    
    response = client.post(
        "/query",
        json={"question": "What is AI?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["retrieved_chunks"] == 0


def test_query_endpoint_with_documents(client, sample_pdf):
    """Test query endpoint with uploaded documents."""
    # Upload document first
    with open(sample_pdf, 'rb') as f:
        client.post(
            "/upload",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    
    # Query
    response = client.post(
        "/query",
        json={"question": "What is AI?", "temperature": 0.5}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["retrieved_chunks"] > 0


def test_query_empty_question(client):
    """Test query with empty question."""
    response = client.post(
        "/query",
        json={"question": ""}
    )
    assert response.status_code == 400


def test_clear_documents(client, sample_pdf):
    """Test clearing documents."""
    # Upload document
    with open(sample_pdf, 'rb') as f:
        client.post(
            "/upload",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    
    # Clear
    response = client.delete("/documents")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Verify cleared
    stats = client.get("/stats").json()
    assert stats["total_documents"] == 0


def test_query_with_custom_params(client, sample_pdf):
    """Test query with custom parameters."""
    # Upload document first
    with open(sample_pdf, 'rb') as f:
        client.post(
            "/upload",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    
    response = client.post(
        "/query",
        json={
            "question": "What is machine learning?",
            "top_k": 3,
            "temperature": 0.3
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) <= 3
