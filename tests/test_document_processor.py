"""
Unit tests for document processor.
"""
import os
import tempfile
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from src.rag_system.document_processor import DocumentProcessor


@pytest.fixture
def sample_pdf():
    """Create a sample PDF for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        c = canvas.Canvas(tmp.name, pagesize=letter)
        c.drawString(100, 750, "This is a test document.")
        c.drawString(100, 730, "It contains multiple sentences for testing.")
        c.drawString(100, 710, "We will use this to test PDF extraction.")
        c.save()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def doc_processor():
    """Create a DocumentProcessor instance."""
    return DocumentProcessor(chunk_size=100, chunk_overlap=20)


def test_document_processor_initialization():
    """Test DocumentProcessor initialization."""
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    assert processor.chunk_size == 512
    assert processor.chunk_overlap == 50


def test_extract_text_pypdf2(sample_pdf, doc_processor):
    """Test text extraction with PyPDF2."""
    text = doc_processor.extract_text_from_pdf(sample_pdf, method="pypdf2")
    assert isinstance(text, str)
    assert len(text) > 0
    assert "test document" in text.lower()


def test_extract_text_pdfplumber(sample_pdf, doc_processor):
    """Test text extraction with pdfplumber."""
    text = doc_processor.extract_text_from_pdf(sample_pdf, method="pdfplumber")
    assert isinstance(text, str)
    assert len(text) > 0
    assert "test document" in text.lower()


def test_extract_text_invalid_method(sample_pdf, doc_processor):
    """Test error handling for invalid extraction method."""
    with pytest.raises(ValueError):
        doc_processor.extract_text_from_pdf(sample_pdf, method="invalid")


def test_extract_text_missing_file(doc_processor):
    """Test error handling for missing file."""
    with pytest.raises(FileNotFoundError):
        doc_processor.extract_text_from_pdf("nonexistent.pdf")


def test_preprocess_text(doc_processor):
    """Test text preprocessing."""
    raw_text = "This  is   a    test.\n\n\nWith multiple    spaces."
    processed = doc_processor.preprocess_text(raw_text)
    assert "  " not in processed
    assert "\n\n\n" not in processed


def test_chunk_text(doc_processor):
    """Test text chunking."""
    text = "This is a test sentence. " * 20
    chunks = doc_processor.chunk_text(text)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all('text' in chunk for chunk in chunks)
    assert all('id' in chunk for chunk in chunks)
    assert all(len(chunk['text']) <= doc_processor.chunk_size for chunk in chunks)


def test_chunk_text_with_overlap(doc_processor):
    """Test that chunks have proper overlap."""
    text = "Word " * 50
    chunks = doc_processor.chunk_text(text)
    
    if len(chunks) > 1:
        # Check that there's some overlap between consecutive chunks
        assert chunks[0]['end_pos'] > chunks[1]['start_pos']


def test_process_document(sample_pdf, doc_processor):
    """Test complete document processing pipeline."""
    chunks = doc_processor.process_document(sample_pdf)
    
    assert len(chunks) > 0
    assert all('text' in chunk for chunk in chunks)
    assert all('document' in chunk for chunk in chunks)
    assert all('source' in chunk for chunk in chunks)
    assert chunks[0]['source'] == sample_pdf
