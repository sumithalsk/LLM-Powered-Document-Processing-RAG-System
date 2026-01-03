"""
Document ingestion and preprocessing module.
Handles PDF parsing, text extraction, and chunking.
"""
import os
import re
from typing import List, Dict, Any
from pathlib import Path

import PyPDF2
import pdfplumber


class DocumentProcessor:
    """Handles document ingestion and preprocessing."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str, method: str = "pdfplumber") -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method ('pdfplumber' or 'pypdf2')
            
        Returns:
            Extracted text content
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if method == "pdfplumber":
            return self._extract_with_pdfplumber(pdf_path)
        elif method == "pypdf2":
            return self._extract_with_pypdf2(pdf_path)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (better for complex layouts)."""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 (faster, simpler)."""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?;:()\-\n]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # Find the last complete sentence or word boundary
            if end < len(text):
                # Try to break at sentence end
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    # Fall back to word boundary
                    space_pos = text.rfind(' ', start, end)
                    if space_pos > start:
                        end = space_pos
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': end,
                    'length': len(chunk_text)
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_document(self, pdf_path: str, method: str = "pdfplumber") -> List[Dict[str, Any]]:
        """
        Complete pipeline: extract, preprocess, and chunk document.
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method
            
        Returns:
            List of processed text chunks with metadata
        """
        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path, method)
        
        # Preprocess
        clean_text = self.preprocess_text(raw_text)
        
        # Chunk
        chunks = self.chunk_text(clean_text)
        
        # Add document metadata
        doc_name = Path(pdf_path).stem
        for chunk in chunks:
            chunk['document'] = doc_name
            chunk['source'] = pdf_path
        
        return chunks
