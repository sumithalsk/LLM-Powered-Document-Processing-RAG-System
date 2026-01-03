"""
Main RAG system orchestrator.
"""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import settings
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .retriever import Retriever
from .llm import LLMHandler


class RAGSystem:
    """Main RAG system that orchestrates all components."""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        vector_store_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        openai_model: Optional[str] = None
    ):
        """
        Initialize the RAG system.
        
        Args:
            openai_api_key: OpenAI API key (uses settings if not provided)
            vector_store_path: Path to vector store (uses settings if not provided)
            embedding_model: Embedding model name (uses settings if not provided)
            openai_model: OpenAI model name (uses settings if not provided)
        """
        # Configuration
        self.openai_api_key = openai_api_key or settings.openai_api_key
        self.vector_store_path = vector_store_path or settings.vector_store_path
        self.embedding_model_name = embedding_model or settings.embedding_model
        self.openai_model = openai_model or settings.openai_model
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        self.embedding_generator = EmbeddingGenerator(self.embedding_model_name)
        
        # Initialize or load vector store
        self.vector_store = self._init_vector_store()
        
        self.retriever = Retriever(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store,
            top_k=settings.top_k_results,
            similarity_threshold=settings.similarity_threshold
        )
        
        self.llm_handler = LLMHandler(
            api_key=self.openai_api_key,
            model=self.openai_model
        )
    
    def _init_vector_store(self) -> VectorStore:
        """Initialize or load vector store."""
        if os.path.exists(self.vector_store_path):
            try:
                return VectorStore.load(self.vector_store_path)
            except Exception as e:
                print(f"Warning: Could not load vector store: {e}")
                print("Creating new vector store...")
        
        # Create new vector store
        dimension = self.embedding_generator.get_dimension()
        return VectorStore(dimension=dimension, index_type="FlatIP")
    
    def ingest_document(self, pdf_path: str, method: str = "pdfplumber") -> Dict[str, Any]:
        """
        Ingest a PDF document into the system.
        
        Args:
            pdf_path: Path to PDF file
            method: PDF extraction method
            
        Returns:
            Ingestion results
        """
        # Process document
        chunks = self.document_processor.process_document(pdf_path, method)
        
        if not chunks:
            return {
                'status': 'error',
                'message': 'No text could be extracted from the document',
                'chunks_processed': 0
            }
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Add to vector store
        self.vector_store.add_documents(embeddings, chunks)
        
        return {
            'status': 'success',
            'document': Path(pdf_path).stem,
            'chunks_processed': len(chunks),
            'total_documents': self.vector_store.get_total_documents()
        }
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            temperature: LLM temperature
            
        Returns:
            Response with answer and metadata
        """
        # Retrieve relevant context
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
        
        if not retrieved_docs:
            return {
                'answer': "I couldn't find any relevant information in the knowledge base to answer your question.",
                'sources': [],
                'retrieved_chunks': 0
            }
        
        # Get context text
        context = self.retriever.retrieve_context(question, top_k=top_k)
        
        # Generate response
        llm_response = self.llm_handler.generate_response(
            query=question,
            context=context,
            temperature=temperature
        )
        
        # Prepare sources
        sources = []
        for doc in retrieved_docs:
            sources.append({
                'document': doc['metadata'].get('document', 'Unknown'),
                'score': doc['score'],
                'text_preview': doc['text'][:200] + '...' if len(doc['text']) > 200 else doc['text']
            })
        
        return {
            'answer': llm_response.get('answer', ''),
            'sources': sources,
            'retrieved_chunks': len(retrieved_docs),
            'model': llm_response.get('model'),
            'tokens_used': llm_response.get('tokens_used')
        }
    
    def save_vector_store(self) -> None:
        """Save the vector store to disk."""
        self.vector_store.save(self.vector_store_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            'total_documents': self.vector_store.get_total_documents(),
            'embedding_dimension': self.embedding_generator.get_dimension(),
            'embedding_model': self.embedding_model_name,
            'llm_model': self.openai_model,
            'vector_store_path': self.vector_store_path
        }
