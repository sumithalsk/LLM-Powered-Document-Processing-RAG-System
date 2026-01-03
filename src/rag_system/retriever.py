"""
Retrieval pipeline for finding relevant documents.
"""
from typing import List, Dict, Any, Optional
import numpy as np

from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore


class Retriever:
    """Handles document retrieval with similarity search."""
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize the retriever.
        
        Args:
            embedding_generator: Embedding generator instance
            vector_store: Vector store instance
            top_k: Number of top documents to retrieve
            similarity_threshold: Minimum similarity score threshold
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query text
            top_k: Override default top_k
            threshold: Override default similarity threshold
            
        Returns:
            List of relevant documents with scores
        """
        # Use defaults if not specified
        k = top_k if top_k is not None else self.top_k
        thresh = threshold if threshold is not None else self.similarity_threshold
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=k)
        
        # Filter by threshold and format results
        filtered_results = []
        for doc, score in results:
            if score >= thresh:
                result = {
                    'document': doc,
                    'score': score,
                    'text': doc.get('text', ''),
                    'metadata': {k: v for k, v in doc.items() if k != 'text'}
                }
                filtered_results.append(result)
        
        return filtered_results
    
    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> str:
        """
        Retrieve and combine relevant document texts into context.
        
        Args:
            query: Search query text
            top_k: Override default top_k
            threshold: Override default similarity threshold
            
        Returns:
            Combined context string from retrieved documents
        """
        results = self.retrieve(query, top_k, threshold)
        
        # Combine document texts
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result['text']
            source = result['metadata'].get('document', 'Unknown')
            context_parts.append(f"[Source {i}: {source}]\n{text}")
        
        context = "\n\n---\n\n".join(context_parts)
        return context
