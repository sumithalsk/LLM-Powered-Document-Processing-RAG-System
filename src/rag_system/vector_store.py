"""
Vector store implementation using FAISS for similarity search.
"""
import os
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss


class VectorStore:
    """FAISS-based vector store for similarity search."""
    
    def __init__(self, dimension: int, index_type: str = "FlatL2"):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embedding vectors
            index_type: Type of FAISS index ('FlatL2', 'FlatIP', 'IVFFlat')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index(index_type)
        self.documents = []  # Store document chunks
        self.id_to_doc_map = {}  # Map vector IDs to document indices
    
    def _create_index(self, index_type: str) -> faiss.Index:
        """Create FAISS index based on type."""
        if index_type == "FlatL2":
            # Exact L2 distance search
            return faiss.IndexFlatL2(self.dimension)
        elif index_type == "FlatIP":
            # Exact inner product (cosine similarity)
            return faiss.IndexFlatIP(self.dimension)
        elif index_type == "IVFFlat":
            # Inverted file index (faster for large datasets)
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            embeddings: Array of embedding vectors (n_docs x dimension)
            documents: List of document metadata dictionaries
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        # Normalize embeddings for cosine similarity (if using IP index)
        if self.index_type == "FlatIP":
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Get current index size
        start_id = self.index.ntotal
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and create ID mapping
        for i, doc in enumerate(documents):
            doc_idx = len(self.documents)
            self.documents.append(doc)
            self.id_to_doc_map[start_id + i] = doc_idx
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of tuples (document, similarity_score)
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        if self.index_type == "FlatIP":
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx in self.id_to_doc_map:
                doc_idx = self.id_to_doc_map[idx]
                doc = self.documents[doc_idx]
                
                # Convert distance to similarity score
                if self.index_type == "FlatL2":
                    # For L2 distance, convert to similarity (inverse)
                    similarity = 1 / (1 + dist)
                else:
                    # For IP, distance is already similarity
                    similarity = float(dist)
                
                results.append((doc, similarity))
        
        return results
    
    def save(self, path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Directory path to save the store
        """
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(path, "index.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save documents and metadata
        metadata = {
            'documents': self.documents,
            'id_to_doc_map': self.id_to_doc_map,
            'dimension': self.dimension,
            'index_type': self.index_type
        }
        metadata_path = os.path.join(path, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load(cls, path: str) -> 'VectorStore':
        """
        Load a vector store from disk.
        
        Args:
            path: Directory path to load from
            
        Returns:
            Loaded VectorStore instance
        """
        # Load metadata
        metadata_path = os.path.join(path, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        store = cls(metadata['dimension'], metadata['index_type'])
        
        # Load FAISS index
        index_path = os.path.join(path, "index.faiss")
        store.index = faiss.read_index(index_path)
        
        # Restore metadata
        store.documents = metadata['documents']
        store.id_to_doc_map = metadata['id_to_doc_map']
        
        return store
    
    def clear(self) -> None:
        """Clear all documents and reset the index."""
        self.index = self._create_index(self.index_type)
        self.documents = []
        self.id_to_doc_map = {}
    
    def get_total_documents(self) -> int:
        """Get total number of documents in the store."""
        return len(self.documents)
