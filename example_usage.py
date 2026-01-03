"""
Example usage of the RAG System.

This script demonstrates how to use the RAG system programmatically.
"""
import os
from pathlib import Path

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

from src.rag_system.rag import RAGSystem


def main():
    """Demonstrate RAG system usage."""
    print("=" * 60)
    print("RAG System Example Usage")
    print("=" * 60)
    
    # Initialize the RAG system
    print("\n1. Initializing RAG System...")
    rag = RAGSystem()
    print(f"   ✓ System initialized")
    print(f"   - Embedding model: {rag.embedding_model_name}")
    print(f"   - LLM model: {rag.openai_model}")
    print(f"   - Embedding dimension: {rag.embedding_generator.get_dimension()}")
    
    # Check if there are sample PDFs to ingest
    print("\n2. Document Ingestion")
    print("   Note: Place PDF files in the current directory to test ingestion")
    
    # Get stats
    stats = rag.get_stats()
    print(f"\n3. System Statistics:")
    print(f"   - Total documents: {stats['total_documents']}")
    print(f"   - Embedding dimension: {stats['embedding_dimension']}")
    print(f"   - Vector store path: {stats['vector_store_path']}")
    
    # Example query (only if documents are loaded)
    if stats['total_documents'] > 0:
        print("\n4. Querying the system...")
        response = rag.query(
            question="What is the main topic of the documents?",
            top_k=3,
            temperature=0.7
        )
        print(f"   Question: What is the main topic of the documents?")
        print(f"   Answer: {response['answer']}")
        print(f"   Retrieved {response['retrieved_chunks']} chunks")
        print(f"   Sources: {len(response['sources'])} documents")
    else:
        print("\n4. No documents loaded yet")
        print("   Upload PDF documents using the API or programmatically:")
        print("   Example: rag.ingest_document('path/to/your/document.pdf')")
    
    print("\n" + "=" * 60)
    print("For API usage, run: python main.py")
    print("Then visit: http://localhost:8000/docs")
    print("=" * 60)


if __name__ == "__main__":
    main()
