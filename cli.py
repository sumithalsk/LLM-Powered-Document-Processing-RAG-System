"""
Simple CLI tool for the RAG system.
"""
import os
import sys
import argparse
from pathlib import Path

from src.rag_system.rag import RAGSystem


def ingest_command(args):
    """Ingest a document."""
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return 1
    
    print(f"Ingesting document: {args.file}")
    rag = RAGSystem()
    result = rag.ingest_document(args.file, method=args.method)
    
    if result['status'] == 'success':
        print(f"✓ Successfully ingested {result['document']}")
        print(f"  - Chunks processed: {result['chunks_processed']}")
        print(f"  - Total documents: {result['total_documents']}")
        rag.save_vector_store()
        print("✓ Vector store saved")
    else:
        print(f"✗ Error: {result.get('message', 'Unknown error')}")
        return 1
    
    return 0


def query_command(args):
    """Query the system."""
    print(f"Query: {args.question}")
    rag = RAGSystem()
    
    stats = rag.get_stats()
    if stats['total_documents'] == 0:
        print("Error: No documents in the system. Please ingest documents first.")
        return 1
    
    response = rag.query(
        question=args.question,
        top_k=args.top_k,
        temperature=args.temperature
    )
    
    print("\nAnswer:")
    print("-" * 60)
    print(response['answer'])
    print("-" * 60)
    print(f"\nRetrieved {response['retrieved_chunks']} chunks from {len(response['sources'])} sources")
    
    if args.show_sources:
        print("\nSources:")
        for i, source in enumerate(response['sources'], 1):
            print(f"{i}. {source['document']} (score: {source['score']:.3f})")
            if args.verbose:
                print(f"   {source['text_preview']}")
    
    return 0


def stats_command(args):
    """Show system statistics."""
    rag = RAGSystem()
    stats = rag.get_stats()
    
    print("RAG System Statistics")
    print("=" * 60)
    print(f"Total documents:      {stats['total_documents']}")
    print(f"Embedding dimension:  {stats['embedding_dimension']}")
    print(f"Embedding model:      {stats['embedding_model']}")
    print(f"LLM model:            {stats['llm_model']}")
    print(f"Vector store path:    {stats['vector_store_path']}")
    print("=" * 60)
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest a PDF document')
    ingest_parser.add_argument('file', help='Path to PDF file')
    ingest_parser.add_argument('--method', choices=['pdfplumber', 'pypdf2'],
                               default='pdfplumber', help='PDF extraction method')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--top-k', type=int, default=5,
                             help='Number of documents to retrieve')
    query_parser.add_argument('--temperature', type=float, default=0.7,
                             help='LLM temperature (0-2)')
    query_parser.add_argument('--show-sources', action='store_true',
                             help='Show source documents')
    query_parser.add_argument('--verbose', action='store_true',
                             help='Show detailed source previews')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show system statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it in your .env file or export it:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return 1
    
    # Execute command
    if args.command == 'ingest':
        return ingest_command(args)
    elif args.command == 'query':
        return query_command(args)
    elif args.command == 'stats':
        return stats_command(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
