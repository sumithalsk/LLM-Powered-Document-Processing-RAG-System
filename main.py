"""
Main entry point for running the RAG system API server.
"""
import uvicorn
from src.rag_system.config import settings


if __name__ == "__main__":
    uvicorn.run(
        "src.rag_system.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )
