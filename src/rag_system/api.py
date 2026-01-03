"""
FastAPI REST API for the RAG system.
"""
import os
import tempfile
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .rag import RAGSystem
from .config import settings


# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for Q&A queries."""
    question: str = Field(..., description="Question to ask the system")
    top_k: Optional[int] = Field(None, description="Number of documents to retrieve")
    temperature: float = Field(0.7, ge=0, le=2, description="LLM temperature")


class QueryResponse(BaseModel):
    """Response model for Q&A queries."""
    answer: str
    sources: list
    retrieved_chunks: int
    model: Optional[str] = None
    tokens_used: Optional[int] = None


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    status: str
    document: Optional[str] = None
    chunks_processed: int
    total_documents: int
    message: Optional[str] = None


class SystemStatsResponse(BaseModel):
    """Response model for system statistics."""
    total_documents: int
    embedding_dimension: int
    embedding_model: str
    llm_model: str
    vector_store_path: str


# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Production-ready Retrieval-Augmented Generation API for document Q&A",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup."""
    global rag_system
    rag_system = RAGSystem()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG System API",
        "version": "0.1.0",
        "endpoints": {
            "query": "/query",
            "upload": "/upload",
            "stats": "/stats",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "total_documents": rag_system.vector_store.get_total_documents() if rag_system else 0
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    Args:
        request: Query request with question and parameters
        
    Returns:
        Answer with sources and metadata
    """
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = rag_system.query(
            question=request.question,
            top_k=request.top_k,
            temperature=request.temperature
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    method: str = Query("pdfplumber", description="PDF extraction method (pdfplumber or pypdf2)")
):
    """
    Upload and ingest a PDF document.
    
    Args:
        file: PDF file to upload
        method: Extraction method to use
        
    Returns:
        Upload results with chunk count
    """
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Ingest document
        result = rag_system.ingest_document(tmp_path, method=method)
        
        # Save vector store
        rag_system.save_vector_store()
        
        return DocumentUploadResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/stats", response_model=SystemStatsResponse)
async def get_stats():
    """
    Get system statistics.
    
    Returns:
        System statistics including document count and configuration
    """
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    stats = rag_system.get_stats()
    return SystemStatsResponse(**stats)


@app.delete("/documents")
async def clear_documents():
    """
    Clear all documents from the vector store.
    
    Returns:
        Confirmation message
    """
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        rag_system.vector_store.clear()
        rag_system.save_vector_store()
        return {"status": "success", "message": "All documents cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
