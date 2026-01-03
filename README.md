# LLM-Powered Document Processing RAG System

A production-ready Retrieval-Augmented Generation (RAG) system that integrates OpenAI GPT models with vector databases for intelligent document understanding and Q&A.

## 🚀 Key Features

- **Document Ingestion & Preprocessing**: Supports PDF parsing with multiple extraction methods (PyPDF2, pdfplumber)
- **Intelligent Text Chunking**: Configurable chunk size with overlap for better context preservation
- **Vector Embeddings**: Uses sentence-transformers for high-quality semantic embeddings
- **Efficient Retrieval**: FAISS-based vector store for fast similarity search
- **LLM Integration**: OpenAI GPT models with optimized prompts for accurate responses
- **REST API**: FastAPI-based API for easy integration
- **Automated Testing**: Comprehensive unit and integration tests
- **Production Ready**: Configurable, scalable, and well-documented

## 📋 Requirements

- Python 3.8+
- OpenAI API key

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/sumithalsk/LLM-Powered-Document-Processing-RAG-System.git
cd LLM-Powered-Document-Processing-RAG-System
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## 🚀 Quick Start

### Running the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Example Usage

#### 1. Upload a Document

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

#### 2. Query the System

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the document?",
    "top_k": 5,
    "temperature": 0.7
  }'
```

#### 3. Get System Statistics

```bash
curl -X GET "http://localhost:8000/stats"
```

## 🏗️ Architecture

The system consists of several key components:

### 1. Document Processor (`document_processor.py`)
- Extracts text from PDF files
- Preprocesses and cleans text
- Splits text into overlapping chunks

### 2. Embedding Generator (`embeddings.py`)
- Generates vector embeddings using sentence-transformers
- Supports batch processing for efficiency

### 3. Vector Store (`vector_store.py`)
- FAISS-based vector database
- Supports different index types (FlatL2, FlatIP)
- Persistent storage and loading

### 4. Retriever (`retriever.py`)
- Similarity-based document retrieval
- Configurable top-k and threshold filtering
- Context aggregation

### 5. LLM Handler (`llm.py`)
- OpenAI GPT integration
- Prompt optimization for RAG
- Response generation with metadata

### 6. RAG System (`rag.py`)
- Main orchestrator that combines all components
- End-to-end document ingestion and query pipeline

### 7. REST API (`api.py`)
- FastAPI-based API endpoints
- Document upload and query endpoints
- Health checks and statistics

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/rag_system --cov-report=html

# Run specific test file
pytest tests/test_document_processor.py

# Run with verbose output
pytest -v
```

## ⚙️ Configuration

Configure the system via environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | OpenAI model name | `gpt-3.5-turbo` |
| `EMBEDDING_MODEL` | Sentence-transformer model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Text chunk size | `512` |
| `CHUNK_OVERLAP` | Chunk overlap size | `50` |
| `TOP_K_RESULTS` | Number of retrieval results | `5` |
| `SIMILARITY_THRESHOLD` | Minimum similarity score | `0.7` |
| `VECTOR_STORE_PATH` | Vector store directory | `./vector_store` |
| `API_HOST` | API server host | `0.0.0.0` |
| `API_PORT` | API server port | `8000` |

## 📊 API Endpoints

### POST /upload
Upload a PDF document for ingestion.

**Request:**
- `file`: PDF file (multipart/form-data)
- `method`: Extraction method (optional, default: "pdfplumber")

**Response:**
```json
{
  "status": "success",
  "document": "document_name",
  "chunks_processed": 25,
  "total_documents": 100
}
```

### POST /query
Query the RAG system.

**Request:**
```json
{
  "question": "What is machine learning?",
  "top_k": 5,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "answer": "Machine learning is...",
  "sources": [...],
  "retrieved_chunks": 5,
  "model": "gpt-3.5-turbo",
  "tokens_used": 245
}
```

### GET /stats
Get system statistics.

**Response:**
```json
{
  "total_documents": 100,
  "embedding_dimension": 384,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model": "gpt-3.5-turbo",
  "vector_store_path": "./vector_store"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "total_documents": 100
}
```

### DELETE /documents
Clear all documents from the vector store.

**Response:**
```json
{
  "status": "success",
  "message": "All documents cleared"
}
```

## 🔧 Usage Examples

### Python SDK Usage

```python
from src.rag_system.rag import RAGSystem

# Initialize the system
rag = RAGSystem(
    openai_api_key="your-api-key",
    vector_store_path="./my_vector_store"
)

# Ingest a document
result = rag.ingest_document("path/to/document.pdf")
print(f"Processed {result['chunks_processed']} chunks")

# Query the system
response = rag.query("What is the main topic?")
print(f"Answer: {response['answer']}")
print(f"Sources: {response['sources']}")

# Save the vector store
rag.save_vector_store()
```

## 🎯 Performance Optimization

- **Batch Processing**: Embeddings are generated in batches for efficiency
- **Vector Indexing**: FAISS provides fast similarity search
- **Chunking Strategy**: Configurable chunk size and overlap for optimal retrieval
- **Caching**: Vector store can be persisted and reloaded

## 🔒 Security Considerations

- Store API keys in environment variables, never in code
- Use `.gitignore` to exclude sensitive files
- Validate and sanitize user inputs
- Implement rate limiting for production deployments

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Sentence-Transformers for embeddings
- FAISS for vector search
- FastAPI for the web framework

## 📧 Contact

For questions or support, please open an issue on GitHub.
