# Development Guide

**MedArchive RAG Local Development Setup**

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Development Workflow](#development-workflow)
4. [Testing](#testing)
5. [Code Standards](#code-standards)
6. [Debugging Tips](#debugging-tips)
7. [Common Issues](#common-issues)

---

## Prerequisites

### Required Software

- **Python 3.11+**: [Download](https://www.python.org/downloads/)
- **Poetry 1.7+**: [Installation Guide](https://python-poetry.org/docs/#installation)
- **Docker Desktop**: [Download](https://www.docker.com/products/docker-desktop/)
- **Git**: [Download](https://git-scm.com/downloads)

### Recommended Tools

- **VS Code**: With Python and Docker extensions
- **Postman/Thunder Client**: For API testing
- **Docker Compose**: Usually included with Docker Desktop

### API Keys (Required for Full Functionality)

1. **Groq API Key**: 
   - Sign up at [https://console.groq.com/](https://console.groq.com/)
   - Navigate to API Keys and create a new key
   - Free tier: 30 requests/minute

2. **LlamaParse API Key**:
   - Sign up at [https://llamaparse.com/](https://llamaparse.com/)
   - Generate API key from dashboard
   - Free tier: 1000 pages/month

---

## Initial Setup

### 1. Clone and Configure

```powershell
# Clone repository
git clone <repository-url>
cd MedArchive-Rag

# Create environment file from template
Copy-Item .env.example .env

# Edit .env and add your API keys
notepad .env
```

**Critical Environment Variables:**
```env
GROQ_API_KEY=gsk_your_actual_key_here
LLAMAPARSE_API_KEY=llx_your_actual_key_here
```

### 2. Install Dependencies

```powershell
# Install Poetry if not already installed
# (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Install project dependencies
poetry install

# Verify installation
poetry run python --version
# Should output: Python 3.11.x
```

### 3. Start Docker Services

```powershell
# Build and start all services
docker-compose up --build

# Or run in detached mode (background)
docker-compose up -d

# Check service health
docker-compose ps
# All services should show "healthy" status after ~30 seconds
```

**Expected Output:**
```
NAME                  STATUS              PORTS
medarchive-qdrant     Up (healthy)        0.0.0.0:6333->6333/tcp
medarchive-api        Up (healthy)        0.0.0.0:8000->8000/tcp
medarchive-ingestion  Up                  (no exposed ports)
```

### 4. Verify Services

```powershell
# Test API health
Invoke-WebRequest http://localhost:8000/health | Select-Object -Expand Content

# Test Qdrant
Invoke-WebRequest http://localhost:6333/health | Select-Object -Expand Content

# Access API docs in browser
start http://localhost:8000/docs
```

---

## Development Workflow

### Option 1: Docker Compose (Recommended for Full Stack)

```powershell
# Start all services with hot reload
docker-compose up

# The API and Ingestion services automatically reload on code changes
# Edit files in services/api/ or services/ingestion/ and see changes instantly
```

**Pros:**
- Full stack running (Qdrant + API + Ingestion)
- Matches production environment
- No manual dependency management

**Cons:**
- Slower feedback loop for debugging
- Logs mixed across services

### Option 2: Local Development (Recommended for API Work)

```powershell
# Start only Qdrant in Docker
docker-compose up qdrant

# In a separate terminal, run API locally
poetry shell
python -m services.api.src.main

# Or use uvicorn directly for hot reload
poetry run uvicorn services.api.src.main:app --reload --port 8000
```

**Pros:**
- Faster iteration (instant reload)
- Better debugging (breakpoints work)
- Easier to see logs

**Cons:**
- Need to manage environment manually
- Might miss container-specific issues

### Option 3: Hybrid (API Local, Ingestion in Docker)

```powershell
# Start Qdrant and Ingestion
docker-compose up qdrant ingestion

# Run API locally
poetry run uvicorn services.api.src.main:app --reload
```

---

## Testing

### Running Tests

```powershell
# Activate Poetry environment
poetry shell

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test categories
pytest -m unit          # Fast unit tests only
pytest -m integration   # Integration tests (requires Docker)
pytest -m "not slow"    # Skip slow tests

# Run tests in a specific file
pytest tests/unit/test_models.py

# Run a specific test function
pytest tests/unit/test_models.py::TestDocumentMetadata::test_create_document_metadata
```

### Coverage Reports

```powershell
# Generate coverage report
pytest --cov=services --cov=shared --cov-report=html

# Open HTML report
start htmlcov/index.html
```

**Goal: 80%+ coverage for production code**

### Writing Tests

**Unit Test Example:**
```python
# tests/unit/test_example.py
import pytest
from shared.models import DocumentMetadata, DocumentType

def test_document_creation():
    doc = DocumentMetadata(
        title="Test Doc",
        document_type=DocumentType.FORMULARY,
        source_path="/data/test.pdf",
        file_hash="abc123"
    )
    
    assert doc.title == "Test Doc"
    assert doc.document_type == DocumentType.FORMULARY
```

**Integration Test Example:**
```python
# tests/integration/test_api.py
import pytest

@pytest.mark.integration
def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

---

## Code Standards

### Linting and Formatting

```powershell
# Format code with Black
poetry run black services/ shared/ tests/

# Sort imports with isort
poetry run isort services/ shared/ tests/

# Check code style with Flake8
poetry run flake8 services/ shared/ tests/

# Type checking with mypy
poetry run mypy services/ shared/
```

### Pre-Commit Hooks (Optional but Recommended)

```powershell
# Install pre-commit
poetry run pre-commit install

# Now Black, isort, and Flake8 run automatically on git commit
git commit -m "Your changes"  # Auto-formats before committing
```

### Code Style Guidelines

- **Line Length**: 100 characters (enforced by Black)
- **Imports**: Standard library → Third-party → Local (enforced by isort)
- **Type Hints**: Required for all function signatures (checked by mypy)
- **Docstrings**: Required for public functions (Google style)

**Example:**
```python
from typing import List

from pydantic import BaseModel

from shared.models import QueryRequest


def process_query(request: QueryRequest) -> List[str]:
    """
    Process a user query and return relevant documents.
    
    Args:
        request: The query request containing user input
        
    Returns:
        List of document IDs matching the query
        
    Raises:
        ValueError: If query is empty or invalid
    """
    if not request.query.strip():
        raise ValueError("Query cannot be empty")
    
    return ["doc1", "doc2"]
```

---

## Debugging Tips

### API Debugging

**1. Enable Debug Logging:**
```env
# In .env
LOG_LEVEL=DEBUG
```

**2. Use Breakpoints in VS Code:**
```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "FastAPI Debug",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "services.api.src.main:app",
                "--reload",
                "--port", "8000"
            ],
            "jinja": true
        }
    ]
}
```

**3. Interactive API Testing:**
- Use Swagger UI: http://localhost:8000/docs
- Use ReDoc: http://localhost:8000/redoc

### Docker Debugging

```powershell
# View logs for a specific service
docker-compose logs -f api

# Execute commands inside container
docker-compose exec api bash
# Now you're inside the container shell

# Check container resource usage
docker stats

# Rebuild specific service
docker-compose up --build api
```

### Qdrant Debugging

```powershell
# Access Qdrant dashboard
start http://localhost:6333/dashboard

# Check collections via API
Invoke-WebRequest http://localhost:6333/collections | ConvertFrom-Json

# View collection details
Invoke-WebRequest http://localhost:6333/collections/medarchive_documents | ConvertFrom-Json
```

---

## Common Issues

### Issue 1: "ModuleNotFoundError" when running tests

**Cause:** Python path not set correctly

**Solution:**
```powershell
# Ensure you're in Poetry shell
poetry shell

# Or run tests via Poetry
poetry run pytest
```

### Issue 2: Docker services fail to start

**Cause:** Port conflicts or previous containers

**Solution:**
```powershell
# Stop all containers
docker-compose down

# Remove volumes (if persistent data is corrupted)
docker-compose down -v

# Rebuild from scratch
docker-compose up --build
```

### Issue 3: "API Key not found" errors

**Cause:** `.env` file not loaded or incorrect keys

**Solution:**
```powershell
# Verify .env exists
Test-Path .env  # Should return True

# Check environment variables in Docker
docker-compose exec api printenv | Select-String "GROQ"

# Restart services after changing .env
docker-compose restart
```

### Issue 4: Slow embedding model loading

**Cause:** First-time download of large models

**Solution:**
```powershell
# Pre-download models
poetry run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

# Or use smaller model for development
# In .env:
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### Issue 5: Tests fail with ValidationError

**Cause:** Missing required fields in test data

**Solution:**
```python
# Use fixtures from conftest.py
def test_example(sample_document_metadata):
    # Fixture provides valid test data
    assert sample_document_metadata.title is not None
```

---

## Git Workflow

### Branch Strategy

```powershell
# Create feature branch
git checkout -b feature/phase2-llamaparse

# Make changes and commit
git add .
git commit -m "feat: Add LlamaParse integration for PDF parsing"

# Push to remote
git push origin feature/phase2-llamaparse

# Create pull request on GitHub/GitLab
```

### Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding tests
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

**Examples:**
```
feat: Implement two-stage retrieval with reranking
fix: Correct chunk overlap calculation in semantic splitter
docs: Update README with Phase 2 completion status
test: Add unit tests for QueryRequest validation
```

---

## Performance Profiling

### API Latency

```powershell
# Install httpie for better output
pip install httpie

# Time API requests
Measure-Command { http GET http://localhost:8000/health }
```

### Memory Profiling

```python
# Add to code temporarily
from memory_profiler import profile

@profile
def my_function():
    # Your code here
    pass
```

### Load Testing

```powershell
# Install locust
poetry add --group dev locust

# Create locustfile.py
# Run load test
poetry run locust -f locustfile.py
```

---

## Next Steps

1. **Complete Phase 2**: Implement PDF ingestion pipeline
2. **Review Architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Join Discussions**: Participate in code reviews
4. **Report Bugs**: Use GitHub Issues with detailed reproduction steps

---

## Resources

- **FastAPI Tutorial**: https://fastapi.tiangolo.com/tutorial/
- **Poetry Documentation**: https://python-poetry.org/docs/
- **Pytest Guide**: https://docs.pytest.org/
- **Docker Compose Reference**: https://docs.docker.com/compose/

---

**Happy Coding! 🚀**
