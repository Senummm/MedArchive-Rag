# Contributing to MedArchive RAG

**Welcome to the MedArchive RAG project!** We're building production-ready medical AI that helps clinicians provide better patient care.

---

## 🎯 Project Mission

**Goal**: Create a clinical decision support system that provides sub-second, evidence-based answers sourced directly from verified medical documents with zero hallucination.

**Values**:
- **Patient Safety First**: Every feature prioritizes accuracy and verifiability
- **Clinical Workflow**: Designed for real healthcare environments
- **Open Innovation**: Collaborative development with medical professionals
- **Quality Code**: Production-ready, well-tested, maintainable systems

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Git** with configured SSH keys
- **API Keys**: Groq (LLM), LlamaParse (PDF processing)
- **Development Tools**: VS Code (recommended), Docker (optional)

### Development Setup

```bash
# 1. Fork and clone the repository
git clone git@github.com:your-username/MedArchive-Rag.git
cd MedArchive-Rag

# 2. Set up Python virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# or
.venv\Scripts\Activate.ps1     # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Verify installation
python -m pytest tests/ -v
python -c "from services.api.src.main import app; print('✅ Setup complete!')"
```

### First Contribution Workflow

```bash
# 1. Create feature branch
git checkout -b feature/your-feature-name
git checkout -b fix/issue-description
git checkout -b docs/documentation-update

# 2. Make your changes
# ... develop amazing features ...

# 3. Run quality checks
black services/ shared/           # Code formatting
flake8 services/ shared/          # Linting
mypy services/ shared/            # Type checking
pytest tests/ -v --cov=services  # Testing

# 4. Commit with clear messages
git add .
git commit -m "feat: add conversation session persistence"

# 5. Push and create pull request
git push origin feature/your-feature-name
# Create PR via GitHub interface
```

---

## 📁 Project Structure

### Module Organization

```
MedArchive-RAG/
├── services/              # Core application services
│   ├── api/              # FastAPI REST + WebSocket server
│   │   └── src/
│   │       ├── llm/      # Groq LLM integration
│   │       ├── retrieval/ # Vector search engine
│   │       ├── conversation/ # Session management
│   │       └── citations/    # Source attribution
│   └── ingestion/        # Background PDF processing
│       └── src/
│           ├── parsers/  # LlamaParse integration
│           ├── chunking/ # Semantic text segmentation
│           └── indexing/ # Qdrant vector database
│
├── shared/               # Cross-service utilities
│   ├── models/          # Pydantic data models
│   └── utils/           # Configuration, logging
│
├── static/              # Frontend web interface
│   └── index.html       # React-like vanilla JS
│
├── tests/               # Comprehensive test suite
│   ├── unit/           # Fast, isolated tests
│   ├── integration/    # Service-level tests
│   └── e2e/            # End-to-end workflows
│
└── docs/               # Documentation
    ├── modules/        # Module-specific guides
    ├── ARCHITECTURE.md # System design
    └── DEVELOPMENT.md  # Dev environment setup
```

### Naming Conventions

- **Files**: `snake_case.py`, `kebab-case.md`
- **Classes**: `PascalCase` (e.g., `ConversationSession`)
- **Functions**: `snake_case` (e.g., `extract_citations`)
- **Variables**: `snake_case` (e.g., `session_id`, `chunk_count`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_TOKENS`, `API_TIMEOUT`)
- **Modules**: `snake_case` directories (e.g., `conversation/`, `llm/`)

---

## 🔧 Development Standards

### Code Quality

#### Python Style Guide

```python
# ✅ Good: Clear, descriptive function with type hints
async def extract_citations(
    answer: str,
    search_results: List[SearchResult],
    similarity_threshold: float = 0.4
) -> List[Citation]:
    """Extract citations from LLM answer using fuzzy text matching.

    Args:
        answer: Generated response text
        search_results: Retrieved document chunks
        similarity_threshold: Minimum relevance score (0.0-1.0)

    Returns:
        List of formatted citations with source references

    Raises:
        CitationError: If text matching fails
    """
    logger.info(f"Extracting citations from {len(search_results)} sources")

    try:
        matches = self._find_answer_sources(answer, search_results)
        citations = [
            self._format_citation(match.chunk, match.similarity)
            for match in matches
            if match.similarity > similarity_threshold
        ]

        logger.info(f"Found {len(citations)} citations")
        return citations

    except Exception as e:
        logger.error(f"Citation extraction failed: {e}")
        raise CitationError(f"Failed to extract citations: {e}") from e


# ❌ Avoid: Unclear function without types or docs
def get_stuff(data, config):
    result = []
    for item in data:
        if item.score > config.threshold:
            result.append(format_thing(item))
    return result
```

#### Error Handling Patterns

```python
# ✅ Good: Specific exception handling with context
try:
    response = await groq_client.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=0.1
    )
    return response.choices[0].message.content

except groq.RateLimitError as e:
    logger.warning(f"Rate limited, retrying in {e.retry_after}s")
    await asyncio.sleep(e.retry_after)
    return await self._generate_with_retry(messages, retry_count + 1)

except groq.AuthenticationError as e:
    logger.error(f"Groq authentication failed: {e}")
    raise LLMServiceError("Invalid API credentials") from e

except Exception as e:
    logger.error(f"Unexpected LLM error: {e}", exc_info=True)
    raise LLMServiceError(f"Generation failed: {e}") from e


# ❌ Avoid: Generic exception catching
try:
    result = some_api_call()
except Exception as e:
    print(f"Error: {e}")  # No context, poor logging
    return None           # Silent failure
```

#### Logging Standards

```python
import logging
from shared.utils import get_logger

logger = get_logger(__name__)

# ✅ Good: Structured logging with context
logger.info(
    "Query processing completed",
    extra={
        "query_id": query_id,
        "latency_ms": latency,
        "chunk_count": len(retrieved_chunks),
        "model_used": self.model_name
    }
)

# ✅ Good: Appropriate log levels
logger.debug(f"Raw embedding shape: {embedding.shape}")        # Development
logger.info(f"Processed {chunk_count} document chunks")        # Operations
logger.warning(f"High memory usage: {memory_mb}MB")            # Attention needed
logger.error(f"Vector search failed: {error}", exc_info=True) # Requires action

# ❌ Avoid: Poor logging practices
print(f"Processing query: {query}")              # Use logger, not print
logger.info(f"Error occurred: {sensitive_data}") # Don't log sensitive data
logger.error("Something went wrong")              # Too vague, no context
```

### Testing Requirements

#### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, Mock
from services.api.src.conversation import SessionManager, ConversationSession

class TestSessionManager:

    @pytest.fixture
    def session_manager(self):
        return SessionManager()

    def test_create_session_generates_unique_id(self, session_manager):
        """Test that each new session gets a unique UUID."""
        session1 = session_manager.create_session()
        session2 = session_manager.create_session()

        assert session1.session_id != session2.session_id
        assert len(session1.messages) == 0
        assert len(session2.messages) == 0

    def test_get_or_create_existing_session(self, session_manager):
        """Test retrieving existing session by ID."""
        # Create initial session
        original = session_manager.create_session()
        original.add_message("user", "Hello")

        # Retrieve same session
        retrieved = session_manager.get_or_create_session(original.session_id)

        assert retrieved.session_id == original.session_id
        assert len(retrieved.messages) == 1
        assert retrieved.messages[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_conversation_context_extraction(self, session_manager):
        """Test conversation history context building."""
        session = session_manager.create_session()

        # Add conversation turns
        session.add_message("user", "What is diabetes?")
        session.add_message("assistant", "Diabetes is a chronic condition...")
        session.add_message("user", "How can I prevent it?")

        # Extract context
        context = session.get_context(max_turns=2)

        assert len(context) == 3  # 2 complete turns + 1 partial
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "What is diabetes?"
        assert context[-1]["content"] == "How can I prevent it?"
```

#### Integration Tests

```python
import pytest
import asyncio
from httpx import AsyncClient

@pytest.mark.integration
class TestChatAPI:

    @pytest.fixture
    async def client(self):
        from services.api.src.main import app
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac

    @pytest.mark.asyncio
    async def test_conversation_flow(self, client):
        """Test end-to-end conversation with follow-up."""

        # First message
        response1 = await client.post("/api/v1/chat", json={
            "message": "What is diabetes?",
            "enable_reranking": True
        })

        assert response1.status_code == 200
        data1 = response1.json()
        session_id = data1["session_id"]

        assert "diabetes" in data1["message"].lower()
        assert len(data1["citations"]) > 0

        # Follow-up message
        response2 = await client.post("/api/v1/chat", json={
            "message": "How can I prevent it?",
            "session_id": session_id,
            "enable_reranking": True
        })

        assert response2.status_code == 200
        data2 = response2.json()

        # Should maintain same session
        assert data2["session_id"] == session_id

        # Should understand "it" refers to diabetes
        assert "prevention" in data2["message"].lower()
        assert "diabetes" in data2["message"].lower()
```

---

## 📝 Commit Guidelines

### Conventional Commits

We use [Conventional Commits](https://conventionalcommits.org/) for clear, automated versioning:

```bash
# Feature additions
git commit -m "feat: add conversation session persistence"
git commit -m "feat(api): implement WebSocket streaming for real-time chat"

# Bug fixes
git commit -m "fix: resolve memory leak in citation extraction"
git commit -m "fix(ingestion): handle malformed PDF tables gracefully"

# Documentation
git commit -m "docs: add module architecture diagrams"
git commit -m "docs(api): update conversation API examples"

# Performance improvements
git commit -m "perf: optimize vector search with batch queries"
git commit -m "perf(llm): implement response caching"

# Refactoring
git commit -m "refactor: simplify session management interface"
git commit -m "refactor(retrieval): extract reranking into separate service"

# Tests
git commit -m "test: add integration tests for citation extraction"
git commit -m "test(conversation): cover edge cases for session cleanup"

# Breaking changes (add !)
git commit -m "feat!: redesign citation API for better performance"
```

### Commit Message Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types**:
- `feat`: New functionality
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style/formatting (no logic changes)
- `refactor`: Code restructuring without functional changes
- `perf`: Performance improvements
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Scopes** (optional):
- `api`, `ingestion`, `conversation`, `llm`, `retrieval`, `citations`, `frontend`

---

## 🔄 Pull Request Process

### PR Checklist

```markdown
## PR Checklist

- [ ] **Code Quality**
  - [ ] Code follows project style guidelines
  - [ ] Type hints added for all functions
  - [ ] Docstrings added for public APIs
  - [ ] Error handling implemented

- [ ] **Testing**
  - [ ] Unit tests added/updated
  - [ ] Integration tests pass
  - [ ] Test coverage >90% for new code
  - [ ] Manual testing completed

- [ ] **Documentation**
  - [ ] README updated if needed
  - [ ] Module docs updated
  - [ ] API docs reflect changes
  - [ ] Comments explain complex logic

- [ ] **Performance & Security**
  - [ ] No performance regressions
  - [ ] No sensitive data logged
  - [ ] Input validation implemented
  - [ ] Error messages don't leak internals

- [ ] **Medical Safety** (if applicable)
  - [ ] Medical accuracy verified
  - [ ] Citation sources confirmed
  - [ ] No hallucination risk introduced
```

### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Medical Impact
- [ ] No medical/clinical impact
- [ ] Improves medical accuracy
- [ ] Changes citation behavior
- [ ] Affects clinical workflow

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance benchmarks run

## Screenshots (if applicable)
<!-- Add screenshots for UI changes -->

## Additional Notes
<!-- Any other relevant information -->
```

### Review Process

1. **Automated Checks**: CI runs tests, linting, type checking
2. **Code Review**: At least one team member reviews code quality
3. **Medical Review**: For clinical features, medical expert reviews accuracy
4. **Performance Review**: For performance-critical changes, benchmark review
5. **Documentation Review**: For public APIs, documentation clarity review

---

## 🏥 Medical Safety Guidelines

### Clinical Accuracy Standards

- **Source Verification**: All medical information must be traceable to citations
- **Hallucination Prevention**: LLM responses must be grounded in retrieved context
- **Currency**: Medical guidelines should be recent (prefer <2 years old)
- **Peer Review**: Clinical features reviewed by medical professionals

### Prohibited Content

- **No Diagnosis**: System provides information, not medical diagnosis
- **No Treatment Advice**: Information only, not specific treatment recommendations
- **No Emergency Guidance**: Clear disclaimers for emergency situations
- **No Medication Dosing**: General information only, not specific dosing

### Error Handling for Medical Context

```python
# ✅ Good: Safe medical error handling
if not retrieved_chunks:
    return {
        "answer": "I don't have sufficient information from the medical documents to answer this question safely. Please consult current clinical guidelines or a healthcare professional.",
        "citations": [],
        "warning": "INSUFFICIENT_CONTEXT"
    }

# ✅ Good: Clear limitations
response_disclaimer = """
This information is for educational purposes only and
should not replace professional medical advice.
Always consult with qualified healthcare providers for
patient-specific recommendations.
"""

# ❌ Avoid: Overconfident medical claims
# "The patient should take 500mg of metformin twice daily"
# Better: "Guidelines suggest metformin starting doses typically range from..."
```

---

## 🚀 Feature Development Workflow

### Planning Phase

1. **Issue Creation**: Create GitHub issue with medical context
2. **Requirements Review**: Medical accuracy requirements defined
3. **Architecture Design**: Technical approach documented
4. **Acceptance Criteria**: Clear, testable requirements

### Development Phase

1. **Branch Creation**: Feature branch from `main`
2. **TDD Approach**: Write tests first for critical functionality
3. **Iterative Development**: Small commits with clear progress
4. **Documentation**: Update relevant module docs

### Testing Phase

1. **Unit Testing**: Function-level correctness
2. **Integration Testing**: Module interaction verification
3. **Performance Testing**: Latency and throughput validation
4. **Medical Testing**: Clinical accuracy validation

### Deployment Phase

1. **Code Review**: Technical and medical review
2. **Staging Deployment**: Test in staging environment
3. **User Acceptance**: Clinical user feedback
4. **Production Deployment**: Gradual rollout

---

## 📚 Resources for Contributors

### Technical Documentation
- **[Module Documentation](modules/README.md)** - Detailed component guides
- **[Architecture Guide](ARCHITECTURE.md)** - System design patterns
- **[Development Setup](DEVELOPMENT.md)** - Local environment configuration

### Medical Resources
- **[UpToDate](https://www.uptodate.com/)** - Evidence-based medical information
- **[PubMed](https://pubmed.ncbi.nlm.nih.gov/)** - Medical literature database
- **[Clinical Guidelines](https://www.guidelines.gov/)** - Evidence-based recommendations

### AI/ML Resources
- **[Groq Documentation](https://console.groq.com/docs)** - LLM API reference
- **[Qdrant Documentation](https://qdrant.tech/documentation/)** - Vector database guide
- **[BGE Models](https://huggingface.co/BAAI)** - Embedding model documentation

### Development Tools
- **[Pre-commit Hooks](https://pre-commit.com/)** - Automated code quality
- **[pytest](https://pytest.org/)** - Testing framework
- **[FastAPI](https://fastapi.tiangolo.com/)** - API framework documentation

---

## 🌍 Community & Support

### Getting Help

- **GitHub Discussions**: Technical questions and feature discussions
- **GitHub Issues**: Bug reports and feature requests
- **Code Review**: Learning opportunity through PR feedback
- **Documentation**: Comprehensive guides for all components

### Contributing Beyond Code

- **Documentation**: Improve guides and examples
- **Testing**: Add test cases and performance benchmarks
- **Medical Review**: Clinical accuracy validation
- **UI/UX**: Frontend interface improvements
- **Translation**: Multi-language support

### Recognition

- **Contributor Credits**: Listed in project documentation
- **Feature Attribution**: Significant contributions highlighted in releases
- **Community Roles**: Active contributors can become maintainers
- **Conference Presentations**: Opportunities to present work

---

## 📄 License & Code of Conduct

### MIT License
This project is licensed under the MIT License - see [LICENSE](../LICENSE) for details.

### Code of Conduct
We are committed to providing a welcoming and inspiring community for all. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for guidelines on respectful collaboration.

### Medical Ethics
All contributors must adhere to medical ethics principles:
- **First, do no harm**: Patient safety is paramount
- **Accuracy**: Medical information must be evidence-based
- **Transparency**: Sources and limitations clearly communicated
- **Privacy**: No patient data in open source code

---

**🏥 Thank you for contributing to better medical AI!**

*Together, we're building tools that help clinicians provide better patient care through evidence-based decision support.*
