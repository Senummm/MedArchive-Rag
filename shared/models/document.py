"""
Shared Pydantic models for MedArchive RAG system.

These models ensure type safety and validation across microservices.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    """Supported document types."""

    CLINICAL_GUIDELINE = "clinical_guideline"
    FORMULARY = "formulary"
    PROTOCOL = "protocol"
    POLICY = "policy"
    RESEARCH_PAPER = "research_paper"
    OTHER = "other"


class ProcessingStatus(str, Enum):
    """Document processing status in the ingestion pipeline."""

    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """
    Metadata for source documents in the document store.

    Tracks document identity, versioning, and lineage for audit trails.
    """

    document_id: UUID = Field(default_factory=uuid4, description="Unique document identifier")
    title: str = Field(..., description="Human-readable document title")
    document_type: DocumentType = Field(..., description="Classification of document")
    source_path: str = Field(..., description="Original file path or URL")
    file_hash: str = Field(..., description="MD5 hash for change detection")
    version: str = Field(default="1.0", description="Document version")
    department: Optional[str] = Field(None, description="Owning department (e.g., Cardiology)")
    author: Optional[str] = Field(None, description="Document author or maintainer")
    effective_date: Optional[datetime] = Field(None, description="When this version became active")
    review_date: Optional[datetime] = Field(
        None, description="Next scheduled review date"
    )
    page_count: Optional[int] = Field(None, ge=1, description="Total pages in document")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    error_message: Optional[str] = Field(None, description="Error details if processing failed")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Pediatric Antibiotic Formulary 2026",
                "document_type": "formulary",
                "source_path": "/data/guidelines/pediatric_antibiotics_2026.pdf",
                "file_hash": "5d41402abc4b2a76b9719d911017c592",
                "department": "Pediatrics",
                "effective_date": "2026-01-01T00:00:00Z",
                "page_count": 42,
            }
        }


class ChunkMetadata(BaseModel):
    """
    Metadata for semantic chunks stored in the vector database.

    Each chunk represents a semantically meaningful section of a document.
    """

    chunk_id: UUID = Field(default_factory=uuid4, description="Unique chunk identifier")
    document_id: UUID = Field(..., description="Parent document reference")
    chunk_index: int = Field(..., ge=0, description="Sequential position in document")
    text: str = Field(..., min_length=10, description="The actual chunk text content")

    # Semantic Context
    section_path: Optional[str] = Field(
        None, description="Hierarchical path (e.g., 'Chapter 3 > Dosing > Pediatric')"
    )
    heading: Optional[str] = Field(None, description="Nearest heading or section title")

    # Source Location
    page_numbers: List[int] = Field(default_factory=list, description="Source page numbers")
    start_char_idx: Optional[int] = Field(None, description="Character offset in source document")
    end_char_idx: Optional[int] = Field(None, description="End character offset")

    # Vector Metadata (populated after embedding)
    embedding_model: Optional[str] = Field(None, description="Model used for embedding")
    token_count: Optional[int] = Field(None, ge=1, description="Number of tokens in chunk")

    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Ensure text contains meaningful content."""
        if not v.strip():
            raise ValueError("Chunk text cannot be empty or whitespace-only")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "chunk_index": 5,
                "text": "Amoxicillin pediatric dosing: 15mg/kg twice daily for children under 40kg.",
                "section_path": "Antibiotics > Beta-Lactams > Amoxicillin",
                "heading": "Pediatric Dosing",
                "page_numbers": [12, 13],
                "token_count": 18,
            }
        }


class RetrievalResult(BaseModel):
    """
    A single retrieved chunk with its relevance score.

    Used in the two-stage retrieval pipeline.
    """

    chunk_id: UUID
    document_id: UUID
    text: str
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Chunk metadata (section, pages, etc.)"
    )

    # Reranking (Stage 2)
    rerank_score: Optional[float] = Field(None, description="Score after reranking")


class QueryRequest(BaseModel):
    """
    User query request to the RAG API.

    Supports optional filters and retrieval tuning.
    """

    query: str = Field(..., min_length=3, description="The user's question or search query")
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of results to return after reranking"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Metadata filters (e.g., {'department': 'Cardiology'})"
    )
    enable_reranking: bool = Field(default=True, description="Apply two-stage retrieval")
    stream_response: bool = Field(default=True, description="Stream LLM response")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the pediatric dosage for Amoxicillin?",
                "top_k": 5,
                "filters": {"document_type": "formulary"},
                "enable_reranking": True,
            }
        }


class Citation(BaseModel):
    """
    A citation linking generated text to source documents.

    Ensures zero-hallucination by providing verifiable references.
    """

    document_id: UUID
    document_title: str
    page_numbers: List[int]
    text_snippet: str = Field(..., description="The exact text from the source")
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    """
    RAG system response with generated answer and citations.

    This is what the API returns to the user.
    """

    query: str
    answer: str = Field(..., description="The generated response from the LLM")
    citations: List[Citation] = Field(
        default_factory=list, description="Source references for fact-checking"
    )
    retrieved_chunks: List[RetrievalResult] = Field(
        default_factory=list, description="Raw retrieval results (for debugging)"
    )
    latency_ms: float = Field(..., description="Total query processing time")
    model_used: str = Field(..., description="LLM model identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the pediatric dosage for Amoxicillin?",
                "answer": "The pediatric dosage for Amoxicillin is 15mg/kg twice daily...",
                "citations": [
                    {
                        "document_title": "Pediatric Antibiotic Formulary 2026",
                        "page_numbers": [12],
                        "text_snippet": "15mg/kg twice daily for children under 40kg",
                        "relevance_score": 0.95,
                    }
                ],
                "latency_ms": 287.3,
                "model_used": "llama-3.3-70b-versatile",
            }
        }


class HealthResponse(BaseModel):
    """Health check response for monitoring."""

    status: str = Field(default="healthy", description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    dependencies: Dict[str, bool] = Field(
        default_factory=dict, description="Status of external dependencies"
    )
