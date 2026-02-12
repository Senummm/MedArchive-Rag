"""
Configuration management using Pydantic Settings.

Loads configuration from environment variables with validation and type safety.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Uses .env file in development, environment variables in production.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    environment: str = Field(default="development", description="Deployment environment")
    log_level: str = Field(default="INFO", description="Logging verbosity")
    log_format: str = Field(default="text", description="Log format: json or text")
    log_output: str = Field(default="stdout", description="Log destination")
    app_name: str = Field(default="MedArchive-RAG", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")

    # API Service
    api_host: str = Field(default="0.0.0.0", description="API bind host")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API port")
    api_workers: int = Field(default=4, ge=1, description="Uvicorn workers")
    api_reload: bool = Field(default=True, description="Hot reload in dev")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="Allowed CORS origins (comma-separated)",
    )

    # LLM Services
    groq_api_key: str = Field(..., description="Groq API key for inference")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile", description="Groq model identifier"
    )
    groq_max_tokens: int = Field(default=4096, ge=1, description="Max output tokens")
    groq_temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="LLM temperature (lower = more factual)"
    )

    llamaparse_api_key: str = Field(..., description="LlamaParse API key")
    llamaparse_result_type: str = Field(default="markdown", description="Parse output format")
    llamaparse_verbose: bool = Field(default=False, description="Verbose parsing logs")

    # Vector Database (Qdrant)
    qdrant_url: str = Field(default="http://qdrant:6333", description="Qdrant server URL")
    qdrant_api_key: Optional[str] = Field(None, description="Qdrant Cloud API key (optional)")
    qdrant_collection_name: str = Field(
        default="medarchive_documents", description="Collection name"
    )
    qdrant_vector_size: int = Field(default=1024, ge=1, description="Embedding dimension")
    qdrant_distance_metric: str = Field(default="Cosine", description="Distance metric")
    qdrant_use_binary_quantization: bool = Field(
        default=True, description="Enable BQ for performance"
    )

    # Embedding Models
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5", description="Dense embedding model"
    )
    embedding_device: str = Field(default="cpu", description="Device: cpu or cuda")
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3", description="Reranking model"
    )
    reranker_device: str = Field(default="cpu", description="Device: cpu or cuda")

    # Retrieval Configuration
    retrieval_top_k: int = Field(default=50, ge=1, description="Initial retrieval count")
    retrieval_score_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )
    rerank_top_k: int = Field(default=5, ge=1, description="Final chunks for LLM")
    rerank_enabled: bool = Field(default=True, description="Enable two-stage retrieval")
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Semantic weight")
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Keyword weight")

    # Document Processing
    chunk_size: int = Field(default=1024, ge=100, description="Target chunk size (tokens)")
    chunk_overlap: int = Field(default=200, ge=0, description="Chunk overlap (tokens)")
    min_chunk_size: int = Field(default=100, ge=1, description="Minimum chunk size")
    document_source_path: str = Field(
        default="./data/document_store", description="PDF source directory"
    )
    enable_incremental_sync: bool = Field(
        default=True, description="Only process changed files"
    )
    hash_algorithm: str = Field(default="md5", description="File hashing algorithm")

    # Data Storage
    vector_storage_path: str = Field(
        default="./data/vector_storage", description="Vector DB persistence"
    )
    document_metadata_path: str = Field(
        default="./data/document_store", description="Document metadata storage"
    )

    # Observability
    enable_tracing: bool = Field(default=False, description="Enable Arize Phoenix tracing")
    phoenix_collector_endpoint: str = Field(
        default="http://localhost:6006", description="Phoenix collector URL"
    )

    # Feature Flags
    enable_async_processing: bool = Field(default=True, description="Async document processing")
    enable_rate_limiting: bool = Field(default=True, description="API rate limiting")
    rate_limit_per_minute: int = Field(default=60, ge=1, description="Requests per minute")
    enable_query_cache: bool = Field(default=True, description="Cache query results")
    cache_ttl_seconds: int = Field(default=3600, ge=1, description="Cache TTL")

    @field_validator("cors_origins")
    @classmethod
    def parse_cors_origins(cls, v: str) -> List[str]:
        """Parse comma-separated CORS origins."""
        return [origin.strip() for origin in v.split(",") if origin.strip()]

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Ensure environment is valid."""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Using lru_cache ensures settings are loaded once and reused.
    """
    return Settings()  # type: ignore
