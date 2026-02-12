"""
Unit tests for configuration management.

Tests Settings loading, validation, and environment parsing.
"""

import os

import pytest
from pydantic import ValidationError

from shared.utils import Settings, get_settings


class TestSettings:
    """Tests for Settings configuration class."""

    def test_settings_with_defaults(self, monkeypatch):
        """Test Settings with default values."""
        # Set required environment variables
        monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")
        monkeypatch.setenv("LLAMAPARSE_API_KEY", "test_llamaparse_key")

        settings = Settings()

        assert settings.environment == "development"
        assert settings.log_level == "INFO"
        assert settings.api_port == 8000
        assert settings.embedding_model == "BAAI/bge-large-en-v1.5"

    def test_settings_from_environment(self, monkeypatch):
        """Test Settings loaded from environment variables."""
        monkeypatch.setenv("GROQ_API_KEY", "my_groq_key")
        monkeypatch.setenv("LLAMAPARSE_API_KEY", "my_llamaparse_key")
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        monkeypatch.setenv("API_PORT", "9000")

        settings = Settings()

        assert settings.groq_api_key == "my_groq_key"
        assert settings.environment == "production"
        assert settings.log_level == "WARNING"
        assert settings.api_port == 9000

    def test_settings_parse_cors_origins(self, monkeypatch):
        """Test CORS origins are parsed from comma-separated string."""
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        monkeypatch.setenv("LLAMAPARSE_API_KEY", "test_key")
        monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000, http://localhost:8080")

        settings = Settings()

        assert isinstance(settings.cors_origins, list)
        assert len(settings.cors_origins) == 2
        assert "http://localhost:3000" in settings.cors_origins
        assert "http://localhost:8080" in settings.cors_origins

    def test_settings_invalid_environment(self, monkeypatch):
        """Test that invalid environment values are rejected."""
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        monkeypatch.setenv("LLAMAPARSE_API_KEY", "test_key")
        monkeypatch.setenv("ENVIRONMENT", "invalid_env")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "Environment must be one of" in str(exc_info.value)

    def test_settings_properties(self, test_settings):
        """Test Settings helper properties."""
        # Development environment
        dev_settings = test_settings
        assert dev_settings.is_development is False  # testing environment
        assert dev_settings.is_production is False

        # Production environment
        prod_settings = Settings(
            groq_api_key="key",
            llamaparse_api_key="key",
            environment="production",
        )
        assert prod_settings.is_production is True
        assert prod_settings.is_development is False

    def test_get_settings_caching(self, monkeypatch):
        """Test that get_settings() returns cached instance."""
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        monkeypatch.setenv("LLAMAPARSE_API_KEY", "test_key")

        # Clear cache first
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        # Should return the same instance
        assert settings1 is settings2

    def test_settings_validation_ranges(self, monkeypatch):
        """Test numeric field validation ranges."""
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        monkeypatch.setenv("LLAMAPARSE_API_KEY", "test_key")

        # Valid values
        settings = Settings(api_port=8080, retrieval_top_k=10)
        assert settings.api_port == 8080
        assert settings.retrieval_top_k == 10

        # Invalid port
        with pytest.raises(ValidationError):
            Settings(api_port=70000)  # Over 65535

        # Invalid top_k
        with pytest.raises(ValidationError):
            Settings(retrieval_top_k=0)  # Must be >= 1
