"""
Unit tests for the embedder module.
"""

import numpy as np
import pytest

from services.ingestion.src.embedding import Embedder


class TestEmbedder:
    """Test cases for the Embedder class."""

    @pytest.fixture
    def embedder(self, test_settings):
        """Create an embedder instance for testing."""
        return Embedder()

    def test_embedder_initialization(self, embedder):
        """Test that the embedder initializes correctly."""
        assert embedder.model is not None
        assert embedder.model_name == "BAAI/bge-large-en-v1.5"
        assert embedder.device is not None

    def test_get_embedding_dimension(self, embedder):
        """Test getting embedding dimension."""
        dim = embedder.get_embedding_dimension()
        assert isinstance(dim, int)
        assert dim == 1024  # bge-large-en-v1.5 dimension

    def test_embed_single_text(self, embedder):
        """Test embedding a single text."""
        text = "Hypertension is a common condition."
        embedding = embedder.embed_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)  # L2 normalized

    def test_embed_empty_text(self, embedder):
        """Test embedding empty text."""
        embedding = embedder.embed_text("")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)

    def test_embed_batch(self, embedder):
        """Test batch embedding."""
        texts = [
            "Hypertension treatment guidelines.",
            "ACE inhibitors mechanism of action.",
            "Side effects monitoring protocol.",
        ]

        embeddings = embedder.embed_batch(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 1024)
        # Check each embedding is normalized
        for i in range(3):
            norm = np.linalg.norm(embeddings[i])
            assert np.isclose(norm, 1.0, atol=1e-5)

    def test_embed_batch_custom_size(self, embedder):
        """Test batch embedding with custom batch size."""
        texts = ["Text " + str(i) for i in range(10)]

        embeddings = embedder.embed_batch(texts, batch_size=3)

        assert embeddings.shape == (10, 1024)

    def test_embed_batch_empty_list(self, embedder):
        """Test batch embedding with empty list."""
        embeddings = embedder.embed_batch([])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0, 1024)

    def test_embed_special_characters(self, embedder):
        """Test embedding text with special characters."""
        text = "Dose: 10mg/dL; pH 7.4 ± 0.2; ≥90% efficacy"
        embedding = embedder.embed_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)

    def test_embed_very_long_text(self, embedder):
        """Test embedding very long text."""
        text = "This is a test sentence. " * 200  # ~1000 words

        embedding = embedder.embed_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)

    def test_embedding_consistency(self, embedder):
        """Test that same text produces same embedding."""
        text = "Consistent embedding test."

        embedding1 = embedder.embed_text(text)
        embedding2 = embedder.embed_text(text)

        assert np.allclose(embedding1, embedding2, atol=1e-6)

    def test_embedding_similarity(self, embedder):
        """Test that similar texts have higher similarity."""
        text1 = "Treatment for hypertension."
        text2 = "Managing high blood pressure."
        text3 = "The weather is sunny today."

        emb1 = embedder.embed_text(text1)
        emb2 = embedder.embed_text(text2)
        emb3 = embedder.embed_text(text3)

        # Cosine similarity (dot product of normalized vectors)
        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)

        # Similar medical texts should be more similar than unrelated text
        assert sim_12 > sim_13

    def test_batch_vs_single_consistency(self, embedder):
        """Test that batch and single embedding produce same results."""
        texts = [
            "First medical text.",
            "Second medical text.",
        ]

        # Batch embedding
        batch_embeddings = embedder.embed_batch(texts)

        # Single embeddings
        single_embeddings = np.array([
            embedder.embed_text(texts[0]),
            embedder.embed_text(texts[1]),
        ])

        # Should be very close (may have minor numerical differences)
        assert np.allclose(batch_embeddings, single_embeddings, atol=1e-4)

    def test_model_name_property(self, embedder):
        """Test model_name property."""
        assert embedder.model_name == "BAAI/bge-large-en-v1.5"

    def test_embed_with_progress(self, embedder):
        """Test batch embedding with progress bar."""
        texts = ["Text " + str(i) for i in range(5)]

        # Should not raise error with show_progress
        embeddings = embedder.embed_batch(texts, show_progress=True)

        assert embeddings.shape == (5, 1024)

    def test_embed_unicode(self, embedder):
        """Test embedding text with unicode characters."""
        text = "β-blocker contraindicated in患者 with ≥ 2 risk factors"
        embedding = embedder.embed_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)
