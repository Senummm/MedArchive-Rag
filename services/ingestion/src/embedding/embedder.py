"""
Embedding Model for Semantic Vectors.

Uses sentence-transformers (BAAI/bge-large-en-v1.5) to generate
dense vector embeddings for semantic search.
"""

from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from shared.utils import get_settings, setup_logging

settings = get_settings()
logger = setup_logging("ingestion.embedder", settings.log_level)


class Embedder:
    """
    Embedding generator using sentence-transformers.

    Uses BAAI/bge-large-en-v1.5 by default, which provides excellent
    performance on medical and scientific text.
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier (defaults to settings)
            device: Device to use ('cpu' or 'cuda', defaults to settings)
        """
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device

        logger.info(
            "Loading embedding model...",
            extra={
                "model": self.model_name,
                "device": self.device,
            },
        )

        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )

            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            logger.info(
                "Embedding model loaded successfully",
                extra={
                    "model": self.model_name,
                    "dimension": self.embedding_dim,
                    "device": self.device,
                },
            )

        except Exception as e:
            logger.error(
                "Failed to load embedding model",
                exc_info=e,
                extra={"model": self.model_name},
            )
            raise

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            NumPy array of shape (embedding_dim,)
        """
        if not text or not text.strip():
            logger.warning("Attempted to embed empty text")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(
                "Failed to generate embedding",
                exc_info=e,
                extra={"text_length": len(text)},
            )
            raise

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (default: 32)
            show_progress: Show progress bar (default: False)

        Returns:
            NumPy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            logger.warning("Attempted to embed empty batch")
            return np.array([]).reshape(0, self.embedding_dim)

        # Filter out empty strings
        valid_texts = [text if text and text.strip() else " " for text in texts]

        logger.info(
            "Generating embeddings for batch",
            extra={
                "batch_size": len(texts),
                "model": self.model_name,
            },
        )

        try:
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )

            logger.info(
                "Batch embedding complete",
                extra={
                    "texts_processed": len(texts),
                    "embedding_shape": embeddings.shape,
                },
            )

            return embeddings.astype(np.float32)

        except Exception as e:
            logger.error(
                "Failed to generate batch embeddings",
                exc_info=e,
                extra={"batch_size": len(texts)},
            )
            raise

    def embed_chunks(
        self,
        chunks: List[dict],
        text_key: str = "text",
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of chunk dictionaries.

        Args:
            chunks: List of chunk dictionaries
            text_key: Key to extract text from chunks (default: 'text')
            batch_size: Batch size for processing

        Returns:
            NumPy array of shape (len(chunks), embedding_dim)
        """
        texts = [chunk.get(text_key, "") for chunk in chunks]
        return self.embed_batch(texts, batch_size=batch_size, show_progress=True)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension (e.g., 1024 for BGE-large)
        """
        return self.embedding_dim

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model name, dimension, and device
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "max_seq_length": self.model.max_seq_length,
        }
