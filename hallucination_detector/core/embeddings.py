"""
Embedding Engine Module
Generates vector embeddings using Sentence Transformers models (BGE, MiniLM, E5).
Supports batch processing and caching.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger


class EmbeddingEngine:
    """
    Generates dense vector embeddings using Sentence Transformers.
    Default model: BAAI/bge-small-en-v1.5 (free, high quality, fast).
    Also supports: all-MiniLM-L6-v2, intfloat/e5-small-v2.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = None):
        """
        Initialize the embedding engine.
        
        Args:
            model_name: HuggingFace model name for sentence embeddings.
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dimension}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string.
            
        Returns:
            Numpy array of shape (embedding_dim,).
        """
        # BGE models require a prefix for queries
        if "bge" in self.model_name.lower():
            text = "Represent this sentence: " + text

        embedding = self.model.encode(text, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input text strings.
            batch_size: Number of texts to process in each batch.
            
        Returns:
            Numpy array of shape (num_texts, embedding_dim).
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dimension)

        # Add prefix for BGE models
        processed_texts = texts
        if "bge" in self.model_name.lower():
            processed_texts = ["Represent this sentence: " + t for t in texts]

        logger.debug(f"Embedding {len(texts)} texts with batch_size={batch_size}")
        embeddings = self.model.encode(
            processed_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        Uses query-specific prefix for BGE models.
        
        Args:
            query: Search query string.
            
        Returns:
            Numpy array of shape (embedding_dim,).
        """
        if "bge" in self.model_name.lower():
            query = "Represent this sentence for searching relevant passages: " + query

        embedding = self.model.encode(query, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.
            
        Returns:
            Cosine similarity score between -1 and 1.
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dimension
