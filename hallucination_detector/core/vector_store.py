"""
FAISS Vector Store Module
Manages the FAISS index for efficient similarity search.
Supports adding, searching, saving, and loading vector indices.
"""

import os
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import faiss
from loguru import logger

from core.chunking import TextChunk


@dataclass
class SearchResult:
    """Represents a single search result from the vector store."""
    content: str
    score: float
    source: str
    chunk_id: int
    metadata: dict


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    Stores document chunks with their embeddings and metadata.
    Supports persistence (save/load from disk).
    """

    def __init__(self, embedding_dimension: int, store_path: str = "./data/vector_store"):
        """
        Initialize the FAISS vector store.
        
        Args:
            embedding_dimension: Dimension of the embedding vectors.
            store_path: Directory path for persisting the index.
        """
        self.embedding_dimension = embedding_dimension
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        # Initialize FAISS index with L2 distance (inner product after normalization = cosine)
        self.index = faiss.IndexFlatIP(embedding_dimension)
        self.documents: List[dict] = []  # Parallel list of document metadata
        self._id_counter = 0

        logger.info(f"Initialized FAISS vector store with dimension {embedding_dimension}")

    def add_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray) -> List[int]:
        """
        Add document chunks and their embeddings to the store.
        
        Args:
            chunks: List of TextChunk objects.
            embeddings: Numpy array of shape (num_chunks, embedding_dim).
            
        Returns:
            List of assigned IDs for the added chunks.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings")

        # Normalize embeddings for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        ids = []
        for i, chunk in enumerate(chunks):
            doc_entry = {
                "id": self._id_counter,
                "content": chunk.content,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "metadata": chunk.metadata,
            }
            self.documents.append(doc_entry)
            ids.append(self._id_counter)
            self._id_counter += 1

        self.index.add(embeddings)
        logger.info(f"Added {len(chunks)} chunks to vector store. Total: {self.index.ntotal}")
        return ids

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """
        Search for the most similar documents to the query embedding.
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,).
            top_k: Number of results to return.
            
        Returns:
            List of SearchResult objects ordered by similarity (highest first).
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty. No results to return.")
            return []

        # Reshape and normalize query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        # Limit top_k to available documents
        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            results.append(SearchResult(
                content=doc["content"],
                score=float(score),
                source=doc["source"],
                chunk_id=doc["chunk_id"],
                metadata=doc["metadata"],
            ))

        return results

    def save(self, index_name: str = "default") -> None:
        """
        Persist the FAISS index and metadata to disk.
        
        Args:
            index_name: Name for the saved index files.
        """
        index_path = self.store_path / f"{index_name}.index"
        meta_path = self.store_path / f"{index_name}_metadata.json"

        faiss.write_index(self.index, str(index_path))

        metadata = {
            "documents": self.documents,
            "id_counter": self._id_counter,
            "embedding_dimension": self.embedding_dimension,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved vector store to {self.store_path} (index: {index_name})")

    def load(self, index_name: str = "default") -> bool:
        """
        Load a FAISS index and metadata from disk.
        
        Args:
            index_name: Name of the saved index files to load.
            
        Returns:
            True if successfully loaded, False otherwise.
        """
        index_path = self.store_path / f"{index_name}.index"
        meta_path = self.store_path / f"{index_name}_metadata.json"

        if not index_path.exists() or not meta_path.exists():
            logger.warning(f"No saved index found at {self.store_path} for '{index_name}'")
            return False

        self.index = faiss.read_index(str(index_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.documents = metadata["documents"]
        self._id_counter = metadata["id_counter"]

        logger.info(f"Loaded vector store: {self.index.ntotal} vectors")
        return True

    def clear(self) -> None:
        """Clear all data from the vector store."""
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.documents = []
        self._id_counter = 0
        logger.info("Vector store cleared")

    @property
    def size(self) -> int:
        """Return the number of vectors in the store."""
        return self.index.ntotal
