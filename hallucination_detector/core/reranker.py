"""
Cross-Encoder Reranker Module
Reranks retrieved documents using a cross-encoder model for better relevance scoring.
Uses ms-marco-MiniLM for passage reranking.
"""

from typing import List, Tuple
from dataclasses import dataclass

from sentence_transformers import CrossEncoder
from loguru import logger

from core.vector_store import SearchResult


@dataclass
class RankedResult:
    """A search result with cross-encoder relevance score."""
    content: str
    relevance_score: float
    original_score: float
    source: str
    chunk_id: int
    metadata: dict
    rank: int


class CrossEncoderReranker:
    """
    Reranks search results using a cross-encoder model.
    Cross-encoders process query-document pairs jointly, providing
    more accurate relevance scores than bi-encoder similarity.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", device: str = None):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: HuggingFace cross-encoder model name.
            device: Device for inference ('cpu', 'cuda', 'mps').
        """
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5,
    ) -> List[RankedResult]:
        """
        Rerank search results using the cross-encoder.
        
        Args:
            query: The original search query.
            results: List of SearchResult objects from vector search.
            top_k: Number of top results to return after reranking.
            
        Returns:
            List of RankedResult objects sorted by relevance (highest first).
        """
        if not results:
            return []

        # Create query-document pairs for cross-encoder
        pairs = [(query, result.content) for result in results]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Combine scores with original results
        scored_results = []
        for i, (result, score) in enumerate(zip(results, scores)):
            scored_results.append(RankedResult(
                content=result.content,
                relevance_score=float(score),
                original_score=result.score,
                source=result.source,
                chunk_id=result.chunk_id,
                metadata=result.metadata,
                rank=0,  # Will be set after sorting
            ))

        # Sort by cross-encoder score (descending)
        scored_results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Assign ranks and return top_k
        top_results = scored_results[:top_k]
        for i, result in enumerate(top_results):
            result.rank = i + 1

        logger.debug(f"Reranked {len(results)} results, returning top {top_k}")
        return top_results

    def deduplicate_and_rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5,
        similarity_threshold: float = 0.85,
    ) -> List[RankedResult]:
        """
        Remove near-duplicate documents, then rerank the remaining ones.
        
        Uses content overlap to detect duplicates before reranking.
        
        Args:
            query: The original search query.
            results: List of SearchResult objects.
            top_k: Number of results to return.
            similarity_threshold: Threshold for considering two passages as duplicates.
            
        Returns:
            Deduplicated and reranked results.
        """
        if not results:
            return []

        # Deduplicate based on content similarity
        unique_results = []
        seen_contents = []

        for result in results:
            is_duplicate = False
            for seen in seen_contents:
                if self._content_overlap(result.content, seen) > similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_results.append(result)
                seen_contents.append(result.content)

        logger.debug(f"Removed {len(results) - len(unique_results)} duplicates")

        # Rerank the unique results
        return self.rerank(query, unique_results, top_k)

    def _content_overlap(self, text1: str, text2: str) -> float:
        """
        Compute word-level Jaccard similarity between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Jaccard similarity score between 0 and 1.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
