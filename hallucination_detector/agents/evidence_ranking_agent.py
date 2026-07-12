"""
Evidence Ranking Agent
Responsible for:
1. Reranking retrieved documents by relevance using cross-encoder.
2. Removing duplicate/near-duplicate evidence.
3. Selecting the top-k most relevant documents.
"""

from typing import List
from dataclasses import dataclass, field

from loguru import logger

from core.reranker import CrossEncoderReranker, RankedResult
from core.vector_store import SearchResult


@dataclass
class RankingResult:
    """Complete result from the evidence ranking process."""
    query: str
    ranked_evidence: List[RankedResult]
    num_input: int
    num_output: int
    num_duplicates_removed: int


class EvidenceRankingAgent:
    """
    Ranks and filters retrieved evidence for quality and relevance.
    
    Performs:
    - Cross-encoder reranking for precise relevance scoring
    - Near-duplicate removal via content overlap
    - Top-k selection with configurable threshold
    - Minimum relevance filtering
    """

    def __init__(
        self,
        reranker: CrossEncoderReranker,
        top_k: int = 5,
        min_relevance_score: float = -5.0,
        dedup_threshold: float = 0.85,
    ):
        """
        Initialize the Evidence Ranking Agent.
        
        Args:
            reranker: CrossEncoderReranker instance.
            top_k: Number of top results to keep.
            min_relevance_score: Minimum cross-encoder score to include.
            dedup_threshold: Jaccard similarity threshold for deduplication.
        """
        self.reranker = reranker
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score
        self.dedup_threshold = dedup_threshold
        logger.info(f"Evidence Ranking Agent initialized (top_k={top_k}, min_score={min_relevance_score})")

    def rank(self, query: str, search_results: List[SearchResult]) -> RankingResult:
        """
        Rank and filter search results.
        
        Pipeline:
        1. Deduplicate near-identical passages
        2. Rerank using cross-encoder
        3. Filter by minimum relevance score
        4. Return top-k results
        
        Args:
            query: The user's search query.
            search_results: Raw search results from the retrieval agent.
            
        Returns:
            RankingResult with filtered and ranked evidence.
        """
        if not search_results:
            return RankingResult(
                query=query,
                ranked_evidence=[],
                num_input=0,
                num_output=0,
                num_duplicates_removed=0,
            )

        num_input = len(search_results)

        # Step 1: Deduplicate and rerank
        ranked_results = self.reranker.deduplicate_and_rerank(
            query=query,
            results=search_results,
            top_k=self.top_k,
            similarity_threshold=self.dedup_threshold,
        )

        num_after_dedup = len(ranked_results)
        num_duplicates = num_input - num_after_dedup - (num_input - min(num_input, self.top_k * 2))

        # Step 2: Filter by minimum relevance score
        filtered_results = [
            r for r in ranked_results
            if r.relevance_score >= self.min_relevance_score
        ]

        # Step 3: Ensure we have at least some results (relax threshold if needed)
        if not filtered_results and ranked_results:
            # Take at least the top result even if below threshold
            filtered_results = ranked_results[:1]
            logger.warning("All results below min_relevance_score, keeping top-1")

        logger.info(
            f"Ranking complete: {num_input} input -> {len(filtered_results)} output "
            f"(dedup removed ~{max(0, num_duplicates)})"
        )

        return RankingResult(
            query=query,
            ranked_evidence=filtered_results,
            num_input=num_input,
            num_output=len(filtered_results),
            num_duplicates_removed=max(0, num_input - num_after_dedup),
        )

    def get_evidence_texts(self, ranking_result: RankingResult) -> List[str]:
        """
        Extract just the text content from ranking results.
        
        Args:
            ranking_result: Complete ranking result.
            
        Returns:
            List of evidence text strings.
        """
        return [r.content for r in ranking_result.ranked_evidence]

    def get_evidence_with_scores(self, ranking_result: RankingResult) -> List[dict]:
        """
        Get evidence with metadata for display.
        
        Args:
            ranking_result: Complete ranking result.
            
        Returns:
            List of dicts with content, score, source, and rank.
        """
        return [
            {
                "content": r.content,
                "relevance_score": r.relevance_score,
                "source": r.source,
                "rank": r.rank,
            }
            for r in ranking_result.ranked_evidence
        ]
