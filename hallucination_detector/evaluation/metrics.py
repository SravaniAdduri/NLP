"""
Evaluation metrics computation module.
Provides detailed metric calculations for benchmarking.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class DetailedMetrics:
    """Detailed evaluation metrics for the system."""
    # Classification metrics
    precision: float
    recall: float
    f1_score: float
    accuracy: float

    # Hallucination-specific
    hallucination_rate: float
    false_positive_rate: float
    false_negative_rate: float

    # Retrieval metrics
    retrieval_precision_at_k: float
    retrieval_recall_at_k: float
    ndcg_at_k: float
    mrr: float

    # Latency
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


def compute_precision_recall_f1(
    predicted: List[str],
    ground_truth: List[str],
    positive_label: str = "CONTRADICTED",
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for binary classification.
    
    Args:
        predicted: List of predicted labels.
        ground_truth: List of true labels.
        positive_label: The label considered as "positive" (hallucination).
        
    Returns:
        Tuple of (precision, recall, f1).
    """
    tp = sum(1 for p, g in zip(predicted, ground_truth) if p == positive_label and g == positive_label)
    fp = sum(1 for p, g in zip(predicted, ground_truth) if p == positive_label and g != positive_label)
    fn = sum(1 for p, g in zip(predicted, ground_truth) if p != positive_label and g == positive_label)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_accuracy(predicted: List[str], ground_truth: List[str]) -> float:
    """Compute overall accuracy."""
    if not predicted or not ground_truth:
        return 0.0
    correct = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
    return correct / len(predicted)


def compute_hallucination_rate(labels: List[str]) -> float:
    """Compute hallucination rate from predicted labels."""
    if not labels:
        return 0.0
    hallucinated = sum(1 for l in labels if l in ("CONTRADICTED", "NOT_ENOUGH_EVIDENCE"))
    return hallucinated / len(labels)


def compute_ndcg_at_k(relevance_scores: List[float], k: int = 5) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.
    
    Args:
        relevance_scores: List of relevance scores (1 = relevant, 0 = not relevant).
        k: Number of top results to consider.
        
    Returns:
        NDCG@k score.
    """
    scores = relevance_scores[:k]
    if not scores:
        return 0.0

    # DCG
    dcg = scores[0]
    for i in range(1, len(scores)):
        dcg += scores[i] / np.log2(i + 1)

    # Ideal DCG (sort scores descending)
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = ideal_scores[0]
    for i in range(1, len(ideal_scores)):
        idcg += ideal_scores[i] / np.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_mrr(ranked_relevance: List[List[bool]]) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        ranked_relevance: List of query results, each being a list of booleans
                         indicating if the result at that rank is relevant.
        
    Returns:
        MRR score.
    """
    if not ranked_relevance:
        return 0.0

    reciprocal_ranks = []
    for results in ranked_relevance:
        for i, is_relevant in enumerate(results):
            if is_relevant:
                reciprocal_ranks.append(1.0 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks))


def compute_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
    """
    Compute latency percentiles.
    
    Args:
        latencies: List of latency measurements in milliseconds.
        
    Returns:
        Dict with avg, p50, p95, p99 latency values.
    """
    if not latencies:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    arr = np.array(latencies)
    return {
        "avg": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
