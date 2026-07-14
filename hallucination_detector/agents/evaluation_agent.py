"""
Evaluation Agent
Responsible for:
1. Computing precision, recall, F1-score, accuracy.
2. Measuring hallucination rate.
3. Tracking latency.
4. Evaluating retrieval quality.
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import numpy as np
from loguru import logger


@dataclass
class EvaluationMetrics:
    """Complete set of evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    hallucination_rate: float
    latency_ms: float
    retrieval_precision: float
    retrieval_recall: float
    num_claims_verified: int
    num_claims_supported: int
    num_claims_contradicted: int
    num_claims_neutral: int
    average_confidence: float
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Benchmark comparison between systems."""
    system_name: str
    metrics: EvaluationMetrics
    num_queries: int
    avg_latency_ms: float
    avg_hallucination_rate: float
    avg_f1: float


class EvaluationAgent:
    """
    Evaluates the performance of the hallucination detection and fact verification system.
    
    Metrics computed:
    - Precision: Of claims marked as hallucinated, how many truly are
    - Recall: Of all actual hallucinations, how many were detected
    - F1-score: Harmonic mean of precision and recall
    - Accuracy: Overall correctness of classification
    - Hallucination Rate: Fraction of claims that are not supported
    - Latency: End-to-end processing time
    - Retrieval Quality: Relevance of retrieved documents
    """

    def __init__(self):
        """Initialize the Evaluation Agent."""
        self._benchmark_history: List[EvaluationMetrics] = []
        logger.info("Evaluation Agent initialized")

    def evaluate(
        self,
        predicted_labels: List[str],
        true_labels: List[str],
        latency_ms: float,
        confidence_scores: List[float] = None,
        retrieval_relevant: List[bool] = None,
    ) -> EvaluationMetrics:
        """
        Compute evaluation metrics by comparing predictions against ground truth.
        
        Args:
            predicted_labels: Predicted labels for each claim (SUPPORTED/CONTRADICTED/NOT_ENOUGH_EVIDENCE).
            true_labels: Ground truth labels for each claim.
            latency_ms: End-to-end latency in milliseconds.
            confidence_scores: Confidence scores for each prediction.
            retrieval_relevant: Boolean flags indicating if each retrieved doc was relevant.
            
        Returns:
            EvaluationMetrics with all computed metrics.
        """
        if not predicted_labels or not true_labels:
            return self._empty_metrics(latency_ms)

        # Ensure equal lengths
        min_len = min(len(predicted_labels), len(true_labels))
        predicted = predicted_labels[:min_len]
        true = true_labels[:min_len]

        # Binary classification: SUPPORTED vs NOT_SUPPORTED (CONTRADICTED + NEUTRAL)
        pred_binary = [1 if p == "SUPPORTED" else 0 for p in predicted]
        true_binary = [1 if t == "SUPPORTED" else 0 for t in true]

        # Compute core metrics
        tp = sum(1 for p, t in zip(pred_binary, true_binary) if p == 0 and t == 0)  # True hallucination detected
        fp = sum(1 for p, t in zip(pred_binary, true_binary) if p == 0 and t == 1)  # False alarm
        fn = sum(1 for p, t in zip(pred_binary, true_binary) if p == 1 and t == 0)  # Missed hallucination
        tn = sum(1 for p, t in zip(pred_binary, true_binary) if p == 1 and t == 1)  # Correct support

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(predicted) if predicted else 0.0

        # Hallucination rate
        hallucination_count = sum(1 for p in predicted if p != "SUPPORTED")
        hallucination_rate = hallucination_count / len(predicted) if predicted else 0.0

        # Retrieval quality
        retrieval_precision = 0.0
        retrieval_recall = 0.0
        if retrieval_relevant:
            relevant_count = sum(retrieval_relevant)
            retrieval_precision = relevant_count / len(retrieval_relevant) if retrieval_relevant else 0.0
            retrieval_recall = relevant_count / max(relevant_count, 1)  # Simplified

        # Claim counts
        num_supported = sum(1 for p in predicted if p == "SUPPORTED")
        num_contradicted = sum(1 for p in predicted if p == "CONTRADICTED")
        num_neutral = sum(1 for p in predicted if p == "NOT_ENOUGH_EVIDENCE")

        # Average confidence
        avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0

        metrics = EvaluationMetrics(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            accuracy=round(accuracy, 4),
            hallucination_rate=round(hallucination_rate, 4),
            latency_ms=round(latency_ms, 2),
            retrieval_precision=round(retrieval_precision, 4),
            retrieval_recall=round(retrieval_recall, 4),
            num_claims_verified=len(predicted),
            num_claims_supported=num_supported,
            num_claims_contradicted=num_contradicted,
            num_claims_neutral=num_neutral,
            average_confidence=round(avg_confidence, 4),
        )

        self._benchmark_history.append(metrics)
        return metrics

    def evaluate_from_report(
        self,
        hallucination_report: dict,
        latency_ms: float,
    ) -> EvaluationMetrics:
        """
        Compute metrics directly from a hallucination detection report (without ground truth).
        
        This is used when ground truth is not available. Metrics are self-reported.
        
        Args:
            hallucination_report: Dict with sentence_results containing labels and confidences.
            latency_ms: Processing latency.
            
        Returns:
            EvaluationMetrics based on system's own predictions.
        """
        sentence_results = hallucination_report.get("sentence_results", [])

        if not sentence_results:
            return self._empty_metrics(latency_ms)

        labels = [r.get("label", "NOT_ENOUGH_EVIDENCE") for r in sentence_results]
        confidences = [r.get("confidence", 0.0) for r in sentence_results]

        supported = sum(1 for l in labels if l == "SUPPORTED")
        contradicted = sum(1 for l in labels if l == "CONTRADICTED")
        neutral = sum(1 for l in labels if l == "NOT_ENOUGH_EVIDENCE")
        total = len(labels)

        # Hallucination rate = contradicted / total (consistent with hallucination_detection_agent)
        # NEUTRAL is NOT counted as hallucination.
        hallucination_rate = contradicted / total if total > 0 else 0.0
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        # Without ground truth, precision/recall are estimated from confidence
        high_conf_correct = sum(1 for c in confidences if c > 0.7)
        estimated_precision = high_conf_correct / total if total > 0 else 0.0
        estimated_recall = estimated_precision  # Symmetric estimate without ground truth
        estimated_f1 = (
            2 * estimated_precision * estimated_recall / (estimated_precision + estimated_recall)
            if (estimated_precision + estimated_recall) > 0 else 0.0
        )

        return EvaluationMetrics(
            precision=round(estimated_precision, 4),
            recall=round(estimated_recall, 4),
            f1_score=round(estimated_f1, 4),
            accuracy=round(avg_confidence, 4),
            hallucination_rate=round(hallucination_rate, 4),
            latency_ms=round(latency_ms, 2),
            retrieval_precision=0.0,
            retrieval_recall=0.0,
            num_claims_verified=total,
            num_claims_supported=supported,
            num_claims_contradicted=contradicted,
            num_claims_neutral=neutral,
            average_confidence=round(avg_confidence, 4),
        )

    def benchmark_comparison(
        self,
        system_results: Dict[str, List[EvaluationMetrics]],
    ) -> List[BenchmarkResult]:
        """
        Compare multiple systems based on their evaluation metrics.
        
        Args:
            system_results: Dict mapping system name to list of metric results.
            
        Returns:
            List of BenchmarkResult for each system.
        """
        benchmarks = []
        for system_name, metrics_list in system_results.items():
            if not metrics_list:
                continue

            avg_latency = float(np.mean([m.latency_ms for m in metrics_list]))
            avg_hallucination = float(np.mean([m.hallucination_rate for m in metrics_list]))
            avg_f1 = float(np.mean([m.f1_score for m in metrics_list]))

            benchmarks.append(BenchmarkResult(
                system_name=system_name,
                metrics=metrics_list[-1],  # Latest metrics
                num_queries=len(metrics_list),
                avg_latency_ms=round(avg_latency, 2),
                avg_hallucination_rate=round(avg_hallucination, 4),
                avg_f1=round(avg_f1, 4),
            ))

        # Sort by F1 score (higher is better)
        benchmarks.sort(key=lambda x: x.avg_f1, reverse=True)
        return benchmarks

    def get_history(self) -> List[EvaluationMetrics]:
        """Return the evaluation history."""
        return self._benchmark_history

    def _empty_metrics(self, latency_ms: float) -> EvaluationMetrics:
        """Return empty metrics when no data is available."""
        return EvaluationMetrics(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            accuracy=0.0,
            hallucination_rate=0.0,
            latency_ms=latency_ms,
            retrieval_precision=0.0,
            retrieval_recall=0.0,
            num_claims_verified=0,
            num_claims_supported=0,
            num_claims_contradicted=0,
            num_claims_neutral=0,
            average_confidence=0.0,
        )

    @staticmethod
    def measure_latency(func):
        """Decorator to measure function execution time in milliseconds."""
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            logger.debug(f"{func.__name__} took {latency_ms:.2f}ms")
            return result, latency_ms
        return wrapper
