"""
Benchmark module for comparing the multi-agent system against baseline RAG.
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from evaluation.metrics import (
    compute_precision_recall_f1,
    compute_accuracy,
    compute_hallucination_rate,
    compute_ndcg_at_k,
    compute_latency_percentiles,
)


@dataclass
class BenchmarkSample:
    """A single benchmark sample with query, response, and ground truth."""
    query: str
    llm_response: str
    ground_truth_labels: List[str]  # Per-sentence labels
    relevant_evidence: List[str]


@dataclass
class SystemResult:
    """Results from running a system on a benchmark sample."""
    predicted_labels: List[str]
    latency_ms: float
    retrieval_relevance: List[float]
    confidence_scores: List[float]


@dataclass
class BenchmarkReport:
    """Complete benchmark comparison report."""
    system_name: str
    num_samples: int
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_accuracy: float
    avg_hallucination_rate: float
    avg_ndcg: float
    latency_stats: Dict[str, float]
    per_sample_results: List[Dict]


class BenchmarkRunner:
    """
    Runs benchmark comparisons between systems.
    
    Compares:
    - Multi-agent system (full pipeline)
    - Baseline RAG (simple retrieval + generation)
    """

    def __init__(self):
        """Initialize the benchmark runner."""
        self.results: Dict[str, List[SystemResult]] = {}
        logger.info("Benchmark runner initialized")

    def create_benchmark_dataset(self) -> List[BenchmarkSample]:
        """
        Create a sample benchmark dataset for testing.
        
        Returns:
            List of BenchmarkSample objects.
        """
        samples = [
            BenchmarkSample(
                query="What causes climate change?",
                llm_response=(
                    "Climate change is primarily caused by volcanic eruptions and solar flares. "
                    "Human activities like burning fossil fuels release greenhouse gases. "
                    "The Earth has been cooling for the past century."
                ),
                ground_truth_labels=["CONTRADICTED", "SUPPORTED", "CONTRADICTED"],
                relevant_evidence=[
                    "Climate change is primarily caused by human activities, particularly the burning of fossil fuels.",
                    "Greenhouse gases like CO2 and methane trap heat in the atmosphere.",
                    "Global average temperatures have risen by about 1.1°C since pre-industrial times.",
                ],
            ),
            BenchmarkSample(
                query="What is photosynthesis?",
                llm_response=(
                    "Photosynthesis is the process by which plants convert sunlight into chemical energy. "
                    "It occurs in the mitochondria of plant cells. "
                    "The process produces oxygen as a byproduct."
                ),
                ground_truth_labels=["SUPPORTED", "CONTRADICTED", "SUPPORTED"],
                relevant_evidence=[
                    "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
                    "Photosynthesis takes place in the chloroplasts of plant cells.",
                    "During photosynthesis, plants absorb CO2 and release oxygen.",
                ],
            ),
            BenchmarkSample(
                query="How does the internet work?",
                llm_response=(
                    "The internet uses TCP/IP protocols to transmit data. "
                    "Data is sent as packets through a single centralized server. "
                    "DNS translates domain names to IP addresses."
                ),
                ground_truth_labels=["SUPPORTED", "CONTRADICTED", "SUPPORTED"],
                relevant_evidence=[
                    "The internet is a global network that uses TCP/IP protocols.",
                    "Data on the internet is broken into packets and routed through a distributed network of servers.",
                    "The Domain Name System (DNS) translates human-readable domain names into IP addresses.",
                ],
            ),
        ]
        return samples

    def run_benchmark(
        self,
        system_name: str,
        samples: List[BenchmarkSample],
        system_results: List[SystemResult],
    ) -> BenchmarkReport:
        """
        Run benchmark evaluation for a system.
        
        Args:
            system_name: Name of the system being benchmarked.
            samples: Benchmark samples with ground truth.
            system_results: System outputs for each sample.
            
        Returns:
            BenchmarkReport with aggregated metrics.
        """
        if len(samples) != len(system_results):
            raise ValueError("Number of samples and results must match")

        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []
        hallucination_rates = []
        ndcg_scores = []
        latencies = []
        per_sample = []

        for sample, result in zip(samples, system_results):
            # Compute per-sample metrics
            min_len = min(len(result.predicted_labels), len(sample.ground_truth_labels))
            pred = result.predicted_labels[:min_len]
            truth = sample.ground_truth_labels[:min_len]

            p, r, f1 = compute_precision_recall_f1(pred, truth)
            acc = compute_accuracy(pred, truth)
            h_rate = compute_hallucination_rate(pred)
            ndcg = compute_ndcg_at_k(result.retrieval_relevance) if result.retrieval_relevance else 0.0

            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)
            accuracies.append(acc)
            hallucination_rates.append(h_rate)
            ndcg_scores.append(ndcg)
            latencies.append(result.latency_ms)

            per_sample.append({
                "query": sample.query,
                "precision": p,
                "recall": r,
                "f1": f1,
                "accuracy": acc,
                "hallucination_rate": h_rate,
                "latency_ms": result.latency_ms,
            })

        latency_stats = compute_latency_percentiles(latencies)

        report = BenchmarkReport(
            system_name=system_name,
            num_samples=len(samples),
            avg_precision=float(np.mean(precisions)),
            avg_recall=float(np.mean(recalls)),
            avg_f1=float(np.mean(f1_scores)),
            avg_accuracy=float(np.mean(accuracies)),
            avg_hallucination_rate=float(np.mean(hallucination_rates)),
            avg_ndcg=float(np.mean(ndcg_scores)),
            latency_stats=latency_stats,
            per_sample_results=per_sample,
        )

        logger.info(
            f"Benchmark '{system_name}': F1={report.avg_f1:.3f}, "
            f"Accuracy={report.avg_accuracy:.3f}, "
            f"Avg Latency={latency_stats['avg']:.0f}ms"
        )

        return report

    def compare_systems(self, reports: List[BenchmarkReport]) -> Dict:
        """
        Compare multiple system benchmark reports.
        
        Args:
            reports: List of BenchmarkReport from different systems.
            
        Returns:
            Comparison summary dict.
        """
        comparison = {
            "systems": [],
            "best_f1": None,
            "best_accuracy": None,
            "fastest": None,
        }

        best_f1_score = -1
        best_acc = -1
        fastest_latency = float("inf")

        for report in reports:
            system_summary = {
                "name": report.system_name,
                "f1": report.avg_f1,
                "accuracy": report.avg_accuracy,
                "hallucination_rate": report.avg_hallucination_rate,
                "avg_latency_ms": report.latency_stats["avg"],
                "ndcg": report.avg_ndcg,
            }
            comparison["systems"].append(system_summary)

            if report.avg_f1 > best_f1_score:
                best_f1_score = report.avg_f1
                comparison["best_f1"] = report.system_name

            if report.avg_accuracy > best_acc:
                best_acc = report.avg_accuracy
                comparison["best_accuracy"] = report.system_name

            if report.latency_stats["avg"] < fastest_latency:
                fastest_latency = report.latency_stats["avg"]
                comparison["fastest"] = report.system_name

        return comparison
