"""
End-to-end integration test demonstrating the full pipeline.

Run this script to verify the entire system works correctly.

Usage:
    cd hallucination_detector
    python scripts/run_demo.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from config.settings import get_config
from orchestrator import MultiAgentOrchestrator


def run_demo():
    """Run a complete demonstration of the hallucination detection pipeline."""

    logger.info("=" * 70)
    logger.info("HALLUCINATION DETECTION DEMO")
    logger.info("=" * 70)

    # Initialize the orchestrator
    logger.info("\n[1/5] Initializing the Multi-Agent Orchestrator...")
    config = get_config()
    orchestrator = MultiAgentOrchestrator(config=config, use_llm=False)

    # Ingest sample knowledge base
    logger.info("\n[2/5] Ingesting knowledge base...")
    sample_knowledge = [
        (
            "Climate change is primarily caused by human activities, particularly the burning of "
            "fossil fuels such as coal, oil, and natural gas. These activities release greenhouse "
            "gases, mainly carbon dioxide (CO2) and methane (CH4), into the atmosphere. The "
            "Intergovernmental Panel on Climate Change (IPCC) has confirmed with high confidence "
            "that human influence has warmed the climate at a rate unprecedented in at least the "
            "last 2,000 years. Global average temperatures have risen by approximately 1.1°C "
            "above pre-industrial levels."
        ),
        (
            "Photosynthesis is the process by which green plants, algae, and some bacteria convert "
            "light energy into chemical energy stored in glucose. It takes place primarily in the "
            "chloroplasts of plant cells, using chlorophyll to absorb light. The overall equation "
            "is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. This process is fundamental to "
            "life on Earth as it produces oxygen and forms the base of most food chains."
        ),
        (
            "The internet is a global network of interconnected computer networks that uses the "
            "TCP/IP protocol suite. Data is transmitted as packets that are routed independently "
            "through a distributed network of routers. There is no single centralized server; "
            "instead, the internet operates as a decentralized system. The Domain Name System "
            "(DNS) translates human-readable domain names like 'example.com' into numerical IP "
            "addresses that computers use to identify each other."
        ),
    ]

    for i, text in enumerate(sample_knowledge):
        chunks = orchestrator.ingest_text(text, f"knowledge_base_{i+1}")
        logger.info(f"  Ingested document {i+1}: {chunks} chunks")

    logger.info(f"  Total index size: {orchestrator.index_size} vectors")

    # Test query with hallucinated response
    logger.info("\n[3/5] Testing hallucination detection...")
    query = "What causes climate change?"
    hallucinated_response = (
        "Climate change is primarily caused by volcanic eruptions and solar flares. "
        "Human activities have no significant impact on global warming. "
        "The Earth has been cooling for the past 50 years according to all scientific studies."
    )

    logger.info(f"  Query: {query}")
    logger.info(f"  LLM Response (to check): {hallucinated_response}")

    start_time = time.perf_counter()
    result = orchestrator.run(query=query, llm_response=hallucinated_response)
    total_time = (time.perf_counter() - start_time) * 1000

    # Display results
    logger.info("\n[4/5] Results:")
    logger.info("-" * 50)

    # Query analysis
    qa = result.get("query_analysis")
    if qa:
        logger.info(f"  Intent: {qa.get('intent')}")
        logger.info(f"  Entities: {qa.get('entities')}")
        logger.info(f"  Rewritten Query: {result.get('rewritten_query')}")

    # Retrieval
    logger.info(f"\n  Retrieved: {result.get('num_retrieved', 0)} documents")
    logger.info(f"  After ranking: {result.get('num_ranked', 0)} evidence passages")

    # Hallucination report
    hr = result.get("hallucination_report")
    if hr:
        logger.info(f"\n  === HALLUCINATION REPORT ===")
        logger.info(f"  Hallucination Rate: {hr.get('hallucination_rate', 0):.2%}")
        logger.info(f"  Supported: {hr.get('supported_count', 0)}")
        logger.info(f"  Contradicted: {hr.get('contradicted_count', 0)}")
        logger.info(f"  Not Enough Evidence: {hr.get('neutral_count', 0)}")
        logger.info(f"\n  Sentence-Level Analysis:")
        for sr in hr.get("sentence_results", []):
            icon = {"SUPPORTED": "✓", "CONTRADICTED": "✗", "NOT_ENOUGH_EVIDENCE": "?"}.get(sr["label"], "?")
            logger.info(f"    [{icon}] {sr['label']} ({sr['confidence']:.2f}): {sr['sentence'][:70]}...")

    # Verification report
    vr = result.get("verification_report")
    if vr:
        logger.info(f"\n  === FACT VERIFICATION ===")
        logger.info(f"  Total Claims: {vr.get('total_claims', 0)}")
        logger.info(f"  Verified: {vr.get('verified_count', 0)}")
        logger.info(f"  Refuted: {vr.get('refuted_count', 0)}")
        logger.info(f"  Overall Accuracy: {vr.get('overall_accuracy', 0):.2%}")

    # Corrected response
    logger.info(f"\n  === CORRECTED RESPONSE ===")
    logger.info(f"  {result.get('corrected_response', 'N/A')[:200]}")

    # Metrics
    metrics = result.get("metrics")
    if metrics:
        logger.info(f"\n  === METRICS ===")
        logger.info(f"  Hallucination Rate: {metrics.get('hallucination_rate', 0):.3f}")
        logger.info(f"  Average Confidence: {metrics.get('average_confidence', 0):.3f}")
        logger.info(f"  Latency: {metrics.get('latency_ms', 0):.0f} ms")

    # Errors
    errors = result.get("errors", [])
    if errors:
        logger.warning(f"\n  Warnings: {errors}")

    logger.info(f"\n[5/5] Total pipeline time: {total_time:.0f} ms")
    logger.info("=" * 70)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 70)

    return result


if __name__ == "__main__":
    run_demo()
