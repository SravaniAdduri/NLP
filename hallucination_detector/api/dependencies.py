"""API Dependencies - Manages shared state and dependency injection for FastAPI routes."""

import os
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
from loguru import logger

from orchestrator import MultiAgentOrchestrator
from config.settings import get_config

# Explicitly load .env from the project root
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path, override=True)


@lru_cache(maxsize=1)
def get_orchestrator() -> MultiAgentOrchestrator:
    """
    Get or create the singleton orchestrator instance.
    Runs fully offline — no external API calls for generation.
    Uses extractive mode with cross-encoder reranking for best results.
    """
    logger.info("Creating MultiAgentOrchestrator instance...")
    config = get_config()

    # Always run offline — extractive mode with high-quality reranking
    orchestrator = MultiAgentOrchestrator(config=config, use_llm=False)
    logger.info("System ready (offline extractive mode). Upload a document to begin.")

    return orchestrator
