"""
API Dependencies
Manages shared state and dependency injection for FastAPI routes.
"""

from functools import lru_cache
from loguru import logger

from orchestrator import MultiAgentOrchestrator
from config.settings import get_config


@lru_cache(maxsize=1)
def get_orchestrator() -> MultiAgentOrchestrator:
    """
    Get or create the singleton orchestrator instance.
    Uses LRU cache to ensure only one instance exists.
    """
    logger.info("Creating MultiAgentOrchestrator instance...")
    config = get_config()

    # Enable LLM if HF_TOKEN is set
    import os
    use_llm = bool(os.environ.get("HF_TOKEN", ""))
    if not use_llm:
        logger.warning("HF_TOKEN not set. Using extractive mode. Set HF_TOKEN in .env for LLM responses.")

    orchestrator = MultiAgentOrchestrator(config=config, use_llm=use_llm)

    # Try to load existing index
    if orchestrator.load_index():
        logger.info(f"Loaded existing index with {orchestrator.index_size} vectors")
    else:
        logger.info("No existing index found. Starting with empty vector store.")

    return orchestrator
