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
    orchestrator = MultiAgentOrchestrator(config=config, use_llm=False)

    # Try to load existing index
    if orchestrator.load_index():
        logger.info(f"Loaded existing index with {orchestrator.index_size} vectors")
    else:
        logger.info("No existing index found. Starting with empty vector store.")

    return orchestrator
