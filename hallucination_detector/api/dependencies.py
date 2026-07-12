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
    Starts with an EMPTY vector store — no persistent data.
    """
    logger.info("Creating MultiAgentOrchestrator instance...")
    config = get_config()

    hf_token = os.environ.get("HF_TOKEN", "")
    # Ignore placeholder values
    if hf_token in ("", "your_token_here"):
        use_llm = False
        logger.warning("HF_TOKEN not set or is placeholder. Using extractive mode.")
        logger.warning("Set a real HF_TOKEN in .env for proper LLM responses.")
    else:
        use_llm = True
        logger.info(f"HF_TOKEN found. LLM generation enabled.")

    orchestrator = MultiAgentOrchestrator(config=config, use_llm=use_llm)
    logger.info(f"LLM active: {orchestrator.response_agent.use_llm}")
    logger.info("Starting with empty vector store. Upload a document to begin.")

    return orchestrator
