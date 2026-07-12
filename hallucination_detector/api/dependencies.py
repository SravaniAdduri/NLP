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
    use_llm = False

    # Only enable LLM if token is valid and not placeholder
    if hf_token and hf_token not in ("your_token_here", ""):
        # Test if HF API is reachable
        try:
            import requests
            resp = requests.get("https://huggingface.co/api/models", timeout=5)
            if resp.status_code == 200:
                use_llm = True
                logger.info("HF API reachable. LLM generation enabled.")
        except Exception:
            logger.warning("HF API not reachable (network restricted). Using extractive mode.")

    if not use_llm:
        logger.info("Running in OFFLINE extractive mode (no external API needed).")

    orchestrator = MultiAgentOrchestrator(config=config, use_llm=use_llm)
    logger.info(f"LLM active: {orchestrator.response_agent.use_llm}")
    logger.info("Ready. Upload a document to begin.")

    return orchestrator
