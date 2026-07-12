"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import router

app = FastAPI(
    title="Hallucination Detection & Fact Verification API",
    description="Multi-Agent RAG Framework for detecting hallucinations and verifying facts in LLM responses.",
    version="1.0.0",
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    """Initialize components on application startup."""
    logger.info("FastAPI application starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("FastAPI application shutting down...")
