#!/usr/bin/env python
"""
Document Ingestion Script
Processes documents from the data/raw directory and adds them to the vector store.

Usage:
    python scripts/ingest_documents.py --dir data/raw
    python scripts/ingest_documents.py --file data/raw/paper.pdf
    python scripts/ingest_documents.py --url https://example.com/article
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from config.settings import get_config
from core.embeddings import EmbeddingEngine
from core.vector_store import FAISSVectorStore
from core.chunking import TextChunker
from core.document_processor import DocumentProcessor
from agents.retrieval_agent import RetrievalAgent


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument("--dir", type=str, help="Directory containing documents to ingest")
    parser.add_argument("--file", type=str, help="Single file to ingest")
    parser.add_argument("--url", type=str, help="URL to fetch and ingest")
    parser.add_argument("--text", type=str, help="Raw text to ingest")
    parser.add_argument("--index-name", type=str, default="default", help="Name for the vector index")
    args = parser.parse_args()

    if not any([args.dir, args.file, args.url, args.text]):
        parser.error("At least one of --dir, --file, --url, or --text is required")

    config = get_config()

    # Initialize components
    logger.info("Initializing embedding engine...")
    embedding_engine = EmbeddingEngine(model_name=config.models.embedding_model)

    vector_store = FAISSVectorStore(
        embedding_dimension=embedding_engine.get_dimension(),
        store_path=config.vector_store.vector_store_path,
    )

    # Try to load existing index
    vector_store.load(args.index_name)
    logger.info(f"Current index size: {vector_store.size} vectors")

    chunker = TextChunker(
        chunk_size=config.vector_store.chunk_size,
        chunk_overlap=config.vector_store.chunk_overlap,
    )
    document_processor = DocumentProcessor()

    retrieval_agent = RetrievalAgent(
        embedding_engine=embedding_engine,
        vector_store=vector_store,
        chunker=chunker,
        document_processor=document_processor,
        top_k=config.vector_store.top_k_retrieval,
    )

    # Process inputs
    total_chunks = 0

    if args.dir:
        logger.info(f"Ingesting directory: {args.dir}")
        total_chunks += retrieval_agent.ingest_directory(args.dir)

    if args.file:
        logger.info(f"Ingesting file: {args.file}")
        total_chunks += retrieval_agent.ingest_document(args.file)

    if args.url:
        logger.info(f"Ingesting URL: {args.url}")
        total_chunks += retrieval_agent.ingest_url(args.url)

    if args.text:
        logger.info("Ingesting raw text")
        total_chunks += retrieval_agent.ingest_text(args.text, "cli_input")

    # Save the index
    retrieval_agent.save_index(args.index_name)
    logger.info(f"Ingestion complete. Added {total_chunks} chunks. Total index size: {vector_store.size}")


if __name__ == "__main__":
    main()
