"""
Database Module
SQLite database for storing documents, evaluation results, and session history.
Uses SQLAlchemy for ORM and async support.
"""

import os
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from loguru import logger

Base = declarative_base()


class DocumentRecord(Base):
    """Database record for ingested documents."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(500), nullable=False)
    doc_type = Column(String(50), nullable=False)
    content_hash = Column(String(64), unique=True)
    num_chunks = Column(Integer, default=0)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


class EvaluationRecord(Base):
    """Database record for evaluation results."""
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    original_response = Column(Text)
    corrected_response = Column(Text)
    hallucination_rate = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    latency_ms = Column(Float)
    num_claims = Column(Integer)
    num_supported = Column(Integer)
    num_contradicted = Column(Integer)
    num_neutral = Column(Integer)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


class QueryLog(Base):
    """Database record for query history."""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    rewritten_query = Column(Text)
    num_results_retrieved = Column(Integer)
    num_results_reranked = Column(Integer)
    response = Column(Text)
    latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """
    Manages database connections and CRUD operations.
    Uses SQLite for simplicity and portability.
    """

    def __init__(self, database_url: str = "sqlite:///./data/hallucination_detector.db"):
        """
        Initialize the database manager.
        
        Args:
            database_url: SQLAlchemy database URL.
        """
        # Ensure directory exists for SQLite
        if database_url.startswith("sqlite"):
            db_path = database_url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized: {database_url}")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def add_document_record(
        self,
        source: str,
        doc_type: str,
        content_hash: str,
        num_chunks: int,
        metadata: dict = None,
    ) -> int:
        """
        Record a newly ingested document.
        
        Args:
            source: Document source path or URL.
            doc_type: Type of document (pdf, text, web).
            content_hash: SHA256 hash of the content.
            num_chunks: Number of chunks created.
            metadata: Additional metadata.
            
        Returns:
            ID of the created record.
        """
        session = self.get_session()
        try:
            record = DocumentRecord(
                source=source,
                doc_type=doc_type,
                content_hash=content_hash,
                num_chunks=num_chunks,
                metadata_json=metadata or {},
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.id
        finally:
            session.close()

    def add_evaluation_record(
        self,
        query: str,
        original_response: str,
        corrected_response: str,
        hallucination_rate: float,
        precision_score: float,
        recall_score: float,
        f1_score: float,
        latency_ms: float,
        num_claims: int,
        num_supported: int,
        num_contradicted: int,
        num_neutral: int,
        metadata: dict = None,
    ) -> int:
        """
        Record evaluation metrics for a verification run.
        
        Returns:
            ID of the created record.
        """
        session = self.get_session()
        try:
            record = EvaluationRecord(
                query=query,
                original_response=original_response,
                corrected_response=corrected_response,
                hallucination_rate=hallucination_rate,
                precision_score=precision_score,
                recall_score=recall_score,
                f1_score=f1_score,
                latency_ms=latency_ms,
                num_claims=num_claims,
                num_supported=num_supported,
                num_contradicted=num_contradicted,
                num_neutral=num_neutral,
                metadata_json=metadata or {},
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.id
        finally:
            session.close()

    def log_query(
        self,
        query: str,
        rewritten_query: str,
        num_results_retrieved: int,
        num_results_reranked: int,
        response: str,
        latency_ms: float,
    ) -> int:
        """
        Log a query for analysis.
        
        Returns:
            ID of the created log entry.
        """
        session = self.get_session()
        try:
            record = QueryLog(
                query=query,
                rewritten_query=rewritten_query,
                num_results_retrieved=num_results_retrieved,
                num_results_reranked=num_results_reranked,
                response=response,
                latency_ms=latency_ms,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.id
        finally:
            session.close()

    def get_evaluation_history(self, limit: int = 50) -> List[dict]:
        """
        Retrieve recent evaluation records.
        
        Args:
            limit: Maximum number of records to return.
            
        Returns:
            List of evaluation records as dictionaries.
        """
        session = self.get_session()
        try:
            records = (
                session.query(EvaluationRecord)
                .order_by(EvaluationRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "query": r.query,
                    "hallucination_rate": r.hallucination_rate,
                    "precision": r.precision_score,
                    "recall": r.recall_score,
                    "f1": r.f1_score,
                    "latency_ms": r.latency_ms,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in records
            ]
        finally:
            session.close()

    def get_document_count(self) -> int:
        """Return total number of ingested documents."""
        session = self.get_session()
        try:
            return session.query(DocumentRecord).count()
        finally:
            session.close()

    def document_exists(self, content_hash: str) -> bool:
        """Check if a document with the given hash already exists."""
        session = self.get_session()
        try:
            return session.query(DocumentRecord).filter_by(content_hash=content_hash).first() is not None
        finally:
            session.close()
