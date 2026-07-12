"""
FastAPI Routes
Defines all REST API endpoints for the hallucination detection system.
"""

import os
import hashlib
import tempfile
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from loguru import logger

from api.schemas import (
    QueryRequest,
    QueryResponse,
    UploadRequest,
    UploadResponse,
    URLIngestRequest,
    VerificationRequest,
    VerificationResponse,
    EvaluationResponse,
    HealthResponse,
    SentenceAnalysis,
    HallucinationReportSchema,
    VerificationReportSchema,
    VerifiedClaimSchema,
    CitationSchema,
    MetricsSchema,
)
from api.dependencies import get_orchestrator
from orchestrator import MultiAgentOrchestrator

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and index status."""
    try:
        orchestrator = get_orchestrator()
        return HealthResponse(
            status="healthy",
            index_size=orchestrator.index_size,
            message="System is operational",
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            index_size=0,
            message=f"System error: {str(e)}",
        )


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query through the full multi-agent pipeline.
    
    - Analyzes the query
    - Retrieves relevant evidence
    - Detects hallucinations in the provided LLM response
    - Verifies facts
    - Generates a corrected response
    - Returns evaluation metrics
    """
    orchestrator = get_orchestrator()

    if orchestrator.index_size == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents in the knowledge base. Please upload documents first.",
        )

    try:
        result = orchestrator.run(
            query=request.query,
            llm_response=request.llm_response,
        )

        # Build response
        hallucination_report = None
        if result.get("hallucination_report"):
            hr = result["hallucination_report"]
            hallucination_report = HallucinationReportSchema(
                total_sentences=hr.get("total_sentences", 0),
                supported_count=hr.get("supported_count", 0),
                contradicted_count=hr.get("contradicted_count", 0),
                neutral_count=hr.get("neutral_count", 0),
                hallucination_rate=hr.get("hallucination_rate", 0.0),
                overall_confidence=hr.get("overall_confidence", 0.0),
                sentence_results=[
                    SentenceAnalysis(
                        sentence=sr["sentence"],
                        label=sr["label"],
                        confidence=sr["confidence"],
                        supporting_evidence=sr.get("supporting_evidence", ""),
                        scores=sr.get("scores", {}),
                    )
                    for sr in hr.get("sentence_results", [])
                ],
            )

        verification_report = None
        if result.get("verification_report"):
            vr = result["verification_report"]
            verification_report = VerificationReportSchema(
                total_claims=vr.get("total_claims", 0),
                verified_count=vr.get("verified_count", 0),
                refuted_count=vr.get("refuted_count", 0),
                unverifiable_count=vr.get("unverifiable_count", 0),
                overall_accuracy=vr.get("overall_accuracy", 0.0),
                average_confidence=vr.get("average_confidence", 0.0),
            )

        verified_claims = [
            VerifiedClaimSchema(
                claim=vc.get("claim", ""),
                verdict=vc.get("verdict", "UNVERIFIABLE"),
                confidence=vc.get("confidence", 0.0),
                supporting_evidence=vc.get("supporting_evidence", []),
                contradicting_evidence=vc.get("contradicting_evidence", []),
            )
            for vc in result.get("verified_claims", [])
        ]

        citations = [
            CitationSchema(id=c["id"], text=c["text"], marker=c["marker"])
            for c in result.get("citations", [])
        ]

        metrics = None
        if result.get("metrics"):
            m = result["metrics"]
            metrics = MetricsSchema(
                precision=m.get("precision", 0.0),
                recall=m.get("recall", 0.0),
                f1_score=m.get("f1_score", 0.0),
                accuracy=m.get("accuracy", 0.0),
                hallucination_rate=m.get("hallucination_rate", 0.0),
                latency_ms=m.get("latency_ms", 0.0),
                retrieval_precision=m.get("retrieval_precision", 0.0),
                num_claims_verified=m.get("num_claims_verified", 0),
                average_confidence=m.get("average_confidence", 0.0),
            )

        return QueryResponse(
            query=result.get("query", request.query),
            rewritten_query=result.get("rewritten_query", ""),
            corrected_response=result.get("corrected_response", ""),
            citations=citations,
            hallucination_report=hallucination_report,
            verification_report=verification_report,
            verified_claims=verified_claims,
            metrics=metrics,
            ranked_evidence=result.get("ranked_evidence", []),
            errors=result.get("errors", []),
        )

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/upload/file", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF or text file to the knowledge base.
    Supports: .pdf, .txt, .md, .html
    """
    orchestrator = get_orchestrator()

    # Validate file extension
    allowed_extensions = {".pdf", ".txt", ".md", ".html", ".htm"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}",
        )

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Ingest the document
        num_chunks = orchestrator.ingest_document(tmp_path)

        # Save the updated index
        orchestrator.save_index()

        # Cleanup temp file
        os.unlink(tmp_path)

        return UploadResponse(
            message=f"Successfully ingested '{file.filename}'",
            num_chunks=num_chunks,
            source=file.filename,
        )

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/upload/text", response_model=UploadResponse)
async def upload_text(request: UploadRequest):
    """Upload raw text directly to the knowledge base."""
    orchestrator = get_orchestrator()

    try:
        num_chunks = orchestrator.ingest_text(request.text, request.source_name)
        orchestrator.save_index()

        return UploadResponse(
            message=f"Successfully ingested text from '{request.source_name}'",
            num_chunks=num_chunks,
            source=request.source_name,
        )

    except Exception as e:
        logger.error(f"Text upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/upload/url", response_model=UploadResponse)
async def upload_url(request: URLIngestRequest):
    """Fetch and ingest a web page into the knowledge base."""
    orchestrator = get_orchestrator()

    try:
        num_chunks = orchestrator.ingest_url(request.url)
        orchestrator.save_index()

        return UploadResponse(
            message=f"Successfully ingested URL: {request.url}",
            num_chunks=num_chunks,
            source=request.url,
        )

    except Exception as e:
        logger.error(f"URL ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"URL ingestion failed: {str(e)}")


@router.post("/verify", response_model=VerificationResponse)
async def verify_text(request: VerificationRequest):
    """
    Standalone fact verification endpoint.
    Verifies text against provided evidence without using the knowledge base.
    """
    orchestrator = get_orchestrator()

    try:
        report = orchestrator.fact_agent.verify(request.text, request.evidence)

        verified_claims = [
            VerifiedClaimSchema(
                claim=vc.claim.text,
                verdict=vc.verdict,
                confidence=vc.confidence,
                supporting_evidence=vc.supporting_evidence[:2],
                contradicting_evidence=vc.contradicting_evidence[:2],
            )
            for vc in report.claims
        ]

        return VerificationResponse(
            verified_claims=verified_claims,
            total_claims=report.total_claims,
            verified_count=report.verified_count,
            refuted_count=report.refuted_count,
            overall_accuracy=report.overall_accuracy,
        )

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@router.get("/evaluations", response_model=EvaluationResponse)
async def get_evaluations():
    """Get evaluation history."""
    orchestrator = get_orchestrator()

    try:
        history = orchestrator.evaluation_agent.get_history()
        evaluations = [
            {
                "precision": m.precision,
                "recall": m.recall,
                "f1_score": m.f1_score,
                "hallucination_rate": m.hallucination_rate,
                "latency_ms": m.latency_ms,
                "num_claims": m.num_claims_verified,
            }
            for m in history
        ]

        return EvaluationResponse(
            evaluations=evaluations,
            total_count=len(evaluations),
        )

    except Exception as e:
        logger.error(f"Evaluation retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/stats")
async def index_stats():
    """Get vector store statistics."""
    orchestrator = get_orchestrator()
    return {
        "index_size": orchestrator.index_size,
        "embedding_model": orchestrator.config.models.embedding_model,
        "chunk_size": orchestrator.config.vector_store.chunk_size,
    }
