"""
Pydantic schemas for API request/response validation.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for querying the system."""
    query: str = Field(..., min_length=1, max_length=2000, description="User's question")
    llm_response: str = Field(default="", description="LLM-generated response to verify")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of evidence passages to retrieve")


class SentenceAnalysis(BaseModel):
    """Analysis result for a single sentence."""
    sentence: str
    label: str
    confidence: float
    supporting_evidence: str
    scores: Dict[str, float]


class HallucinationReportSchema(BaseModel):
    """Hallucination detection report."""
    total_sentences: int
    supported_count: int
    contradicted_count: int
    neutral_count: int
    hallucination_rate: float
    overall_confidence: float
    sentence_results: List[SentenceAnalysis]


class VerifiedClaimSchema(BaseModel):
    """A verified claim with verdict."""
    claim: str
    verdict: str
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]


class VerificationReportSchema(BaseModel):
    """Fact verification report."""
    total_claims: int
    verified_count: int
    refuted_count: int
    unverifiable_count: int
    overall_accuracy: float
    average_confidence: float


class CitationSchema(BaseModel):
    """Citation reference."""
    id: int
    text: str
    marker: str


class MetricsSchema(BaseModel):
    """Evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    hallucination_rate: float
    latency_ms: float
    retrieval_precision: float
    num_claims_verified: int
    average_confidence: float


class QueryResponse(BaseModel):
    """Response schema for query results."""
    query: str
    rewritten_query: str
    corrected_response: str
    citations: List[CitationSchema]
    hallucination_report: Optional[HallucinationReportSchema] = None
    verification_report: Optional[VerificationReportSchema] = None
    verified_claims: List[VerifiedClaimSchema]
    metrics: Optional[MetricsSchema] = None
    ranked_evidence: List[str]
    generated_evidence: List[str] = Field(default_factory=list)
    errors: List[str]


class UploadRequest(BaseModel):
    """Request schema for text upload."""
    text: str = Field(..., min_length=1, description="Text content to ingest")
    source_name: str = Field(default="direct_input", description="Source identifier")


class UploadResponse(BaseModel):
    """Response schema for document upload."""
    message: str
    num_chunks: int
    source: str


class URLIngestRequest(BaseModel):
    """Request schema for URL ingestion."""
    url: str = Field(..., description="URL to fetch and ingest")


class VerificationRequest(BaseModel):
    """Request for standalone fact verification."""
    text: str = Field(..., description="Text to verify")
    evidence: List[str] = Field(..., description="Evidence passages")


class VerificationResponse(BaseModel):
    """Response for standalone verification."""
    verified_claims: List[VerifiedClaimSchema]
    total_claims: int
    verified_count: int
    refuted_count: int
    overall_accuracy: float


class EvaluationResponse(BaseModel):
    """Response for evaluation history."""
    evaluations: List[Dict]
    total_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    index_size: int
    message: str
