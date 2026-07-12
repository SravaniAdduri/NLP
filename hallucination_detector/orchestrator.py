"""
LangGraph Orchestrator
Defines the multi-agent workflow as a directed graph.
Manages state transitions and data flow between agents.

Graph Nodes:
- query_understanding: Analyze and rewrite the user query
- retrieval: Retrieve relevant documents
- evidence_ranking: Rerank and filter evidence
- hallucination_detection: Detect hallucinations in LLM response
- fact_verification: Verify individual claims
- response_generation: Generate corrected response
- evaluation: Compute metrics

Graph Edges:
- query_understanding -> retrieval
- retrieval -> evidence_ranking
- evidence_ranking -> hallucination_detection
- hallucination_detection -> fact_verification
- fact_verification -> response_generation
- response_generation -> evaluation
"""

import time
from typing import TypedDict, List, Optional, Annotated
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END
from loguru import logger

from config.settings import get_config, AppConfig
from core.embeddings import EmbeddingEngine
from core.vector_store import FAISSVectorStore
from core.chunking import TextChunker
from core.document_processor import DocumentProcessor
from core.reranker import CrossEncoderReranker
from core.nli_model import NLIModel
from core.llm_provider import LLMProvider
from agents.query_understanding_agent import QueryUnderstandingAgent, QueryAnalysis
from agents.retrieval_agent import RetrievalAgent
from agents.evidence_ranking_agent import EvidenceRankingAgent
from agents.hallucination_detection_agent import HallucinationDetectionAgent, HallucinationReport
from agents.fact_verification_agent import FactVerificationAgent, FactVerificationReport
from agents.response_generation_agent import ResponseGenerationAgent, GeneratedResponse
from agents.evaluation_agent import EvaluationAgent, EvaluationMetrics


class PipelineState(TypedDict):
    """State that flows through the LangGraph pipeline."""
    # Input
    query: str
    llm_response: str  # The response to check for hallucinations

    # Query Understanding output
    query_analysis: Optional[dict]
    rewritten_query: str

    # Retrieval output
    retrieved_documents: List[str]
    retrieval_scores: List[float]
    num_retrieved: int

    # Evidence Ranking output
    ranked_evidence: List[str]
    evidence_scores: List[float]
    num_ranked: int

    # Hallucination Detection output
    hallucination_report: Optional[dict]
    hallucination_rate: float

    # Fact Verification output
    verification_report: Optional[dict]
    verified_claims: List[dict]

    # Response Generation output
    corrected_response: str
    citations: List[dict]

    # Evaluation output
    metrics: Optional[dict]

    # Metadata
    latency_ms: float
    errors: List[str]
    start_time: float


class MultiAgentOrchestrator:
    """
    Orchestrates the multi-agent pipeline using LangGraph.
    
    Manages initialization of all agents and defines the execution graph.
    """

    def __init__(self, config: Optional[AppConfig] = None, use_llm: bool = False):
        """
        Initialize the orchestrator and all agents.
        
        Args:
            config: Application configuration. Uses default if None.
            use_llm: Whether to use LLM for response generation.
        """
        self.config = config or get_config()
        self._initialize_agents(use_llm)
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        logger.info("Multi-Agent Orchestrator initialized and graph compiled")

    def _initialize_agents(self, use_llm: bool) -> None:
        """Initialize all agent components."""
        logger.info("Initializing agents...")

        # Core components
        self.embedding_engine = EmbeddingEngine(
            model_name=self.config.models.embedding_model
        )
        self.vector_store = FAISSVectorStore(
            embedding_dimension=self.embedding_engine.get_dimension(),
            store_path=self.config.vector_store.vector_store_path,
        )
        self.chunker = TextChunker(
            chunk_size=self.config.vector_store.chunk_size,
            chunk_overlap=self.config.vector_store.chunk_overlap,
        )
        self.document_processor = DocumentProcessor()
        self.reranker = CrossEncoderReranker(
            model_name=self.config.models.reranker_model
        )
        self.nli_model = NLIModel(
            model_name=self.config.models.nli_model
        )

        # Agents
        self.query_agent = QueryUnderstandingAgent()
        self.retrieval_agent = RetrievalAgent(
            embedding_engine=self.embedding_engine,
            vector_store=self.vector_store,
            chunker=self.chunker,
            document_processor=self.document_processor,
            top_k=self.config.vector_store.top_k_retrieval,
        )
        self.ranking_agent = EvidenceRankingAgent(
            reranker=self.reranker,
            top_k=self.config.vector_store.top_k_rerank,
        )
        self.hallucination_agent = HallucinationDetectionAgent(
            nli_model=self.nli_model,
        )
        self.fact_agent = FactVerificationAgent(
            nli_model=self.nli_model,
        )

        # Response generation (optional LLM)
        llm_provider = None
        if use_llm:
            try:
                llm_provider = LLMProvider(
                    provider="ollama",
                    model_name=self.config.models.ollama_model,
                    ollama_base_url=self.config.models.ollama_base_url,
                )
            except Exception as e:
                logger.warning(f"LLM provider failed to initialize: {e}")

        self.response_agent = ResponseGenerationAgent(llm_provider=llm_provider)
        self.evaluation_agent = EvaluationAgent()

        logger.info("All agents initialized successfully")

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph defining the agent pipeline.
        
        Returns:
            Compiled StateGraph ready for execution.
        """
        graph = StateGraph(PipelineState)

        # Add nodes
        graph.add_node("query_understanding", self._node_query_understanding)
        graph.add_node("retrieval", self._node_retrieval)
        graph.add_node("evidence_ranking", self._node_evidence_ranking)
        graph.add_node("hallucination_detection", self._node_hallucination_detection)
        graph.add_node("fact_verification", self._node_fact_verification)
        graph.add_node("response_generation", self._node_response_generation)
        graph.add_node("evaluation", self._node_evaluation)

        # Define edges (linear pipeline)
        graph.set_entry_point("query_understanding")
        graph.add_edge("query_understanding", "retrieval")
        graph.add_edge("retrieval", "evidence_ranking")
        graph.add_edge("evidence_ranking", "hallucination_detection")
        graph.add_edge("hallucination_detection", "fact_verification")
        graph.add_edge("fact_verification", "response_generation")
        graph.add_edge("response_generation", "evaluation")
        graph.add_edge("evaluation", END)

        return graph

    def _node_query_understanding(self, state: PipelineState) -> dict:
        """Node: Analyze and rewrite the user query."""
        try:
            analysis = self.query_agent.analyze(state["query"])
            return {
                "query_analysis": {
                    "intent": analysis.intent,
                    "entities": analysis.entities,
                    "keywords": analysis.keywords,
                    "is_ambiguous": analysis.is_ambiguous,
                    "query_type": analysis.query_type,
                },
                "rewritten_query": analysis.rewritten_query,
            }
        except Exception as e:
            logger.error(f"Query understanding failed: {e}")
            return {
                "query_analysis": None,
                "rewritten_query": state["query"],
                "errors": state.get("errors", []) + [f"Query understanding: {str(e)}"],
            }

    def _node_retrieval(self, state: PipelineState) -> dict:
        """Node: Retrieve relevant documents from vector store."""
        try:
            query = state.get("rewritten_query") or state["query"]
            result = self.retrieval_agent.retrieve(query)

            documents = [r.content for r in result.results]
            scores = [r.score for r in result.results]

            return {
                "retrieved_documents": documents,
                "retrieval_scores": scores,
                "num_retrieved": result.num_results,
            }
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                "retrieved_documents": [],
                "retrieval_scores": [],
                "num_retrieved": 0,
                "errors": state.get("errors", []) + [f"Retrieval: {str(e)}"],
            }

    def _node_evidence_ranking(self, state: PipelineState) -> dict:
        """Node: Rerank and filter retrieved evidence."""
        try:
            from core.vector_store import SearchResult

            # Reconstruct SearchResult objects
            search_results = []
            docs = state.get("retrieved_documents", [])
            scores = state.get("retrieval_scores", [])

            for i, (doc, score) in enumerate(zip(docs, scores)):
                search_results.append(SearchResult(
                    content=doc,
                    score=score,
                    source="",
                    chunk_id=i,
                    metadata={},
                ))

            query = state.get("rewritten_query") or state["query"]
            ranking_result = self.ranking_agent.rank(query, search_results)

            ranked_texts = self.ranking_agent.get_evidence_texts(ranking_result)
            ranked_scores = [r.relevance_score for r in ranking_result.ranked_evidence]

            return {
                "ranked_evidence": ranked_texts,
                "evidence_scores": ranked_scores,
                "num_ranked": ranking_result.num_output,
            }
        except Exception as e:
            logger.error(f"Evidence ranking failed: {e}")
            return {
                "ranked_evidence": state.get("retrieved_documents", [])[:5],
                "evidence_scores": [],
                "num_ranked": min(5, len(state.get("retrieved_documents", []))),
                "errors": state.get("errors", []) + [f"Ranking: {str(e)}"],
            }

    def _node_hallucination_detection(self, state: PipelineState) -> dict:
        """Node: Detect hallucinations in the LLM response."""
        try:
            evidence = state.get("ranked_evidence", [])
            llm_response = state.get("llm_response", "")

            if not llm_response:
                return {
                    "hallucination_report": None,
                    "hallucination_rate": 0.0,
                }

            report = self.hallucination_agent.detect(llm_response, evidence)

            report_dict = {
                "original_response": report.original_response,
                "sentence_results": [
                    {
                        "sentence": r.sentence,
                        "label": r.label,
                        "confidence": r.confidence,
                        "supporting_evidence": r.supporting_evidence,
                        "scores": r.scores,
                    }
                    for r in report.sentence_results
                ],
                "total_sentences": report.total_sentences,
                "supported_count": report.supported_count,
                "contradicted_count": report.contradicted_count,
                "neutral_count": report.neutral_count,
                "hallucination_rate": report.hallucination_rate,
                "overall_confidence": report.overall_confidence,
            }

            return {
                "hallucination_report": report_dict,
                "hallucination_rate": report.hallucination_rate,
            }
        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            return {
                "hallucination_report": None,
                "hallucination_rate": 0.0,
                "errors": state.get("errors", []) + [f"Hallucination detection: {str(e)}"],
            }

    def _node_fact_verification(self, state: PipelineState) -> dict:
        """Node: Verify factual claims."""
        try:
            evidence = state.get("ranked_evidence", [])
            llm_response = state.get("llm_response", "")

            if not llm_response:
                return {
                    "verification_report": None,
                    "verified_claims": [],
                }

            report = self.fact_agent.verify(llm_response, evidence)

            report_dict = {
                "total_claims": report.total_claims,
                "verified_count": report.verified_count,
                "refuted_count": report.refuted_count,
                "unverifiable_count": report.unverifiable_count,
                "overall_accuracy": report.overall_accuracy,
                "average_confidence": report.average_confidence,
            }

            verified_claims = [
                {
                    "claim": vc.claim.text,
                    "verdict": vc.verdict,
                    "confidence": vc.confidence,
                    "supporting_evidence": vc.supporting_evidence[:2],
                    "contradicting_evidence": vc.contradicting_evidence[:2],
                }
                for vc in report.claims
            ]

            return {
                "verification_report": report_dict,
                "verified_claims": verified_claims,
            }
        except Exception as e:
            logger.error(f"Fact verification failed: {e}")
            return {
                "verification_report": None,
                "verified_claims": [],
                "errors": state.get("errors", []) + [f"Fact verification: {str(e)}"],
            }

    def _node_response_generation(self, state: PipelineState) -> dict:
        """Node: Generate corrected response."""
        try:
            evidence = state.get("ranked_evidence", [])
            query = state["query"]

            # Build a HallucinationReport-like object if we have the report dict
            hallucination_report = None
            hr_dict = state.get("hallucination_report")
            if hr_dict:
                from agents.hallucination_detection_agent import HallucinationReport, SentenceVerification
                hallucination_report = HallucinationReport(
                    original_response=hr_dict.get("original_response", ""),
                    sentence_results=[
                        SentenceVerification(
                            sentence=sr["sentence"],
                            label=sr["label"],
                            confidence=sr["confidence"],
                            supporting_evidence=sr.get("supporting_evidence", ""),
                            scores=sr.get("scores", {}),
                        )
                        for sr in hr_dict.get("sentence_results", [])
                    ],
                    total_sentences=hr_dict.get("total_sentences", 0),
                    supported_count=hr_dict.get("supported_count", 0),
                    contradicted_count=hr_dict.get("contradicted_count", 0),
                    neutral_count=hr_dict.get("neutral_count", 0),
                    hallucination_rate=hr_dict.get("hallucination_rate", 0.0),
                    overall_confidence=hr_dict.get("overall_confidence", 0.0),
                )

            result = self.response_agent.generate_corrected_response(
                query=query,
                evidence=evidence,
                hallucination_report=hallucination_report,
                verification_report=None,
            )

            return {
                "corrected_response": result.response,
                "citations": result.citations,
            }
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "corrected_response": "Unable to generate a corrected response.",
                "citations": [],
                "errors": state.get("errors", []) + [f"Response generation: {str(e)}"],
            }

    def _node_evaluation(self, state: PipelineState) -> dict:
        """Node: Compute evaluation metrics."""
        try:
            end_time = time.perf_counter()
            start_time = state.get("start_time", end_time)
            latency_ms = (end_time - start_time) * 1000

            hallucination_report = state.get("hallucination_report")

            if hallucination_report:
                metrics = self.evaluation_agent.evaluate_from_report(
                    hallucination_report, latency_ms
                )
            else:
                # Even without hallucination report, provide basic metrics
                num_evidence = len(state.get("ranked_evidence", []))
                metrics_obj = self.evaluation_agent._empty_metrics(latency_ms)
                metrics_obj.retrieval_precision = 1.0 if num_evidence > 0 else 0.0
                metrics_obj.num_claims_verified = 0
                metrics = metrics_obj

            metrics_dict = {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "accuracy": metrics.accuracy,
                "hallucination_rate": metrics.hallucination_rate,
                "latency_ms": metrics.latency_ms,
                "retrieval_precision": metrics.retrieval_precision,
                "num_claims_verified": metrics.num_claims_verified,
                "average_confidence": metrics.average_confidence,
            }

            return {
                "metrics": metrics_dict,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "metrics": None,
                "latency_ms": 0.0,
                "errors": state.get("errors", []) + [f"Evaluation: {str(e)}"],
            }

    def run(self, query: str, llm_response: str = "") -> dict:
        """
        Execute the full multi-agent pipeline.
        
        Args:
            query: User's question.
            llm_response: LLM-generated response to verify (optional).
            
        Returns:
            Final pipeline state with all results.
        """
        initial_state: PipelineState = {
            "query": query,
            "llm_response": llm_response,
            "query_analysis": None,
            "rewritten_query": "",
            "retrieved_documents": [],
            "retrieval_scores": [],
            "num_retrieved": 0,
            "ranked_evidence": [],
            "evidence_scores": [],
            "num_ranked": 0,
            "hallucination_report": None,
            "hallucination_rate": 0.0,
            "verification_report": None,
            "verified_claims": [],
            "corrected_response": "",
            "citations": [],
            "metrics": None,
            "latency_ms": 0.0,
            "errors": [],
            "start_time": time.perf_counter(),
        }

        logger.info(f"Running pipeline for query: '{query[:80]}...'")
        result = self.app.invoke(initial_state)
        logger.info(f"Pipeline complete. Latency: {result.get('latency_ms', 0):.0f}ms")

        return result

    def ingest_document(self, file_path: str) -> int:
        """Ingest a document into the vector store."""
        return self.retrieval_agent.ingest_document(file_path)

    def ingest_text(self, text: str, source: str = "direct") -> int:
        """Ingest raw text into the vector store."""
        return self.retrieval_agent.ingest_text(text, source)

    def ingest_url(self, url: str) -> int:
        """Ingest a web page into the vector store."""
        return self.retrieval_agent.ingest_url(url)

    def save_index(self) -> None:
        """Save the vector store to disk."""
        self.retrieval_agent.save_index()

    def load_index(self) -> bool:
        """Load the vector store from disk."""
        return self.retrieval_agent.load_index()

    @property
    def index_size(self) -> int:
        """Number of chunks in the vector store."""
        return self.retrieval_agent.index_size
