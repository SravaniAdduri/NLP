"""
Response Generation Agent
Responsible for:
1. Generating corrected responses using only verified evidence.
2. Including inline citations.
3. Ensuring the response avoids hallucinations.
"""

from typing import List, Optional
from dataclasses import dataclass

from loguru import logger

from core.llm_provider import LLMProvider
from agents.fact_verification_agent import FactVerificationReport, VerifiedClaim
from agents.hallucination_detection_agent import HallucinationReport


@dataclass
class GeneratedResponse:
    """A generated response with citations and metadata."""
    response: str
    citations: List[dict]
    num_evidence_used: int
    generation_method: str


class ResponseGenerationAgent:
    """
    Generates factually accurate responses grounded in verified evidence.
    
    Strategies:
    1. Template-based: For simple factual queries
    2. LLM-based: For complex queries requiring synthesis
    3. Extractive: For when evidence directly answers the question
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize the Response Generation Agent.
        
        Args:
            llm_provider: LLM provider for generation. If None, uses extractive method.
        """
        self.llm_provider = llm_provider
        self.use_llm = llm_provider is not None
        logger.info(f"Response Generation Agent initialized (LLM: {self.use_llm})")

    def generate_corrected_response(
        self,
        query: str,
        evidence: List[str],
        hallucination_report: Optional[HallucinationReport] = None,
        verification_report: Optional[FactVerificationReport] = None,
    ) -> GeneratedResponse:
        """
        Generate a corrected, factually accurate response.
        
        Args:
            query: The original user query.
            evidence: List of verified evidence passages.
            hallucination_report: Report from hallucination detection (optional).
            verification_report: Report from fact verification (optional).
            
        Returns:
            GeneratedResponse with citations.
        """
        if not evidence:
            return GeneratedResponse(
                response="I don't have enough evidence to provide a reliable answer to this question.",
                citations=[],
                num_evidence_used=0,
                generation_method="no_evidence",
            )

        if self.use_llm:
            return self._generate_with_llm(query, evidence, hallucination_report, verification_report)
        else:
            return self._generate_extractive(query, evidence, verification_report)

    def _generate_with_llm(
        self,
        query: str,
        evidence: List[str],
        hallucination_report: Optional[HallucinationReport],
        verification_report: Optional[FactVerificationReport],
    ) -> GeneratedResponse:
        """Generate response using LLM with evidence grounding."""

        # Build context from evidence with citation markers
        context_parts = []
        citations = []
        for i, ev in enumerate(evidence):
            citation_marker = f"[{i+1}]"
            context_parts.append(f"{citation_marker} {ev}")
            citations.append({
                "id": i + 1,
                "text": ev[:200],
                "marker": citation_marker,
            })

        context = "\n\n".join(context_parts)

        # Build prompt with specific instructions to avoid hallucination
        prompt = self._build_generation_prompt(query, context, hallucination_report)

        # Generate response
        response_text = self.llm_provider.generate(prompt, max_tokens=512, temperature=0.2)

        return GeneratedResponse(
            response=response_text,
            citations=citations,
            num_evidence_used=len(evidence),
            generation_method="llm_grounded",
        )

    def _generate_extractive(
        self,
        query: str,
        evidence: List[str],
        verification_report: Optional[FactVerificationReport],
    ) -> GeneratedResponse:
        """
        Generate response by extracting and combining relevant evidence.
        Used when no LLM is available.
        """
        # Build response from verified evidence
        response_parts = []
        citations = []

        for i, ev in enumerate(evidence):
            # Truncate long passages
            truncated = ev if len(ev) <= 300 else ev[:300] + "..."
            response_parts.append(f"{truncated} [{i+1}]")
            citations.append({
                "id": i + 1,
                "text": ev[:200],
                "marker": f"[{i+1}]",
            })

        # Combine evidence into a coherent response
        if response_parts:
            response_text = f"Based on the available evidence:\n\n" + "\n\n".join(response_parts)
        else:
            response_text = "No verified information available for this query."

        return GeneratedResponse(
            response=response_text,
            citations=citations,
            num_evidence_used=len(evidence),
            generation_method="extractive",
        )

    def _build_generation_prompt(
        self,
        query: str,
        context: str,
        hallucination_report: Optional[HallucinationReport],
    ) -> str:
        """Build the generation prompt with anti-hallucination instructions."""

        # Add specific guidance about what was hallucinated
        hallucination_guidance = ""
        if hallucination_report and hallucination_report.contradicted_count > 0:
            contradicted = [
                r.sentence for r in hallucination_report.sentence_results
                if r.label == "CONTRADICTED"
            ]
            if contradicted:
                hallucination_guidance = (
                    "\n\nIMPORTANT: The following statements from a previous response were "
                    "found to be INCORRECT. Do NOT repeat them:\n"
                    + "\n".join(f"- {s}" for s in contradicted[:5])
                )

        prompt = f"""You are a factual assistant. Generate an accurate answer to the question using ONLY the provided evidence.

STRICT RULES:
1. ONLY use information from the evidence passages below.
2. Include citation markers (e.g., [1], [2]) when using information from a passage.
3. If the evidence does not contain enough information, say so explicitly.
4. Do NOT add any information not found in the evidence.
5. Do NOT speculate or make assumptions.
{hallucination_guidance}

EVIDENCE:
{context}

QUESTION: {query}

ANSWER (with citations):"""

        return prompt

    def generate_summary(self, evidence: List[str], max_length: int = 300) -> str:
        """
        Generate a brief summary of the evidence.
        
        Args:
            evidence: List of evidence passages.
            max_length: Maximum length of the summary.
            
        Returns:
            Summary string.
        """
        if not evidence:
            return "No evidence available."

        if self.use_llm:
            context = "\n".join(evidence[:5])
            prompt = f"Summarize the following information in 2-3 sentences:\n\n{context}\n\nSummary:"
            return self.llm_provider.generate(prompt, max_tokens=150, temperature=0.2)
        else:
            # Simple extractive summary: first sentence of top evidence
            summary_parts = []
            for ev in evidence[:3]:
                first_sentence = ev.split('.')[0] + '.'
                if len(first_sentence) > 20:
                    summary_parts.append(first_sentence)
            return " ".join(summary_parts)[:max_length]
