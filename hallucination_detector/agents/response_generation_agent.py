"""
Response Generation Agent
Responsible for:
1. Generating corrected responses using only verified evidence.
2. Including inline citations.
3. Ensuring the response avoids hallucinations.
"""

import re
from typing import List, Optional, Tuple
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
        evidence_scores: Optional[List[float]] = None,
    ) -> GeneratedResponse:
        """
        Generate a corrected, factually accurate response.
        
        Args:
            query: The original user query.
            evidence: List of verified evidence passages.
            hallucination_report: Report from hallucination detection (optional).
            verification_report: Report from fact verification (optional).
            evidence_scores: Cross-encoder relevance scores for each evidence passage.
            
        Returns:
            GeneratedResponse with citations.
        """
        if not evidence:
            return GeneratedResponse(
                response="I don't have enough information in the knowledge base to answer this question. Please upload relevant documents first.",
                citations=[],
                num_evidence_used=0,
                generation_method="no_evidence",
            )

        if self.use_llm:
            result = self._generate_with_llm(query, evidence, hallucination_report, verification_report)
            # If LLM returned empty or error, fall back to extractive
            if not result.response or result.response.startswith("[Error"):
                logger.warning("LLM response empty or errored, falling back to extractive mode")
                return self._generate_extractive(query, evidence, evidence_scores, verification_report)
            return result
        else:
            return self._generate_extractive(query, evidence, evidence_scores, verification_report)

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
        evidence_scores: Optional[List[float]],
        verification_report: Optional[FactVerificationReport],
    ) -> GeneratedResponse:
        """
        Generate response by intelligently extracting from evidence.
        Uses sentence scoring based on query relevance and evidence quality.
        """
        if not evidence:
            return GeneratedResponse(
                response="No relevant information found in the knowledge base.",
                citations=[],
                num_evidence_used=0,
                generation_method="extractive",
            )

        # Check if evidence is actually relevant using scores
        if evidence_scores and len(evidence_scores) > 0:
            best_score = max(evidence_scores)
            # Cross-encoder scores: > 0 means relevant, < -5 means not relevant
            if best_score < -3:
                return GeneratedResponse(
                    response=f"The knowledge base does not contain sufficient information to answer: \"{query}\". Please upload relevant documents.",
                    citations=[],
                    num_evidence_used=0,
                    generation_method="extractive_no_match",
                )

        # Build citations list
        citations = []
        for i, ev in enumerate(evidence):
            citations.append({
                "id": i + 1,
                "text": ev[:200],
                "marker": f"[{i+1}]",
            })

        # Extract and score sentences from all evidence
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
        # Remove common question words from matching
        query_words -= {"what", "who", "when", "where", "why", "how", "which",
                        "does", "the", "are", "was", "were", "can", "could",
                        "about", "tell", "explain", "describe", "main"}

        all_sentences: List[Tuple[str, float, int]] = []  # (sentence, score, evidence_idx)

        for ev_idx, ev_text in enumerate(evidence):
            # Cross-encoder score for this evidence (higher = more relevant)
            ev_score = evidence_scores[ev_idx] if evidence_scores and ev_idx < len(evidence_scores) else 0.0

            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', ev_text.replace('\n', ' '))

            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 15 or len(sent.split()) < 4:
                    continue

                # Score this sentence
                sent_lower = sent.lower()
                sent_words = set(re.findall(r'\b\w{3,}\b', sent_lower))

                # Word overlap with query
                overlap = len(query_words & sent_words)
                overlap_ratio = overlap / max(len(query_words), 1)

                # Boost for sentences containing query entities (capitalized words in query)
                entity_bonus = 0
                query_entities = re.findall(r'\b[A-Z][a-z]+\b', query)
                for ent in query_entities:
                    if ent.lower() in sent_lower:
                        entity_bonus += 0.3

                # Combine: cross-encoder score + word overlap + entity bonus
                final_score = (ev_score * 0.4) + (overlap_ratio * 3.0) + entity_bonus

                # Penalty for very short or very long sentences
                word_count = len(sent.split())
                if word_count < 6:
                    final_score *= 0.5
                elif word_count > 50:
                    final_score *= 0.8

                all_sentences.append((sent, final_score, ev_idx))

        if not all_sentences:
            # Fallback: just show the top evidence passage directly
            return GeneratedResponse(
                response=f"Here is the most relevant information found:\n\n{evidence[0][:500]} [1]",
                citations=citations[:1],
                num_evidence_used=1,
                generation_method="extractive_fallback",
            )

        # Sort by score and take the best unique sentences
        all_sentences.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate similar sentences
        selected = []
        selected_texts = []
        used_ev_ids = set()

        for sent, score, ev_idx in all_sentences:
            if len(selected) >= 5:
                break
            # Check for near-duplicate
            is_dup = False
            for existing in selected_texts:
                if self._sentence_similarity(sent, existing) > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                selected.append((sent, score, ev_idx))
                selected_texts.append(sent)
                used_ev_ids.add(ev_idx)

        # Build the response
        response_lines = []
        for sent, score, ev_idx in selected:
            citation_id = ev_idx + 1
            response_lines.append(f"• {sent} [{citation_id}]")

        response_text = f"Based on the knowledge base, here is what I found regarding \"{query}\":\n\n"
        response_text += "\n\n".join(response_lines)

        return GeneratedResponse(
            response=response_text,
            citations=citations,
            num_evidence_used=len(used_ev_ids),
            generation_method="extractive",
        )

    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Compute word-level Jaccard similarity between two sentences."""
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

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
