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
        Generate a clean, well-formatted response by extracting and reorganizing
        the most relevant sentences from evidence.
        """
        if not evidence:
            return GeneratedResponse(
                response="No relevant information found in the knowledge base.",
                citations=[], num_evidence_used=0, generation_method="extractive",
            )

        # Check if evidence is relevant at all
        if evidence_scores and max(evidence_scores) < -3:
            return GeneratedResponse(
                response=f"The uploaded document does not contain information about: \"{query}\".",
                citations=[], num_evidence_used=0, generation_method="extractive_no_match",
            )

        # Build citations
        citations = [{"id": i+1, "text": ev[:200], "marker": f"[{i+1}]"} for i, ev in enumerate(evidence)]

        # Determine question intent so the formatter can choose a tighter answer style
        query_lc = query.lower()
        is_compare_query = any(word in query_lc for word in ["compare", "difference", "versus", "vs"])
        is_list_query = any(word in query_lc for word in ["list", "all", "show", "what are", "what is", "what's", "how many", "which"])
        is_why_how_query = any(word in query_lc for word in ["why", "how", "explain", "describe"])

        # Extract and score all sentences from evidence
        # Detect summary-type queries
        summary_keywords = {"summary", "summarize", "summarise", "overview", "brief", "gist", "outline"}
        is_summary_query = any(kw in query_lc for kw in summary_keywords)

        query_words = set(re.findall(r'\b\w{3,}\b', query_lc))
        query_words -= {"what", "who", "when", "where", "why", "how", "which",
                        "does", "the", "are", "was", "were", "can", "could",
                        "about", "tell", "explain", "describe", "main", "purpose",
                        "summary", "summarize", "summarise", "give", "provide",
                        "document", "file", "text", "overview", "brief"}
        query_entities = re.findall(r'\b[A-Z][a-z]+\b', query)

        scored: List[Tuple[str, float, int]] = []

        for ev_idx, ev_text in enumerate(evidence):
            ev_score = evidence_scores[ev_idx] if evidence_scores and ev_idx < len(evidence_scores) else 0.0
            # Clean and split into sentences (protect abbreviations and decimals)
            clean_text = re.sub(r'\s+', ' ', ev_text.replace('\n', ' ')).strip()
            protected = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', clean_text)
            for abbr in ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.',
                         'vs.', 'etc.', 'e.g.', 'i.e.', 'Fig.', 'No.', 'Vol.']:
                protected = protected.replace(abbr, abbr.replace('.', '<DOT>'))
            raw_sents = re.split(r'(?<=[.!?])\s+', protected)
            sentences = [s.replace('<DOT>', '.').strip() for s in raw_sents if s.strip()]

            for sent in sentences:
                sent = sent.strip()
                words = sent.split()
                if len(words) < 5 or len(sent) < 20:
                    continue

                sent_words = set(re.findall(r'\b\w{3,}\b', sent.lower()))
                overlap = len(query_words & sent_words) / max(len(query_words), 1)
                entity_hit = sum(1 for e in query_entities if e.lower() in sent.lower()) * 0.3
                length_bonus = 0.1 if 8 <= len(words) <= 30 else 0.0
                score = (ev_score * 0.3) + (overlap * 3.0) + entity_hit + length_bonus

                # For summary queries, boost longer informative sentences
                # since keyword overlap is not meaningful
                if is_summary_query:
                    info_bonus = min(len(words) / 25.0, 1.0) * 2.0
                    score = (ev_score * 0.5) + info_bonus + length_bonus

                scored.append((sent, score, ev_idx))

        if not scored:
            # Fallback: use the top evidence passages directly (verbatim, no risk of hallucination)
            fallback_parts = []
            for i, ev in enumerate(evidence[:3]):
                clean = re.sub(r'\s+', ' ', ev.replace('\n', ' ')).strip()
                if clean:
                    fallback_parts.append(clean[:400])
            response_text = "\n\n".join(fallback_parts) if fallback_parts else "No relevant information found."
            return GeneratedResponse(
                response=response_text,
                citations=citations[:3], num_evidence_used=min(3, len(evidence)),
                generation_method="extractive",
            )

        # Sort by score, deduplicate, pick top sentences
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = []
        used_texts = []
        used_ev = set()

        # For summary queries, select more sentences for broader coverage
        if is_compare_query:
            max_sents = 4
        elif is_list_query or is_why_how_query:
            max_sents = 5
        elif is_summary_query:
            max_sents = 6
        else:
            max_sents = 5

        for sent, sc, ev_idx in scored:
            if len(selected) >= max_sents:
                break
            if not any(self._sentence_similarity(sent, t) > 0.6 for t in used_texts):
                selected.append((sent, ev_idx))
                used_texts.append(sent)
                used_ev.add(ev_idx)

        # Format as a clean, natural-sounding response
        # Instead of dumping raw sentences grouped by source, weave them into
        # a coherent answer with a lead-in, logical ordering, and transitions.

        # Flatten selected sentences in score order (already relevance-ranked)
        flat_sents = []
        for sent, ev_idx in selected:
            sent = sent.rstrip()
            if sent and sent[-1] not in '.!?':
                sent = sent.rstrip(',;:') + '.'
            flat_sents.append(sent)

        # Build a natural lead-in based on query type
        lead_in = ""
        if is_summary_query:
            lead_in = "Here is a summary based on the available information:\n\n"
        elif is_compare_query:
            lead_in = "Here is a comparison based on the available information:\n\n"
        elif is_list_query:
            lead_in = ""  # lists are self-explanatory
        elif is_why_how_query:
            lead_in = ""

        # Format differently depending on query type
        if is_list_query:
            # Bullet list — clean and scannable
            body = "\n".join(f"• {s}" for s in flat_sents)
        elif is_compare_query:
            body = "\n".join(f"• {s}" for s in flat_sents)
        else:
            # Paragraph style — join sentences into flowing prose with light
            # transition words where the topic shifts between evidence sources.
            parts = []
            prev_ev = None
            transitions = ["Additionally, ", "Furthermore, ", "Moreover, ", "Also, "]
            t_idx = 0
            for (sent, ev_idx), clean_sent in zip(selected, flat_sents):
                if prev_ev is not None and ev_idx != prev_ev and len(parts) > 0:
                    # Different evidence source → add a soft transition
                    # Only lowercase the first char if it isn't an acronym/proper noun
                    first_word = clean_sent.split()[0] if clean_sent else ""
                    if first_word and first_word[0].isupper() and not first_word.isupper():
                        bridged = transitions[t_idx % len(transitions)] + clean_sent[0].lower() + clean_sent[1:]
                    else:
                        bridged = transitions[t_idx % len(transitions)] + clean_sent
                    parts.append(bridged)
                    t_idx += 1
                else:
                    parts.append(clean_sent)
                prev_ev = ev_idx
            body = " ".join(parts)

        response_text = (lead_in + body).strip()

        if not response_text.strip():
            # Final safety net
            clean = re.sub(r'\s+', ' ', evidence[0].replace('\n', ' ')).strip()
            response_text = clean[:600]

        return GeneratedResponse(
            response=response_text,
            citations=citations,
            num_evidence_used=len(used_ev),
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
