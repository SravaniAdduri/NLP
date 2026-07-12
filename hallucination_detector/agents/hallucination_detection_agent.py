"""
Hallucination Detection Agent
Responsible for:
1. Comparing LLM-generated responses with retrieved evidence.
2. Detecting unsupported claims at the sentence level.
3. Classifying each sentence as: Supported, Contradicted, or Not Enough Evidence.
"""

import re
from typing import List
from dataclasses import dataclass, field

from loguru import logger

from core.nli_model import NLIModel, NLIResult, EntailmentLabel


@dataclass
class SentenceVerification:
    """Verification result for a single sentence."""
    sentence: str
    label: str  # SUPPORTED, CONTRADICTED, NOT_ENOUGH_EVIDENCE
    confidence: float
    supporting_evidence: str
    scores: dict


@dataclass
class HallucinationReport:
    """Complete hallucination detection report for a response."""
    original_response: str
    sentence_results: List[SentenceVerification]
    total_sentences: int
    supported_count: int
    contradicted_count: int
    neutral_count: int
    hallucination_rate: float
    overall_confidence: float


class HallucinationDetectionAgent:
    """
    Detects hallucinations in LLM-generated text by comparing against evidence.
    
    Uses NLI (Natural Language Inference) to classify each sentence as:
    - SUPPORTED: Evidence entails the claim
    - CONTRADICTED: Evidence contradicts the claim
    - NOT_ENOUGH_EVIDENCE: Insufficient evidence to determine
    
    A sentence is considered hallucinated if it is CONTRADICTED or 
    makes factual claims with NOT_ENOUGH_EVIDENCE.
    """

    def __init__(self, nli_model: NLIModel, confidence_threshold: float = 0.5):
        """
        Initialize the Hallucination Detection Agent.
        
        Args:
            nli_model: NLI model for entailment classification.
            confidence_threshold: Minimum confidence to accept a classification.
        """
        self.nli_model = nli_model
        self.confidence_threshold = confidence_threshold
        logger.info(f"Hallucination Detection Agent initialized (threshold={confidence_threshold})")

    def detect(self, response: str, evidence: List[str]) -> HallucinationReport:
        """
        Analyze a response for hallucinations against provided evidence.
        
        Args:
            response: The LLM-generated response to check.
            evidence: List of evidence passages to verify against.
            
        Returns:
            HallucinationReport with per-sentence analysis.
        """
        if not response.strip():
            return HallucinationReport(
                original_response=response,
                sentence_results=[],
                total_sentences=0,
                supported_count=0,
                contradicted_count=0,
                neutral_count=0,
                hallucination_rate=0.0,
                overall_confidence=0.0,
            )

        # Split response into sentences
        sentences = self._split_into_sentences(response)

        # Filter out non-factual sentences (greetings, transitions, etc.)
        factual_sentences = [(i, s) for i, s in enumerate(sentences) if self._is_factual_claim(s)]

        sentence_results = []
        for _, sentence in factual_sentences:
            verification = self._verify_sentence(sentence, evidence)
            sentence_results.append(verification)

        # Also include non-factual sentences as SUPPORTED (they don't need verification)
        all_results = []
        factual_indices = {i for i, _ in factual_sentences}
        factual_idx = 0
        for i, sentence in enumerate(sentences):
            if i in factual_indices:
                all_results.append(sentence_results[factual_idx])
                factual_idx += 1
            else:
                all_results.append(SentenceVerification(
                    sentence=sentence,
                    label=EntailmentLabel.SUPPORTED.value,
                    confidence=1.0,
                    supporting_evidence="Non-factual statement (no verification needed)",
                    scores={"contradiction": 0.0, "neutral": 0.0, "entailment": 1.0},
                ))

        # Calculate metrics
        supported = sum(1 for r in all_results if r.label == EntailmentLabel.SUPPORTED.value)
        contradicted = sum(1 for r in all_results if r.label == EntailmentLabel.CONTRADICTED.value)
        neutral = sum(1 for r in all_results if r.label == EntailmentLabel.NOT_ENOUGH_EVIDENCE.value)
        total = len(all_results)

        # Hallucination rate = (contradicted + neutral) / total factual claims
        factual_count = len(sentence_results)
        if factual_count > 0:
            hallucination_rate = (contradicted + neutral) / factual_count
        else:
            hallucination_rate = 0.0

        # Overall confidence = average confidence of all classifications
        if all_results:
            overall_confidence = sum(r.confidence for r in all_results) / len(all_results)
        else:
            overall_confidence = 0.0

        report = HallucinationReport(
            original_response=response,
            sentence_results=all_results,
            total_sentences=total,
            supported_count=supported,
            contradicted_count=contradicted,
            neutral_count=neutral,
            hallucination_rate=hallucination_rate,
            overall_confidence=overall_confidence,
        )

        logger.info(
            f"Hallucination detection complete: {total} sentences, "
            f"rate={hallucination_rate:.2f}, "
            f"supported={supported}, contradicted={contradicted}, neutral={neutral}"
        )

        return report

    def _verify_sentence(self, sentence: str, evidence: List[str]) -> SentenceVerification:
        """
        Verify a single sentence against all available evidence.
        
        Uses the NLI model to check the sentence against each piece of evidence,
        then takes the strongest signal.
        """
        if not evidence:
            return SentenceVerification(
                sentence=sentence,
                label=EntailmentLabel.NOT_ENOUGH_EVIDENCE.value,
                confidence=1.0,
                supporting_evidence="",
                scores={"contradiction": 0.0, "neutral": 1.0, "entailment": 0.0},
            )

        # Check sentence against all evidence passages
        nli_result = self.nli_model.classify_claim_against_evidence(sentence, evidence)

        return SentenceVerification(
            sentence=sentence,
            label=nli_result.label.value,
            confidence=nli_result.confidence,
            supporting_evidence=nli_result.evidence,
            scores=nli_result.scores,
        )

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Handle common abbreviations to avoid false splits
        text = re.sub(r'(Mr|Mrs|Dr|Prof|Inc|Ltd|Jr|Sr|vs)\.\s', r'\1<DOT> ', text)

        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]

        # Filter empty sentences
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

    def _is_factual_claim(self, sentence: str) -> bool:
        """
        Determine if a sentence contains a factual claim that needs verification.
        
        Returns False for:
        - Questions
        - Greetings and pleasantries
        - Transitional phrases
        - Opinions explicitly marked as such
        """
        sentence_lower = sentence.lower().strip()

        # Skip questions
        if sentence.strip().endswith('?'):
            return False

        # Skip very short sentences
        if len(sentence.split()) < 4:
            return False

        # Skip common non-factual patterns
        non_factual_patterns = [
            r'^(hi|hello|hey|good morning|good evening)',
            r'^(sure|of course|certainly|absolutely)',
            r'^(i think|i believe|in my opinion|personally)',
            r'^(let me|allow me|i\'ll|i will)',
            r'^(thank|thanks|you\'re welcome)',
            r'^(here is|here are|here\'s|below)',
            r'^(please note|note that|keep in mind)',
        ]

        for pattern in non_factual_patterns:
            if re.match(pattern, sentence_lower):
                return False

        return True

    def get_hallucinated_sentences(self, report: HallucinationReport) -> List[SentenceVerification]:
        """Extract only the hallucinated (contradicted) sentences from a report."""
        return [
            r for r in report.sentence_results
            if r.label == EntailmentLabel.CONTRADICTED.value
        ]

    def get_unsupported_sentences(self, report: HallucinationReport) -> List[SentenceVerification]:
        """Extract sentences with not enough evidence."""
        return [
            r for r in report.sentence_results
            if r.label == EntailmentLabel.NOT_ENOUGH_EVIDENCE.value
        ]
