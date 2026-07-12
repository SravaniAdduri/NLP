"""
Fact Verification Agent
Responsible for:
1. Extracting individual claims from text.
2. Verifying each claim against evidence.
3. Providing confidence scores and citations.
"""

import re
from typing import List, Tuple
from dataclasses import dataclass, field

from loguru import logger

from core.nli_model import NLIModel, EntailmentLabel


@dataclass
class Claim:
    """A single extracted claim from text."""
    text: str
    claim_id: int
    source_sentence: str


@dataclass
class VerifiedClaim:
    """A claim with verification results."""
    claim: Claim
    verdict: str  # VERIFIED, REFUTED, UNVERIFIABLE
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    citation_indices: List[int]


@dataclass
class FactVerificationReport:
    """Complete fact verification report."""
    claims: List[VerifiedClaim]
    total_claims: int
    verified_count: int
    refuted_count: int
    unverifiable_count: int
    overall_accuracy: float
    average_confidence: float


class FactVerificationAgent:
    """
    Verifies factual claims against retrieved evidence.
    
    Pipeline:
    1. Extract atomic claims from the response
    2. Match each claim against evidence passages
    3. Determine verdict (VERIFIED/REFUTED/UNVERIFIABLE)
    4. Assign confidence scores
    5. Provide citations to supporting/contradicting evidence
    """

    def __init__(self, nli_model: NLIModel, verification_threshold: float = 0.6):
        """
        Initialize the Fact Verification Agent.
        
        Args:
            nli_model: NLI model for entailment checking.
            verification_threshold: Minimum confidence to mark as VERIFIED/REFUTED.
        """
        self.nli_model = nli_model
        self.verification_threshold = verification_threshold
        logger.info(f"Fact Verification Agent initialized (threshold={verification_threshold})")

    def verify(self, response: str, evidence: List[str]) -> FactVerificationReport:
        """
        Perform full fact verification on a response.
        
        Args:
            response: LLM-generated text to verify.
            evidence: List of evidence passages.
            
        Returns:
            FactVerificationReport with all verification results.
        """
        # Step 1: Extract claims
        claims = self.extract_claims(response)
        logger.info(f"Extracted {len(claims)} claims from response")

        if not claims:
            return FactVerificationReport(
                claims=[],
                total_claims=0,
                verified_count=0,
                refuted_count=0,
                unverifiable_count=0,
                overall_accuracy=1.0,
                average_confidence=1.0,
            )

        # Step 2: Verify each claim
        verified_claims = []
        for claim in claims:
            verified = self._verify_single_claim(claim, evidence)
            verified_claims.append(verified)

        # Step 3: Calculate metrics
        verified_count = sum(1 for vc in verified_claims if vc.verdict == "VERIFIED")
        refuted_count = sum(1 for vc in verified_claims if vc.verdict == "REFUTED")
        unverifiable_count = sum(1 for vc in verified_claims if vc.verdict == "UNVERIFIABLE")

        total = len(verified_claims)
        overall_accuracy = verified_count / total if total > 0 else 0.0
        avg_confidence = sum(vc.confidence for vc in verified_claims) / total if total > 0 else 0.0

        report = FactVerificationReport(
            claims=verified_claims,
            total_claims=total,
            verified_count=verified_count,
            refuted_count=refuted_count,
            unverifiable_count=unverifiable_count,
            overall_accuracy=overall_accuracy,
            average_confidence=avg_confidence,
        )

        logger.info(
            f"Fact verification complete: {total} claims, "
            f"verified={verified_count}, refuted={refuted_count}, "
            f"unverifiable={unverifiable_count}"
        )

        return report

    def extract_claims(self, text: str) -> List[Claim]:
        """
        Extract atomic factual claims from text.
        
        An atomic claim is a single verifiable statement.
        Complex sentences are decomposed into individual claims.
        
        Args:
            text: Text to extract claims from.
            
        Returns:
            List of Claim objects.
        """
        sentences = self._split_sentences(text)
        claims = []
        claim_id = 0

        for sentence in sentences:
            if not self._is_verifiable(sentence):
                continue

            # Decompose compound sentences into atomic claims
            atomic_claims = self._decompose_sentence(sentence)

            for claim_text in atomic_claims:
                if claim_text.strip() and len(claim_text.split()) >= 3:
                    claims.append(Claim(
                        text=claim_text.strip(),
                        claim_id=claim_id,
                        source_sentence=sentence,
                    ))
                    claim_id += 1

        return claims

    def _verify_single_claim(self, claim: Claim, evidence: List[str]) -> VerifiedClaim:
        """
        Verify a single claim against all evidence passages.
        
        Args:
            claim: The claim to verify.
            evidence: List of evidence passages.
            
        Returns:
            VerifiedClaim with verdict and citations.
        """
        if not evidence:
            return VerifiedClaim(
                claim=claim,
                verdict="UNVERIFIABLE",
                confidence=1.0,
                supporting_evidence=[],
                contradicting_evidence=[],
                citation_indices=[],
            )

        supporting = []
        contradicting = []
        citation_indices = []
        best_support_confidence = 0.0
        best_contradict_confidence = 0.0

        # Check claim against each evidence passage
        pairs = [(ev, claim.text) for ev in evidence]
        results = self.nli_model.predict_batch(pairs)

        for i, result in enumerate(results):
            if result.label == EntailmentLabel.SUPPORTED and result.confidence > 0.5:
                supporting.append(evidence[i])
                citation_indices.append(i)
                best_support_confidence = max(best_support_confidence, result.confidence)
            elif result.label == EntailmentLabel.CONTRADICTED and result.confidence > 0.5:
                contradicting.append(evidence[i])
                best_contradict_confidence = max(best_contradict_confidence, result.confidence)

        # Determine verdict
        if contradicting and best_contradict_confidence >= self.verification_threshold:
            verdict = "REFUTED"
            confidence = best_contradict_confidence
        elif supporting and best_support_confidence >= self.verification_threshold:
            verdict = "VERIFIED"
            confidence = best_support_confidence
        else:
            verdict = "UNVERIFIABLE"
            confidence = max(best_support_confidence, best_contradict_confidence, 0.5)

        return VerifiedClaim(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            supporting_evidence=supporting[:3],  # Limit citations
            contradicting_evidence=contradicting[:3],
            citation_indices=citation_indices[:3],
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        text = re.sub(r'(Mr|Mrs|Dr|Prof|Inc|Ltd|Jr|Sr|vs)\.\s', r'\1<DOT> ', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        return [s.strip() for s in sentences if s.strip()]

    def _is_verifiable(self, sentence: str) -> bool:
        """Check if a sentence contains verifiable factual content."""
        sentence_lower = sentence.lower()

        # Skip questions
        if sentence.endswith('?'):
            return False

        # Skip very short sentences
        if len(sentence.split()) < 4:
            return False

        # Skip hedged statements
        hedge_patterns = [
            r'^(maybe|perhaps|possibly|probably)',
            r'^(i think|i believe|i guess)',
            r'^(it seems|it appears|it looks like)',
        ]
        for pattern in hedge_patterns:
            if re.match(pattern, sentence_lower):
                return False

        return True

    def _decompose_sentence(self, sentence: str) -> List[str]:
        """
        Decompose a compound sentence into atomic claims.
        
        Splits on coordinating conjunctions and semicolons while
        preserving each part as a standalone claim.
        """
        # Split on conjunctions that join independent clauses
        parts = re.split(r'\s*(?:;\s*|,\s*and\s+|,\s*but\s+|,\s*however\s+)', sentence)

        # Filter parts that are too short to be meaningful claims
        claims = []
        for part in parts:
            part = part.strip().rstrip(',;')
            if len(part.split()) >= 3:
                claims.append(part)

        # If no splitting happened, return original sentence
        if not claims:
            claims = [sentence]

        return claims

    def get_verified_claims_text(self, report: FactVerificationReport) -> List[str]:
        """Extract text of all verified claims."""
        return [vc.claim.text for vc in report.claims if vc.verdict == "VERIFIED"]

    def get_refuted_claims_text(self, report: FactVerificationReport) -> List[str]:
        """Extract text of all refuted claims."""
        return [vc.claim.text for vc in report.claims if vc.verdict == "REFUTED"]
