"""
Natural Language Inference (NLI) Model Module
Uses a cross-encoder NLI model to determine textual entailment relationships.
Labels: ENTAILMENT (supported), CONTRADICTION (contradicted), NEUTRAL (not enough evidence).
"""

from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum

from sentence_transformers import CrossEncoder
import numpy as np
from loguru import logger


class EntailmentLabel(str, Enum):
    """NLI classification labels."""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_ENOUGH_EVIDENCE = "NOT_ENOUGH_EVIDENCE"


@dataclass
class NLIResult:
    """Result of NLI classification for a single claim-evidence pair."""
    claim: str
    evidence: str
    label: EntailmentLabel
    confidence: float
    scores: dict  # Raw scores for all three classes


class NLIModel:
    """
    Natural Language Inference model for hallucination detection.
    Uses DeBERTa-v3-base cross-encoder to classify premise-hypothesis pairs.
    
    Label mapping:
    - ENTAILMENT -> SUPPORTED (evidence supports the claim)
    - CONTRADICTION -> CONTRADICTED (evidence contradicts the claim)
    - NEUTRAL -> NOT_ENOUGH_EVIDENCE (evidence neither supports nor contradicts)
    """

    LABEL_MAP = {
        0: EntailmentLabel.CONTRADICTED,    # contradiction
        1: EntailmentLabel.NOT_ENOUGH_EVIDENCE,  # neutral
        2: EntailmentLabel.SUPPORTED,        # entailment
    }

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", device: str = None):
        """
        Initialize the NLI model.
        
        Args:
            model_name: HuggingFace NLI cross-encoder model name.
            device: Compute device ('cpu', 'cuda', 'mps').
        """
        logger.info(f"Loading NLI model: {model_name}")
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        """
        Predict the entailment relationship between a premise and hypothesis.
        
        Args:
            premise: The evidence/reference text (what we know to be true).
            hypothesis: The claim to verify (what the LLM generated).
            
        Returns:
            NLIResult with label and confidence scores.
        """
        scores = self.model.predict([(premise, hypothesis)])[0]

        # Apply softmax to get probabilities
        probabilities = self._softmax(scores)

        # Get the predicted label
        predicted_idx = int(np.argmax(probabilities))
        label = self.LABEL_MAP[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        return NLIResult(
            claim=hypothesis,
            evidence=premise,
            label=label,
            confidence=confidence,
            scores={
                "contradiction": float(probabilities[0]),
                "neutral": float(probabilities[1]),
                "entailment": float(probabilities[2]),
            },
        )

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        """
        Predict entailment for multiple premise-hypothesis pairs.
        
        Args:
            pairs: List of (premise, hypothesis) tuples.
            
        Returns:
            List of NLIResult objects.
        """
        if not pairs:
            return []

        all_scores = self.model.predict(pairs)

        results = []
        for (premise, hypothesis), scores in zip(pairs, all_scores):
            probabilities = self._softmax(scores)
            predicted_idx = int(np.argmax(probabilities))
            label = self.LABEL_MAP[predicted_idx]
            confidence = float(probabilities[predicted_idx])

            results.append(NLIResult(
                claim=hypothesis,
                evidence=premise,
                label=label,
                confidence=confidence,
                scores={
                    "contradiction": float(probabilities[0]),
                    "neutral": float(probabilities[1]),
                    "entailment": float(probabilities[2]),
                },
            ))

        return results

    def classify_claim_against_evidence(
        self,
        claim: str,
        evidence_list: List[str],
    ) -> NLIResult:
        """
        Classify a claim against multiple pieces of evidence.
        Takes the strongest signal (highest confidence) across all evidence.
        
        Args:
            claim: The claim to verify.
            evidence_list: List of evidence passages.
            
        Returns:
            The NLIResult with the highest confidence.
        """
        if not evidence_list:
            return NLIResult(
                claim=claim,
                evidence="",
                label=EntailmentLabel.NOT_ENOUGH_EVIDENCE,
                confidence=1.0,
                scores={"contradiction": 0.0, "neutral": 1.0, "entailment": 0.0},
            )

        pairs = [(evidence, claim) for evidence in evidence_list]
        results = self.predict_batch(pairs)

        # Find the result with highest confidence for SUPPORTED or CONTRADICTED
        # Priority: CONTRADICTED > SUPPORTED > NOT_ENOUGH_EVIDENCE
        best_result = results[0]
        for result in results:
            if result.label == EntailmentLabel.CONTRADICTED and result.confidence > 0.5:
                if best_result.label != EntailmentLabel.CONTRADICTED or result.confidence > best_result.confidence:
                    best_result = result
            elif result.label == EntailmentLabel.SUPPORTED and result.confidence > best_result.confidence:
                if best_result.label != EntailmentLabel.CONTRADICTED:
                    best_result = result

        return best_result

    def _softmax(self, scores) -> np.ndarray:
        """Apply softmax to convert logits to probabilities."""
        scores = np.array(scores, dtype=np.float64)
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()
