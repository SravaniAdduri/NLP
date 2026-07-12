"""
Tests for the Query Understanding Agent
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from agents.query_understanding_agent import QueryUnderstandingAgent, QueryAnalysis


@pytest.fixture
def agent():
    return QueryUnderstandingAgent()


class TestQueryUnderstandingAgent:

    def test_basic_query_analysis(self, agent):
        """Test basic query analysis returns all expected fields."""
        result = agent.analyze("What is machine learning?")
        assert isinstance(result, QueryAnalysis)
        assert result.original_query == "What is machine learning?"
        assert result.rewritten_query != ""
        assert result.intent != ""
        assert result.query_type in ("definition", "factual", "how_to", "comparison", "opinion", "causal", "general")

    def test_entity_extraction(self, agent):
        """Test named entity extraction from queries."""
        result = agent.analyze("How does NASA use AI for Mars exploration?")
        assert "NASA" in result.entities or "AI" in result.entities or "Mars" in result.entities

    def test_keyword_extraction(self, agent):
        """Test keyword extraction filters stop words."""
        result = agent.analyze("What are the benefits of renewable energy?")
        assert "benefits" in result.keywords
        assert "renewable" in result.keywords
        assert "energy" in result.keywords
        # Stop words should be filtered
        assert "the" not in result.keywords
        assert "are" not in result.keywords

    def test_intent_classification_definition(self, agent):
        """Test definition intent detection."""
        result = agent.analyze("What is photosynthesis?")
        assert result.query_type == "definition"

    def test_intent_classification_factual(self, agent):
        """Test factual intent detection."""
        result = agent.analyze("When was the internet invented?")
        assert result.query_type == "factual"

    def test_intent_classification_how_to(self, agent):
        """Test how-to intent detection."""
        result = agent.analyze("How to train a neural network?")
        assert result.query_type == "how_to"

    def test_ambiguous_query_detection(self, agent):
        """Test ambiguity detection for short/vague queries."""
        result = agent.analyze("it")
        assert result.is_ambiguous is True

    def test_clear_query_not_ambiguous(self, agent):
        """Test that clear queries are not marked as ambiguous."""
        result = agent.analyze("What are the three laws of thermodynamics?")
        assert result.is_ambiguous is False

    def test_query_rewriting(self, agent):
        """Test query rewriting produces a non-empty result."""
        result = agent.analyze("Can you tell me what climate change is?")
        assert result.rewritten_query != ""
        assert len(result.rewritten_query) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
