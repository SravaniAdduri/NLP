"""
Query Understanding Agent
Responsible for:
1. Understanding user intent from the query.
2. Extracting named entities and keywords.
3. Rewriting ambiguous or poorly-formed queries for better retrieval.
"""

import re
from typing import List, Dict
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class QueryAnalysis:
    """Result of query understanding analysis."""
    original_query: str
    rewritten_query: str
    intent: str
    entities: List[str]
    keywords: List[str]
    is_ambiguous: bool
    query_type: str  # factual, opinion, comparison, definition, etc.


class QueryUnderstandingAgent:
    """
    Analyzes and transforms user queries for optimal retrieval.
    
    Performs:
    - Intent classification (factual, opinion, comparison, definition, how-to)
    - Named entity extraction using pattern matching and POS-like heuristics
    - Keyword extraction
    - Query rewriting for disambiguation
    """

    # Common stop words to filter out for keyword extraction
    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "because", "but", "and", "or", "if", "while", "about", "what",
        "which", "who", "whom", "this", "that", "these", "those", "it", "its",
        "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
        "she", "her", "they", "them", "their",
    }

    # Intent classification patterns
    INTENT_PATTERNS = {
        "definition": [r"what is", r"what are", r"define", r"meaning of", r"explain what"],
        "factual": [r"who", r"when", r"where", r"how many", r"how much", r"which"],
        "how_to": [r"how to", r"how do", r"how can", r"steps to", r"way to"],
        "comparison": [r"difference between", r"compare", r"vs", r"versus", r"better"],
        "opinion": [r"should", r"best", r"recommend", r"think about", r"opinion"],
        "causal": [r"why", r"cause", r"reason", r"because", r"result of"],
    }

    def __init__(self):
        """Initialize the Query Understanding Agent."""
        logger.info("Query Understanding Agent initialized")

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Perform full analysis of a user query.
        
        Args:
            query: Raw user query string.
            
        Returns:
            QueryAnalysis dataclass with all extracted information.
        """
        logger.debug(f"Analyzing query: {query}")

        # Clean the query
        cleaned_query = self._clean_query(query)

        # Classify intent
        intent, query_type = self._classify_intent(cleaned_query)

        # Extract entities
        entities = self._extract_entities(cleaned_query)

        # Extract keywords
        keywords = self._extract_keywords(cleaned_query)

        # Check ambiguity
        is_ambiguous = self._check_ambiguity(cleaned_query)

        # Rewrite query if needed
        rewritten_query = self._rewrite_query(cleaned_query, entities, keywords, is_ambiguous)

        analysis = QueryAnalysis(
            original_query=query,
            rewritten_query=rewritten_query,
            intent=intent,
            entities=entities,
            keywords=keywords,
            is_ambiguous=is_ambiguous,
            query_type=query_type,
        )

        logger.info(f"Query analysis complete. Intent: {intent}, Entities: {entities}")
        return analysis

    def _clean_query(self, query: str) -> str:
        """Remove extra whitespace and normalize the query."""
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)
        return query

    def _classify_intent(self, query: str) -> tuple:
        """
        Classify the intent and type of the query.
        
        Returns:
            Tuple of (intent_description, query_type).
        """
        query_lower = query.lower()

        for query_type, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    intent_map = {
                        "definition": "User wants a definition or explanation",
                        "factual": "User wants factual information",
                        "how_to": "User wants procedural instructions",
                        "comparison": "User wants a comparison",
                        "opinion": "User wants recommendations or opinions",
                        "causal": "User wants to understand causation",
                    }
                    return intent_map[query_type], query_type

        return "User wants general information", "general"

    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract named entities from the query using pattern matching.
        Identifies capitalized words, quoted phrases, and domain-specific terms.
        """
        entities = []

        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)

        # Extract capitalized words (likely proper nouns) - skip first word of sentence
        words = query.split()
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)
            if i > 0 and clean_word and clean_word[0].isupper() and clean_word.lower() not in self.STOP_WORDS:
                entities.append(clean_word)

        # Extract technical terms (words with numbers, acronyms)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', query)
        entities.extend(acronyms)

        # Extract hyphenated compounds
        compounds = re.findall(r'\b\w+(?:-\w+)+\b', query)
        entities.extend(compounds)

        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                unique_entities.append(entity)

        return unique_entities

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from the query (excluding stop words).
        """
        # Tokenize and clean
        words = re.findall(r'\b\w+\b', query.lower())

        # Filter stop words and short words
        keywords = [
            word for word in words
            if word not in self.STOP_WORDS and len(word) > 2
        ]

        # Deduplicate
        return list(dict.fromkeys(keywords))

    def _check_ambiguity(self, query: str) -> bool:
        """
        Determine if a query is ambiguous and needs rewriting.
        
        A query is considered ambiguous if:
        - It's very short (< 3 meaningful words)
        - Contains pronouns without clear referents
        - Lacks specific nouns or verbs
        """
        words = re.findall(r'\b\w+\b', query.lower())
        meaningful_words = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]

        # Too short
        if len(meaningful_words) < 2:
            return True

        # Contains vague pronouns
        vague_pronouns = {"it", "this", "that", "they", "those", "these"}
        if any(w in vague_pronouns for w in words) and len(meaningful_words) < 4:
            return True

        return False

    def _rewrite_query(
        self,
        query: str,
        entities: List[str],
        keywords: List[str],
        is_ambiguous: bool,
    ) -> str:
        """
        Rewrite the query for better retrieval performance.
        
        Strategies:
        - Expand abbreviations
        - Add context from entities
        - Remove question-form words for retrieval
        - Create a more search-friendly version
        """
        if not is_ambiguous:
            # For clear queries, just optimize for retrieval
            rewritten = self._optimize_for_retrieval(query)
            return rewritten

        # For ambiguous queries, expand with keywords
        if entities:
            entity_context = " ".join(entities)
            rewritten = f"{query} {entity_context}"
        elif keywords:
            rewritten = " ".join(keywords)
        else:
            rewritten = query

        return rewritten.strip()

    def _optimize_for_retrieval(self, query: str) -> str:
        """
        Transform a question into a retrieval-optimized search string.
        Removes question words and restructures for semantic search.
        """
        # Remove common question prefixes
        removal_patterns = [
            r'^(can you |could you |please |tell me )',
            r'^(what is |what are |what was |what were )',
            r'^(who is |who are |who was )',
            r'^(how do |how does |how did |how to )',
            r'^(where is |where are |where was )',
            r'^(when did |when was |when is )',
            r'^(why do |why does |why did |why is )',
        ]

        optimized = query.lower()
        for pattern in removal_patterns:
            optimized = re.sub(pattern, '', optimized, flags=re.IGNORECASE)

        # Remove trailing question mark
        optimized = optimized.rstrip('?').strip()

        # If the optimized version is too short, return original
        if len(optimized.split()) < 2:
            return query

        return optimized
