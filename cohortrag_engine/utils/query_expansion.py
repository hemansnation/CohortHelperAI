from typing import List, Dict, Any
import re

class QueryExpander:
    """Query expansion for improving retrieval with ambiguous or short questions"""

    def __init__(self, llm_model=None):
        """
        Initialize query expander

        Args:
            llm_model: LLM model instance for generating expanded queries
        """
        self.llm_model = llm_model

    def expand_query(self, query: str) -> List[str]:
        """
        Expand a query to include related terms and rephrasings

        Args:
            query: Original user query

        Returns:
            List of expanded query variations (includes original)
        """
        expanded_queries = [query]  # Always include original

        # Rule-based expansion for common patterns
        rule_based_queries = self._rule_based_expansion(query)
        expanded_queries.extend(rule_based_queries)

        # LLM-based expansion if model available
        if self.llm_model:
            llm_queries = self._llm_based_expansion(query)
            expanded_queries.extend(llm_queries)

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in expanded_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)

        return unique_queries

    def _rule_based_expansion(self, query: str) -> List[str]:
        """
        Apply rule-based query expansion

        Args:
            query: Original query

        Returns:
            List of expanded query variations
        """
        expanded = []

        # Expand common abbreviations and synonyms
        expansions = {
            "RAG": ["Retrieval Augmented Generation", "retrieval-augmented generation"],
            "LLM": ["Large Language Model", "language model"],
            "AI": ["artificial intelligence", "machine learning"],
            "ML": ["machine learning", "artificial intelligence"],
            "NLP": ["natural language processing", "text processing"],
            "API": ["application programming interface"],
            "how": ["what is the process", "what are the steps", "explain how"],
            "what": ["explain", "describe", "define"],
            "why": ["what is the reason", "explain the reasoning", "what causes"],
        }

        query_lower = query.lower()
        for abbrev, full_forms in expansions.items():
            if abbrev.lower() in query_lower:
                for full_form in full_forms:
                    expanded_query = re.sub(
                        re.escape(abbrev), full_form, query, flags=re.IGNORECASE
                    )
                    expanded.append(expanded_query)

        # Add question variations for short queries
        if len(query.split()) <= 3:
            expanded.extend([
                f"What is {query}?",
                f"How does {query} work?",
                f"Explain {query}",
                f"{query} definition",
                f"{query} examples"
            ])

        return expanded

    def _llm_based_expansion(self, query: str) -> List[str]:
        """
        Use LLM to generate query expansions

        Args:
            query: Original query

        Returns:
            List of LLM-generated query variations
        """
        if not self.llm_model:
            return []

        try:
            prompt = f"""
            Generate 3 alternative ways to ask the following question, focusing on educational content retrieval:

            Original question: "{query}"

            Requirements:
            - Keep the same intent and meaning
            - Use different words and phrasing
            - Focus on educational/learning context
            - Each variation on a separate line
            - No numbering or bullet points

            Alternative questions:
            """

            response = self.llm_model.generate_text(prompt)

            # Parse response into individual queries
            lines = response.strip().split('\n')
            expanded_queries = []

            for line in lines:
                line = line.strip()
                # Remove common prefixes
                line = re.sub(r'^[-*\d\.)\s]+', '', line).strip()
                if line and line != query:
                    expanded_queries.append(line)

            return expanded_queries[:3]  # Limit to 3 expansions

        except Exception as e:
            print(f"Error in LLM-based query expansion: {e}")
            return []

    def should_expand(self, query: str) -> bool:
        """
        Determine if a query should be expanded

        Args:
            query: User query

        Returns:
            True if query should be expanded, False otherwise
        """
        # Expand short queries
        if len(query.split()) <= 2:
            return True

        # Expand queries with common abbreviations
        abbreviations = ["RAG", "LLM", "AI", "ML", "NLP", "API"]
        if any(abbrev in query.upper() for abbrev in abbreviations):
            return True

        # Expand very generic questions
        generic_starters = ["what", "how", "why", "when", "where"]
        if any(query.lower().startswith(starter) for starter in generic_starters) and len(query.split()) <= 4:
            return True

        return False

def create_expanded_query_string(original_query: str, expanded_queries: List[str]) -> str:
    """
    Combine original and expanded queries into a single search string

    Args:
        original_query: Original user query
        expanded_queries: List of expanded query variations

    Returns:
        Combined query string for improved retrieval
    """
    # Weight original query more heavily
    all_queries = [original_query] * 2  # Double weight for original
    all_queries.extend(expanded_queries)

    # Remove duplicates and join
    unique_terms = set()
    for query in all_queries:
        unique_terms.update(query.lower().split())

    return " ".join(sorted(unique_terms))