from typing import List, Dict, Any, Tuple
import numpy as np

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None
    print("Warning: FlagEmbedding not available, using fallback reranker")

class BGEReranker:
    """BGE (BAAI General Embedding) Reranker for improving retrieval quality"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        """
        Initialize the BGE reranker

        Args:
            model_name: Name of the BGE reranker model to use
        """
        self.model_name = model_name
        self.reranker = None
        self._load_model()

    def _load_model(self):
        """Load the BGE reranker model"""
        if FlagReranker is None:
            print("FlagEmbedding not available, reranker disabled")
            self.reranker = None
            return

        try:
            self.reranker = FlagReranker(self.model_name, use_fp16=True)
            print(f"Loaded BGE reranker: {self.model_name}")
        except Exception as e:
            print(f"Error loading BGE reranker: {e}")
            self.reranker = None

    def rerank(self, query: str, documents: List[Dict[str, Any]],
               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to the query

        Args:
            query: The user's question
            documents: List of retrieved documents with text and metadata
            top_k: Number of top documents to return after reranking

        Returns:
            Reranked documents with rerank_score added to each document
        """
        if not self.reranker or not documents:
            return documents[:top_k]

        try:
            # Prepare input pairs for reranking
            doc_texts = [doc["text"] for doc in documents]
            query_doc_pairs = [[query, text] for text in doc_texts]

            # Get reranking scores
            scores = self.reranker.compute_score(query_doc_pairs)

            # Handle single document case
            if not isinstance(scores, list):
                scores = [scores]

            # Add rerank scores to documents
            for i, doc in enumerate(documents):
                doc["rerank_score"] = float(scores[i])

            # Sort by rerank score (descending) and return top_k
            reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
            return reranked_docs[:top_k]

        except Exception as e:
            print(f"Error during reranking: {e}")
            # Fallback to original order if reranking fails
            return documents[:top_k]

    def is_available(self) -> bool:
        """Check if reranker is available"""
        return self.reranker is not None

class FallbackReranker:
    """Fallback reranker that uses similarity scores when BGE is not available"""

    def __init__(self):
        self.model_name = "similarity_fallback"

    def rerank(self, query: str, documents: List[Dict[str, Any]],
               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank using existing similarity scores

        Args:
            query: The user's question (unused in fallback)
            documents: List of retrieved documents
            top_k: Number of top documents to return

        Returns:
            Documents sorted by similarity score with rerank_score = similarity
        """
        # Use similarity score as rerank score
        for doc in documents:
            doc["rerank_score"] = doc.get("similarity", 0.0)

        # Sort by similarity (already should be sorted, but ensure it)
        sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_docs[:top_k]

    def is_available(self) -> bool:
        """Fallback is always available"""
        return True

def get_reranker(model_name: str = "BAAI/bge-reranker-large"):
    """
    Get a reranker instance with fallback handling

    Args:
        model_name: BGE model name to use

    Returns:
        BGE reranker or fallback reranker if BGE fails to load
    """
    try:
        reranker = BGEReranker(model_name)
        if reranker.is_available():
            return reranker
    except Exception as e:
        print(f"Failed to load BGE reranker: {e}")

    print("Using fallback similarity-based reranker")
    return FallbackReranker()