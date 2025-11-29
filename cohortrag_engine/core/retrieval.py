from typing import List, Dict, Any, Optional, Tuple
import os
import time

# Local imports
try:
    from ..config import get_config
    from ..models.embeddings import get_embedding_model
    from ..models.llm import get_llm_model
    from .ingestion import SimpleVectorStore
    from ..utils.reranking import get_reranker
    from ..utils.query_expansion import QueryExpander
except ImportError:
    # Fallback for direct execution
    from config import get_config
    from models.embeddings import get_embedding_model
    from models.llm import get_llm_model
    from core.ingestion import SimpleVectorStore
    from utils.reranking import get_reranker
    from utils.query_expansion import QueryExpander

class RAGResponse:
    """Response object for RAG queries"""

    def __init__(self, query: str, answer: str, sources: List[Dict[str, Any]],
                 processing_time: float, confidence_score: Optional[float] = None,
                 expanded_queries: Optional[List[str]] = None):
        self.query = query
        self.answer = answer
        self.sources = sources
        self.processing_time = processing_time
        self.confidence_score = confidence_score
        self.expanded_queries = expanded_queries or []

    def __repr__(self):
        return f"RAGResponse(query='{self.query[:50]}...', sources={len(self.sources)})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "processing_time": self.processing_time,
            "confidence_score": self.confidence_score,
            "expanded_queries": self.expanded_queries
        }

class CohortRAGRetriever:
    """Main retrieval and query engine for CohortRAG"""

    def __init__(self, config=None, enable_reranking=True, enable_query_expansion=True):
        self.config = config or get_config()
        self.embedding_model = get_embedding_model(self.config)
        self.llm_model = get_llm_model(self.config)

        # Enhanced retrieval components
        self.enable_reranking = enable_reranking
        self.enable_query_expansion = enable_query_expansion

        # Initialize reranker
        if self.enable_reranking:
            self.reranker = get_reranker()
            print(f"Reranker loaded: {self.reranker.model_name}")
        else:
            self.reranker = None

        # Initialize query expander
        if self.enable_query_expansion:
            self.query_expander = QueryExpander(self.llm_model)
        else:
            self.query_expander = None

        # Load vector store
        persist_path = os.path.join(self.config.chroma_db_path, "simple_store.pkl")
        self.vector_store = SimpleVectorStore(persist_path)

        if not self.vector_store.load():
            print("Warning: No existing vector store found. Run ingestion first.")
        else:
            print(f"Loaded vector store with {len(self.vector_store.texts)} chunks")

    def query(self, question: str, top_k_final: Optional[int] = None,
              top_k_initial: Optional[int] = None, include_sources: bool = True,
              **llm_kwargs) -> RAGResponse:
        """
        Query the RAG system with enhanced two-phase retrieval

        Args:
            question: User's question
            top_k_final: Number of final chunks to use for LLM (defaults to 5)
            top_k_initial: Number of initial chunks to retrieve (defaults to 15)
            include_sources: Whether to include source information
            **llm_kwargs: Additional arguments for the LLM

        Returns:
            RAGResponse object with answer and metadata
        """
        start_time = time.time()

        # Set defaults for two-phase retrieval
        if top_k_final is None:
            top_k_final = 5
        if top_k_initial is None:
            top_k_initial = 15

        expanded_queries = []

        # Step 1: Check if vector store has data
        if not self.vector_store.texts:
            return RAGResponse(
                query=question,
                answer="I don't have any knowledge base loaded. Please run the ingestion process first.",
                sources=[],
                processing_time=time.time() - start_time
            )

        # Step 2: Query expansion (if enabled)
        search_query = question
        if self.enable_query_expansion and self.query_expander:
            if self.query_expander.should_expand(question):
                expanded_queries = self.query_expander.expand_query(question)
                # Use the first expanded query for retrieval, or original if no expansions
                search_query = expanded_queries[0] if expanded_queries else question

        # Step 3: Generate query embedding
        try:
            query_embedding = self.embedding_model.embed_text(search_query)
        except Exception as e:
            return RAGResponse(
                query=question,
                answer=f"Error generating query embedding: {e}",
                sources=[],
                processing_time=time.time() - start_time,
                expanded_queries=expanded_queries
            )

        # Step 4: Phase 1 - Initial retrieval with larger K
        try:
            initial_chunks = self.vector_store.similarity_search(
                query_embedding, k=top_k_initial
            )
        except Exception as e:
            return RAGResponse(
                query=question,
                answer=f"Error during initial retrieval: {e}",
                sources=[],
                processing_time=time.time() - start_time,
                expanded_queries=expanded_queries
            )

        if not initial_chunks:
            return RAGResponse(
                query=question,
                answer="I couldn't find any relevant information for your question.",
                sources=[],
                processing_time=time.time() - start_time,
                expanded_queries=expanded_queries
            )

        # Step 5: Phase 2 - Reranking (if enabled)
        if self.enable_reranking and self.reranker:
            try:
                final_chunks = self.reranker.rerank(
                    query=question,  # Use original query for reranking
                    documents=initial_chunks,
                    top_k=top_k_final
                )
            except Exception as e:
                print(f"Reranking failed, using similarity-based ranking: {e}")
                final_chunks = initial_chunks[:top_k_final]
        else:
            final_chunks = initial_chunks[:top_k_final]

        # Step 6: Calculate confidence score
        confidence_score = self._calculate_confidence_score(final_chunks)

        # Step 7: Prepare context for LLM
        context_texts = [chunk["text"] for chunk in final_chunks]

        # Step 8: Generate answer using LLM
        try:
            answer = self.llm_model.generate_with_context(
                query=question,
                context=context_texts,
                **llm_kwargs
            )
        except Exception as e:
            answer = f"Error generating answer: {e}"

        # Step 9: Prepare enhanced sources information
        sources = []
        if include_sources:
            for i, chunk in enumerate(final_chunks):
                source_info = {
                    "rank": i + 1,
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "similarity": chunk.get("similarity", 0.0),
                    "rerank_score": chunk.get("rerank_score", None),
                    "metadata": chunk["metadata"]
                }
                sources.append(source_info)

        processing_time = time.time() - start_time

        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            processing_time=processing_time,
            confidence_score=confidence_score,
            expanded_queries=expanded_queries
        )

    def _calculate_confidence_score(self, chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on retrieval quality

        Args:
            chunks: List of retrieved chunks with scores

        Returns:
            Confidence score between 0 and 1
        """
        if not chunks:
            return 0.0

        # Use rerank scores if available, otherwise similarity scores
        scores = []
        for chunk in chunks:
            if "rerank_score" in chunk and chunk["rerank_score"] is not None:
                scores.append(chunk["rerank_score"])
            else:
                scores.append(chunk.get("similarity", 0.0))

        if not scores:
            return 0.0

        # Calculate confidence based on:
        # 1. Average score of top chunks
        # 2. Score gap between top chunks
        # 3. Number of high-quality chunks

        avg_score = sum(scores) / len(scores)

        # Normalize different score ranges
        if any("rerank_score" in chunk for chunk in chunks):
            # BGE reranker scores are typically between -10 and 10
            # Normalize to 0-1 range
            normalized_avg = max(0, min(1, (avg_score + 10) / 20))
        else:
            # Similarity scores are typically between 0 and 1
            normalized_avg = max(0, min(1, avg_score))

        # Boost confidence if top score is significantly higher than others
        if len(scores) > 1:
            score_gap = scores[0] - scores[1] if len(scores) > 1 else 0
            gap_bonus = min(0.1, score_gap * 0.05)  # Up to 10% bonus
            normalized_avg += gap_bonus

        # Penalize if we have very few chunks
        count_penalty = 0 if len(chunks) >= 3 else 0.1 * (3 - len(chunks))
        normalized_avg -= count_penalty

        return max(0.0, min(1.0, normalized_avg))

    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced retriever statistics"""
        return {
            "total_chunks": len(self.vector_store.texts),
            "embedding_model": self.config.embedding_model,
            "llm_model": self.config.llm_model,
            "similarity_top_k": self.config.similarity_top_k,
            "vector_store_loaded": len(self.vector_store.texts) > 0,
            "reranking_enabled": self.enable_reranking,
            "query_expansion_enabled": self.enable_query_expansion,
            "reranker_model": self.reranker.model_name if self.reranker else None,
            "reranker_available": self.reranker.is_available() if self.reranker else False
        }

def main():
    """Test the retrieval system"""
    try:
        retriever = CohortRAGRetriever()
        print("CohortRAG Retrieval System")
        print("=" * 40)

        # Show stats
        stats = retriever.get_stats()
        print(f"System stats: {stats}")

        if not stats["vector_store_loaded"]:
            print("No vector store loaded. Please run ingestion first.")
            return

        # Test queries
        test_questions = [
            "What is RAG?",
            "How does document indexing work?",
            "What are the best practices for RAG implementation?",
            "How can RAG be used in education?"
        ]

        for question in test_questions:
            print(f"\nQuestion: {question}")
            print("-" * 30)

            response = retriever.query(question)
            print(f"Answer: {response.answer}")
            print(f"Processing time: {response.processing_time:.2f}s")
            print(f"Sources used: {len(response.sources)}")

            if response.sources:
                print("Top source:")
                top_source = response.sources[0]
                print(f"  - Similarity: {top_source['similarity']:.3f}")
                print(f"  - Text: {top_source['text']}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()