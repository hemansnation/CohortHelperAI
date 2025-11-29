"""
Cached retrieval system for production-ready query processing
===========================================================

This module extends the base retrieval system with intelligent caching,
cost optimization, and performance monitoring for production environments.
"""

import hashlib
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import json

# Local imports
try:
    from ..config import get_config
    from .retrieval import CohortRAGRetriever, RAGResponse
    from ..utils.performance import QueryCache, performance_monitor, sync_timer
except ImportError:
    # Fallback for direct execution
    from config import get_config
    from core.retrieval import CohortRAGRetriever, RAGResponse
    from utils.performance import QueryCache, performance_monitor, sync_timer

class CachedRAGResponse(RAGResponse):
    """Extended RAG response with cache information"""

    def __init__(self, query: str, answer: str, sources: List[Dict[str, Any]],
                 processing_time: float, confidence_score: Optional[float] = None,
                 expanded_queries: Optional[List[str]] = None,
                 cached: bool = False, cache_key: Optional[str] = None,
                 cost_info: Optional[Dict[str, Any]] = None):
        super().__init__(query, answer, sources, processing_time, confidence_score, expanded_queries)
        self.cached = cached
        self.cache_key = cache_key
        self.cost_info = cost_info or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary with cache info"""
        base_dict = super().to_dict()
        base_dict.update({
            "cached": self.cached,
            "cache_key": self.cache_key,
            "cost_info": self.cost_info
        })
        return base_dict

class ProductionCohortRAGRetriever(CohortRAGRetriever):
    """Production-ready RAG retriever with caching and cost optimization"""

    def __init__(self, config=None, enable_caching: bool = True,
                 cache_type: str = "memory", redis_url: Optional[str] = None,
                 cache_ttl: int = 3600):
        """
        Initialize production RAG retriever

        Args:
            config: Configuration object
            enable_caching: Whether to enable query caching
            cache_type: "memory" or "redis"
            redis_url: Redis connection URL (if using Redis)
            cache_ttl: Cache time-to-live in seconds
        """
        super().__init__(config)

        self.enable_caching = enable_caching
        self.cache = QueryCache(
            cache_type=cache_type,
            redis_url=redis_url,
            ttl=cache_ttl
        ) if enable_caching else None

        # Cost tracking
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.query_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Performance thresholds
        self.slow_query_threshold = 5.0  # seconds
        self.slow_queries = []

    @sync_timer("cached_rag_query")
    def query(self, question: str, top_k_final: Optional[int] = None,
              top_k_initial: Optional[int] = None, include_sources: bool = True,
              force_refresh: bool = False, **llm_kwargs) -> CachedRAGResponse:
        """
        Query with intelligent caching

        Args:
            question: User's question
            top_k_final: Number of final chunks to use for LLM (defaults to 5)
            top_k_initial: Number of initial chunks to retrieve (defaults to 15)
            include_sources: Whether to include source information
            force_refresh: Force cache refresh
            **llm_kwargs: Additional arguments for the LLM

        Returns:
            CachedRAGResponse with cache and cost information
        """
        start_time = time.time()
        self.query_count += 1

        # Generate cache key
        cache_key = self._generate_cache_key(
            question, top_k_final, top_k_initial, llm_kwargs
        )

        # Check cache first (unless force refresh)
        cached_result = None
        if self.enable_caching and not force_refresh:
            cached_result = self._get_cached_result(cache_key)

        if cached_result:
            # Cache hit
            self.cache_hits += 1
            processing_time = time.time() - start_time

            return CachedRAGResponse(
                query=question,
                answer=cached_result['answer'],
                sources=cached_result['sources'],
                processing_time=processing_time,
                confidence_score=cached_result.get('confidence_score'),
                expanded_queries=cached_result.get('expanded_queries', []),
                cached=True,
                cache_key=cache_key,
                cost_info={'tokens_used': 0, 'cost': 0.0}  # No cost for cached results
            )

        # Cache miss - perform actual retrieval
        self.cache_misses += 1

        # Call parent class query method
        response = super().query(
            question=question,
            top_k_final=top_k_final,
            top_k_initial=top_k_initial,
            include_sources=include_sources,
            **llm_kwargs
        )

        # Calculate cost information
        cost_info = self._calculate_cost(response.answer)
        self.total_tokens_used += cost_info['tokens_used']
        self.total_cost += cost_info['cost']

        # Check for slow queries
        if response.processing_time > self.slow_query_threshold:
            self.slow_queries.append({
                'question': question,
                'processing_time': response.processing_time,
                'timestamp': time.time()
            })
            logging.warning(f"Slow query detected: {response.processing_time:.2f}s - {question[:100]}")

        # Cache the result
        if self.enable_caching:
            self._cache_result(cache_key, response)

        # Return enhanced response
        return CachedRAGResponse(
            query=response.query,
            answer=response.answer,
            sources=response.sources,
            processing_time=response.processing_time,
            confidence_score=response.confidence_score,
            expanded_queries=response.expanded_queries,
            cached=False,
            cache_key=cache_key,
            cost_info=cost_info
        )

    def _generate_cache_key(self, question: str, top_k_final: Optional[int],
                          top_k_initial: Optional[int], llm_kwargs: Dict) -> str:
        """Generate cache key for query"""
        # Include relevant parameters that affect results
        key_data = {
            'question': question.lower().strip(),
            'top_k_final': top_k_final,
            'top_k_initial': top_k_initial,
            'llm_kwargs': sorted(llm_kwargs.items()) if llm_kwargs else []
        }

        # Add vector store state hash (to invalidate cache when data changes)
        vector_store_hash = self._get_vector_store_hash()
        key_data['vector_store_hash'] = vector_store_hash

        # Generate hash
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_vector_store_hash(self) -> str:
        """Get hash representing current vector store state"""
        # Simple hash based on number of texts and first few texts
        if not self.vector_store.texts:
            return "empty"

        hash_data = f"{len(self.vector_store.texts)}:{self.vector_store.texts[0][:100]}"
        if len(self.vector_store.texts) > 1:
            hash_data += f":{self.vector_store.texts[-1][:100]}"

        return hashlib.md5(hash_data.encode()).hexdigest()[:8]

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache"""
        if not self.cache:
            return None

        return self.cache.get(cache_key)

    def _cache_result(self, cache_key: str, response: RAGResponse):
        """Cache query result"""
        if not self.cache:
            return

        cache_data = {
            'answer': response.answer,
            'sources': response.sources,
            'confidence_score': response.confidence_score,
            'expanded_queries': response.expanded_queries,
            'timestamp': time.time()
        }

        self.cache.set(cache_key, cache_data)

    def _calculate_cost(self, answer: str) -> Dict[str, Any]:
        """Calculate cost information for the response"""
        # Estimate tokens (rough approximation)
        tokens_used = len(answer.split()) * 1.3  # Rough tokens-to-words ratio

        # Gemini 2.5-flash pricing (as of 2024)
        # Input: $0.000075 per 1K tokens
        # Output: $0.0003 per 1K tokens
        # Using output pricing since we're measuring generated text
        cost_per_1k_tokens = 0.0003
        cost = (tokens_used / 1000) * cost_per_1k_tokens

        return {
            'tokens_used': int(tokens_used),
            'cost': cost,
            'cost_per_1k_tokens': cost_per_1k_tokens
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache and cost statistics"""
        cache_stats = self.cache.stats() if self.cache else {}

        hit_rate = self.cache_hits / max(1, self.query_count)
        avg_cost_per_query = self.total_cost / max(1, self.query_count - self.cache_hits)

        return {
            'cache_enabled': self.enable_caching,
            'cache_backend': cache_stats,
            'query_statistics': {
                'total_queries': self.query_count,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': hit_rate,
                'cost_savings_rate': hit_rate  # Cached queries have zero cost
            },
            'cost_statistics': {
                'total_tokens_used': self.total_tokens_used,
                'total_cost': self.total_cost,
                'avg_tokens_per_query': self.total_tokens_used / max(1, self.cache_misses),
                'avg_cost_per_query': avg_cost_per_query,
                'estimated_monthly_cost': avg_cost_per_query * 30 * 24  # Assuming 1 query/hour
            },
            'performance_statistics': {
                'slow_query_threshold': self.slow_query_threshold,
                'slow_queries_count': len(self.slow_queries),
                'recent_slow_queries': self.slow_queries[-5:] if self.slow_queries else []
            }
        }

    def clear_cache(self) -> Dict[str, str]:
        """Clear all cached entries"""
        if self.cache:
            self.cache.clear()
            return {'status': 'success', 'message': 'Cache cleared'}
        else:
            return {'status': 'error', 'message': 'Cache not enabled'}

    def optimize_cache_settings(self) -> Dict[str, Any]:
        """Analyze usage and provide cache optimization recommendations"""
        stats = self.get_cache_stats()
        recommendations = []

        query_stats = stats.get('query_statistics', {})
        hit_rate = query_stats.get('hit_rate', 0)

        if hit_rate < 0.3:
            recommendations.append("Consider increasing cache TTL - low hit rate detected")

        if hit_rate > 0.8:
            recommendations.append("Cache performing well - consider reducing cache size if memory is constrained")

        cost_stats = stats.get('cost_statistics', {})
        monthly_cost = cost_stats.get('estimated_monthly_cost', 0)

        if monthly_cost > 100:  # $100/month threshold
            recommendations.append("High estimated costs - consider aggressive caching or query optimization")

        perf_stats = stats.get('performance_statistics', {})
        slow_queries = perf_stats.get('slow_queries_count', 0)

        if slow_queries > self.query_count * 0.1:  # More than 10% slow queries
            recommendations.append("Many slow queries detected - consider optimizing retrieval or enabling async processing")

        return {
            'current_performance': {
                'hit_rate': hit_rate,
                'avg_cost_per_query': cost_stats.get('avg_cost_per_query', 0),
                'slow_query_rate': slow_queries / max(1, self.query_count)
            },
            'recommendations': recommendations,
            'optimization_actions': [
                'Enable Redis caching for distributed environments',
                'Implement query preprocessing to improve cache hits',
                'Consider semantic caching for similar queries',
                'Monitor and alert on cost thresholds'
            ]
        }

    def warmup_cache(self, common_queries: List[str]) -> Dict[str, Any]:
        """Warmup cache with common queries"""
        if not self.enable_caching:
            return {'status': 'error', 'message': 'Caching not enabled'}

        warmup_start = time.time()
        successful_warmups = 0
        failed_warmups = 0

        print(f"🔥 Warming up cache with {len(common_queries)} queries...")

        for i, query in enumerate(common_queries):
            try:
                # Force refresh to ensure we get fresh results for cache
                response = self.query(query, force_refresh=True)
                if response and not response.cached:
                    successful_warmups += 1
                print(f"⏳ Warmup progress: {i+1}/{len(common_queries)}")
            except Exception as e:
                failed_warmups += 1
                logging.warning(f"Warmup failed for query '{query}': {e}")

        warmup_time = time.time() - warmup_start

        return {
            'status': 'completed',
            'successful_warmups': successful_warmups,
            'failed_warmups': failed_warmups,
            'warmup_time': warmup_time,
            'cache_entries': successful_warmups
        }

    def get_production_stats(self) -> Dict[str, Any]:
        """Get comprehensive production statistics"""
        base_stats = self.get_stats()
        cache_stats = self.get_cache_stats()
        performance_stats = performance_monitor.get_summary("cached_rag_query")

        return {
            'system_stats': base_stats,
            'cache_stats': cache_stats,
            'performance_stats': performance_stats,
            'production_readiness': {
                'cache_enabled': self.enable_caching,
                'cost_tracking': True,
                'performance_monitoring': True,
                'slow_query_detection': True
            }
        }

def main():
    """Test cached retrieval system"""
    print("🧪 Testing Cached Retrieval System")
    print("=" * 40)

    try:
        # Initialize cached retriever
        retriever = ProductionCohortRAGRetriever(
            enable_caching=True,
            cache_type="memory",
            cache_ttl=1800  # 30 minutes
        )

        # Check if knowledge base is loaded
        stats = retriever.get_stats()
        if not stats.get("vector_store_loaded", False):
            print("❌ No knowledge base loaded. Please run ingestion first.")
            return

        print(f"✅ Knowledge base ready ({stats['total_chunks']} chunks)")

        # Test queries
        test_queries = [
            "What is RAG?",
            "How does retrieval work?",
            "What are the benefits of caching?"
        ]

        for i, query in enumerate(test_queries):
            print(f"\n🔍 Query {i+1}: {query}")

            # First query (cache miss)
            response = retriever.query(query)
            print(f"   First call - Cached: {response.cached}, Time: {response.processing_time:.3f}s")
            print(f"   Cost: ${response.cost_info.get('cost', 0):.6f}")

            # Second query (cache hit)
            response2 = retriever.query(query)
            print(f"   Second call - Cached: {response2.cached}, Time: {response2.processing_time:.3f}s")

        # Display cache statistics
        print("\n📊 Cache Statistics:")
        cache_stats = retriever.get_cache_stats()
        query_stats = cache_stats.get('query_statistics', {})
        cost_stats = cache_stats.get('cost_statistics', {})

        print(f"   Hit rate: {query_stats.get('hit_rate', 0):.3f}")
        print(f"   Total cost: ${cost_stats.get('total_cost', 0):.6f}")
        print(f"   Avg cost per query: ${cost_stats.get('avg_cost_per_query', 0):.6f}")

        # Test optimization recommendations
        print("\n🎯 Optimization Recommendations:")
        optimization = retriever.optimize_cache_settings()
        for rec in optimization.get('recommendations', []):
            print(f"   • {rec}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()