"""
Performance optimization utilities for CohortRAG Engine
======================================================

This module provides async processing, caching, and performance monitoring
for production-ready RAG systems.
"""

import asyncio
import time
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

# Optional Redis for distributed caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Local imports
try:
    from ..config import get_config
except ImportError:
    from config import get_config

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None

class PerformanceMonitor:
    """Monitor and track performance metrics"""

    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.lock = threading.Lock()

    def start_operation(self, operation: str) -> dict:
        """Start tracking an operation"""
        return {
            'operation': operation,
            'start_time': time.time()
        }

    def end_operation(self, context: dict, success: bool = True,
                     tokens_used: Optional[int] = None,
                     cost: Optional[float] = None) -> PerformanceMetrics:
        """End tracking an operation"""
        end_time = time.time()
        duration = end_time - context['start_time']

        metric = PerformanceMetrics(
            operation=context['operation'],
            start_time=context['start_time'],
            end_time=end_time,
            duration=duration,
            success=success,
            tokens_used=tokens_used,
            cost=cost
        )

        with self.lock:
            self.metrics.append(metric)

        return metric

    def get_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary"""
        with self.lock:
            if operation:
                filtered_metrics = [m for m in self.metrics if m.operation == operation]
            else:
                filtered_metrics = self.metrics

        if not filtered_metrics:
            return {}

        durations = [m.duration for m in filtered_metrics]
        success_count = sum(1 for m in filtered_metrics if m.success)

        return {
            'operation': operation or 'all',
            'total_operations': len(filtered_metrics),
            'success_rate': success_count / len(filtered_metrics),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_tokens': sum(m.tokens_used or 0 for m in filtered_metrics),
            'total_cost': sum(m.cost or 0 for m in filtered_metrics)
        }

# Global performance monitor
performance_monitor = PerformanceMonitor()

class AsyncIngestionProcessor:
    """Asynchronous document processing for large-scale ingestion"""

    def __init__(self, max_workers: int = 4, chunk_size: int = 100):
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.semaphore = asyncio.Semaphore(max_workers)

    async def process_documents_async(self, documents: List[Path],
                                    processor_func: Callable,
                                    progress_callback: Optional[Callable] = None) -> List[Any]:
        """
        Process documents asynchronously

        Args:
            documents: List of document paths
            processor_func: Function to process each document
            progress_callback: Optional callback for progress updates

        Returns:
            List of processing results
        """
        # Split documents into chunks
        document_chunks = [
            documents[i:i + self.chunk_size]
            for i in range(0, len(documents), self.chunk_size)
        ]

        results = []
        total_chunks = len(document_chunks)

        for i, chunk in enumerate(document_chunks):
            chunk_results = await self._process_chunk_async(
                chunk, processor_func
            )
            results.extend(chunk_results)

            if progress_callback:
                progress_callback(i + 1, total_chunks, len(results))

        return results

    async def _process_chunk_async(self, documents: List[Path],
                                 processor_func: Callable) -> List[Any]:
        """Process a chunk of documents asynchronously"""
        async with self.semaphore:
            tasks = []
            for doc in documents:
                task = asyncio.create_task(
                    self._process_single_document_async(doc, processor_func)
                )
                tasks.append(task)

            return await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_single_document_async(self, document: Path,
                                           processor_func: Callable) -> Any:
        """Process a single document asynchronously"""
        loop = asyncio.get_event_loop()

        # Run CPU-intensive work in thread pool
        with ThreadPoolExecutor() as executor:
            try:
                result = await loop.run_in_executor(
                    executor, processor_func, document
                )
                return result
            except Exception as e:
                return {'error': str(e), 'document': str(document)}

class QueryCache:
    """Query caching system with multiple backends"""

    def __init__(self, cache_type: str = "memory", redis_url: Optional[str] = None,
                 ttl: int = 3600, max_size: int = 1000):
        """
        Initialize query cache

        Args:
            cache_type: "memory" or "redis"
            redis_url: Redis connection URL (if using Redis)
            ttl: Time to live in seconds
            max_size: Maximum cache size (for memory cache)
        """
        self.cache_type = cache_type
        self.ttl = ttl
        self.max_size = max_size

        if cache_type == "redis" and REDIS_AVAILABLE and redis_url:
            self.redis_client = redis.from_url(redis_url)
            self._cache = None
        else:
            self.redis_client = None
            self._cache = {}
            self._access_times = {}
            self._lock = threading.Lock()

    def _get_cache_key(self, query: str, context_hash: Optional[str] = None) -> str:
        """Generate cache key for query"""
        key_data = query
        if context_hash:
            key_data += f":{context_hash}"

        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, query: str, context_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get cached result for query"""
        cache_key = self._get_cache_key(query, context_hash)

        if self.redis_client:
            return self._redis_get(cache_key)
        else:
            return self._memory_get(cache_key)

    def set(self, query: str, result: Dict[str, Any], context_hash: Optional[str] = None):
        """Cache result for query"""
        cache_key = self._get_cache_key(query, context_hash)

        if self.redis_client:
            self._redis_set(cache_key, result)
        else:
            self._memory_set(cache_key, result)

    def _redis_get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get from Redis cache"""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logging.warning(f"Redis cache get error: {e}")
        return None

    def _redis_set(self, cache_key: str, result: Dict[str, Any]):
        """Set in Redis cache"""
        try:
            serialized_result = json.dumps(result, default=str)
            self.redis_client.setex(cache_key, self.ttl, serialized_result)
        except Exception as e:
            logging.warning(f"Redis cache set error: {e}")

    def _memory_get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get from memory cache"""
        with self._lock:
            if cache_key in self._cache:
                # Check TTL
                cached_time = self._access_times[cache_key]
                if time.time() - cached_time < self.ttl:
                    self._access_times[cache_key] = time.time()  # Update access time
                    return self._cache[cache_key]
                else:
                    # Expired
                    del self._cache[cache_key]
                    del self._access_times[cache_key]
        return None

    def _memory_set(self, cache_key: str, result: Dict[str, Any]):
        """Set in memory cache"""
        with self._lock:
            # Implement LRU eviction if cache is full
            if len(self._cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self._access_times.keys(),
                               key=lambda k: self._access_times[k])
                del self._cache[oldest_key]
                del self._access_times[oldest_key]

            self._cache[cache_key] = result
            self._access_times[cache_key] = time.time()

    def clear(self):
        """Clear all cached entries"""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logging.warning(f"Redis cache clear error: {e}")
        else:
            with self._lock:
                self._cache.clear()
                self._access_times.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.redis_client:
            try:
                info = self.redis_client.info()
                return {
                    'cache_type': 'redis',
                    'connected': True,
                    'used_memory': info.get('used_memory_human', 'unknown'),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                }
            except Exception as e:
                return {'cache_type': 'redis', 'connected': False, 'error': str(e)}
        else:
            with self._lock:
                return {
                    'cache_type': 'memory',
                    'entries': len(self._cache),
                    'max_size': self.max_size,
                    'utilization': len(self._cache) / self.max_size
                }

class BatchProcessor:
    """Batch processing for efficient operations"""

    def __init__(self, batch_size: int = 50, max_concurrent: int = 4):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent

    async def process_batch_async(self, items: List[Any],
                                processor: Callable,
                                progress_callback: Optional[Callable] = None) -> List[Any]:
        """Process items in batches asynchronously"""
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = []

        async def process_single_batch(batch_idx: int, batch: List[Any]):
            async with semaphore:
                batch_results = []
                for item in batch:
                    try:
                        result = await self._run_in_thread(processor, item)
                        batch_results.append(result)
                    except Exception as e:
                        batch_results.append({'error': str(e), 'item': str(item)})

                if progress_callback:
                    progress_callback(batch_idx + 1, len(batches), len(batch_results))

                return batch_results

        # Process all batches concurrently
        tasks = [
            process_single_batch(i, batch)
            for i, batch in enumerate(batches)
        ]

        batch_results = await asyncio.gather(*tasks)

        # Flatten results
        for batch_result in batch_results:
            results.extend(batch_result)

        return results

    async def _run_in_thread(self, func: Callable, item: Any) -> Any:
        """Run function in thread pool"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, func, item)

class MemoryMonitor:
    """Monitor memory usage during operations"""

    def __init__(self):
        self.peak_memory = 0
        self.start_memory = 0

    def start_monitoring(self):
        """Start memory monitoring"""
        try:
            import psutil
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory
        except ImportError:
            logging.warning("psutil not available for memory monitoring")

    def update_peak(self):
        """Update peak memory usage"""
        try:
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, current_memory)
        except ImportError:
            pass

    def get_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB

            return {
                'current_mb': current_memory,
                'peak_mb': self.peak_memory,
                'increase_mb': current_memory - self.start_memory,
                'peak_increase_mb': self.peak_memory - self.start_memory
            }
        except ImportError:
            return {'error': 'psutil not available'}

def async_timer(func_name: str):
    """Decorator for timing async functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            context = performance_monitor.start_operation(func_name)
            try:
                result = await func(*args, **kwargs)
                performance_monitor.end_operation(context, success=True)
                return result
            except Exception as e:
                performance_monitor.end_operation(context, success=False)
                raise e
        return wrapper
    return decorator

def sync_timer(func_name: str):
    """Decorator for timing synchronous functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            context = performance_monitor.start_operation(func_name)
            try:
                result = func(*args, **kwargs)
                performance_monitor.end_operation(context, success=True)
                return result
            except Exception as e:
                performance_monitor.end_operation(context, success=False)
                raise e
        return wrapper
    return decorator

class ProductionOptimizer:
    """Main production optimization coordinator"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.cache = QueryCache(
            cache_type=getattr(config, 'cache_type', 'memory'),
            redis_url=getattr(config, 'redis_url', None),
            ttl=getattr(config, 'cache_ttl', 3600)
        )
        self.async_processor = AsyncIngestionProcessor(
            max_workers=getattr(config, 'max_workers', 4)
        )
        self.batch_processor = BatchProcessor(
            batch_size=getattr(config, 'batch_size', 50)
        )
        self.memory_monitor = MemoryMonitor()

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        return {
            'cache_stats': self.cache.stats(),
            'performance_summary': performance_monitor.get_summary(),
            'memory_usage': self.memory_monitor.get_usage(),
            'configuration': {
                'cache_type': self.cache.cache_type,
                'max_workers': self.async_processor.max_workers,
                'batch_size': self.batch_processor.batch_size
            }
        }

    def optimize_for_production(self) -> Dict[str, str]:
        """Apply production optimizations"""
        recommendations = []

        # Check cache hit rate
        cache_stats = self.cache.stats()
        if cache_stats.get('cache_type') == 'redis':
            hits = cache_stats.get('keyspace_hits', 0)
            misses = cache_stats.get('keyspace_misses', 0)
            if hits + misses > 0:
                hit_rate = hits / (hits + misses)
                if hit_rate < 0.5:
                    recommendations.append("Consider increasing cache TTL or size")

        # Check performance metrics
        perf_summary = performance_monitor.get_summary()
        if perf_summary and perf_summary.get('avg_duration', 0) > 2.0:
            recommendations.append("Consider enabling async processing for large operations")

        # Check memory usage
        memory_stats = self.memory_monitor.get_usage()
        if memory_stats.get('peak_mb', 0) > 1000:  # 1GB
            recommendations.append("Consider implementing memory-efficient batch processing")

        return {
            'status': 'optimized',
            'recommendations': recommendations,
            'applied_optimizations': [
                'Query caching enabled',
                'Async processing configured',
                'Performance monitoring active'
            ]
        }