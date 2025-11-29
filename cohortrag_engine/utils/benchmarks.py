"""
Performance benchmarking suite for CohortRAG Engine
==================================================

This module provides comprehensive benchmarking tools for measuring
and optimizing RAG system performance in production environments.
"""

import time
import asyncio
import statistics
import threading
import psutil
import gc
from typing import List, Dict, Any, Optional, Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
from datetime import datetime

# Local imports
try:
    from ..config import get_config
    from .performance import performance_monitor, MemoryMonitor
    from .cost_modeling import token_tracker
    from ..core.retrieval import CohortRAGRetriever
    from ..core.cached_retrieval import ProductionCohortRAGRetriever
    from ..core.async_ingestion import AsyncCohortRAGIngestion
except ImportError:
    # Fallback for direct execution
    from config import get_config
    from utils.performance import performance_monitor, MemoryMonitor
    from utils.cost_modeling import token_tracker
    from core.retrieval import CohortRAGRetriever
    from core.cached_retrieval import ProductionCohortRAGRetriever
    from core.async_ingestion import AsyncCohortRAGIngestion

@dataclass
class BenchmarkResult:
    """Individual benchmark test result"""
    test_name: str
    timestamp: float
    duration: float
    success: bool
    metrics: Dict[str, Any]
    error: Optional[str] = None

@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""
    suite_name: str
    start_time: float
    end_time: float
    total_duration: float
    results: List[BenchmarkResult]
    summary: Dict[str, Any]
    system_info: Dict[str, Any]

class QueryBenchmark:
    """Benchmark query performance and accuracy"""

    def __init__(self, retriever, test_queries: Optional[List[str]] = None):
        self.retriever = retriever
        self.test_queries = test_queries or self._get_default_queries()
        self.results = []

    def _get_default_queries(self) -> List[str]:
        """Get default test queries for benchmarking"""
        return [
            "What is RAG?",
            "How does retrieval augmented generation work?",
            "What are the benefits of using vector databases?",
            "Explain document chunking strategies",
            "How to optimize embedding models for education?",
            "What are best practices for prompt engineering?",
            "Compare different vector similarity metrics",
            "How to implement caching in RAG systems?",
            "What is semantic search and how does it work?",
            "Explain the role of LLMs in educational AI"
        ]

    def run_latency_benchmark(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark query latency"""
        start_time = time.time()
        latencies = []
        successes = 0
        errors = []

        print(f"🚀 Running latency benchmark ({iterations} queries)...")

        for i in range(iterations):
            query = self.test_queries[i % len(self.test_queries)]

            try:
                query_start = time.time()
                response = self.retriever.query(query)
                latency = time.time() - query_start

                latencies.append(latency)
                successes += 1

                if i % 10 == 0:
                    print(f"   Progress: {i+1}/{iterations}")

            except Exception as e:
                errors.append(str(e))
                logging.warning(f"Query failed: {e}")

        # Calculate statistics
        if latencies:
            metrics = {
                'total_queries': iterations,
                'successful_queries': successes,
                'failed_queries': len(errors),
                'success_rate': successes / iterations,
                'avg_latency': statistics.mean(latencies),
                'median_latency': statistics.median(latencies),
                'min_latency': min(latencies),
                'max_latency': max(latencies),
                'p95_latency': statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies),
                'p99_latency': statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else max(latencies),
                'queries_per_second': successes / (time.time() - start_time),
                'errors': errors[:5]  # First 5 errors
            }
        else:
            metrics = {
                'total_queries': iterations,
                'successful_queries': 0,
                'failed_queries': len(errors),
                'success_rate': 0.0,
                'errors': errors
            }

        return BenchmarkResult(
            test_name="latency_benchmark",
            timestamp=start_time,
            duration=time.time() - start_time,
            success=successes > 0,
            metrics=metrics
        )

    def run_throughput_benchmark(self, concurrent_users: int = 10,
                                duration_seconds: int = 60) -> BenchmarkResult:
        """Benchmark query throughput under load"""
        start_time = time.time()
        end_time = start_time + duration_seconds

        print(f"🔥 Running throughput benchmark ({concurrent_users} users, {duration_seconds}s)...")

        results_lock = threading.Lock()
        all_results = []

        def worker_thread(worker_id: int):
            """Worker thread for concurrent queries"""
            worker_results = []
            query_count = 0

            while time.time() < end_time:
                query = self.test_queries[query_count % len(self.test_queries)]

                try:
                    query_start = time.time()
                    response = self.retriever.query(query)
                    latency = time.time() - query_start

                    worker_results.append({
                        'worker_id': worker_id,
                        'latency': latency,
                        'timestamp': query_start,
                        'success': True
                    })

                    query_count += 1

                except Exception as e:
                    worker_results.append({
                        'worker_id': worker_id,
                        'error': str(e),
                        'timestamp': time.time(),
                        'success': False
                    })

            with results_lock:
                all_results.extend(worker_results)

        # Start worker threads
        threads = []
        for i in range(concurrent_users):
            thread = threading.Thread(target=worker_thread, args=(i,))
            thread.start()
            threads.append(thread)

        # Wait for completion
        for thread in threads:
            thread.join()

        # Calculate metrics
        successful_queries = [r for r in all_results if r['success']]
        failed_queries = [r for r in all_results if not r['success']]

        total_duration = time.time() - start_time

        if successful_queries:
            latencies = [r['latency'] for r in successful_queries]

            metrics = {
                'duration_seconds': duration_seconds,
                'concurrent_users': concurrent_users,
                'total_queries': len(all_results),
                'successful_queries': len(successful_queries),
                'failed_queries': len(failed_queries),
                'success_rate': len(successful_queries) / len(all_results),
                'avg_latency': statistics.mean(latencies),
                'p95_latency': statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies),
                'total_throughput': len(successful_queries) / total_duration,
                'per_user_throughput': len(successful_queries) / (total_duration * concurrent_users),
                'errors_per_second': len(failed_queries) / total_duration
            }
        else:
            metrics = {
                'duration_seconds': duration_seconds,
                'concurrent_users': concurrent_users,
                'total_queries': len(all_results),
                'successful_queries': 0,
                'failed_queries': len(failed_queries),
                'success_rate': 0.0,
                'errors': [r.get('error') for r in failed_queries[:5]]
            }

        return BenchmarkResult(
            test_name="throughput_benchmark",
            timestamp=start_time,
            duration=total_duration,
            success=len(successful_queries) > 0,
            metrics=metrics
        )

    def run_accuracy_benchmark(self, ground_truth_pairs: Optional[List[Dict]] = None) -> BenchmarkResult:
        """Benchmark retrieval accuracy"""
        start_time = time.time()

        if not ground_truth_pairs:
            ground_truth_pairs = self._create_sample_ground_truth()

        print(f"🎯 Running accuracy benchmark ({len(ground_truth_pairs)} Q&A pairs)...")

        accuracy_results = []

        for i, pair in enumerate(ground_truth_pairs):
            query = pair['question']
            expected_keywords = pair.get('expected_keywords', [])

            try:
                response = self.retriever.query(query)

                # Simple accuracy check: keyword presence in answer
                keyword_matches = 0
                for keyword in expected_keywords:
                    if keyword.lower() in response.answer.lower():
                        keyword_matches += 1

                accuracy = keyword_matches / len(expected_keywords) if expected_keywords else 0.5

                accuracy_results.append({
                    'query': query,
                    'accuracy': accuracy,
                    'keyword_matches': keyword_matches,
                    'total_keywords': len(expected_keywords),
                    'answer_length': len(response.answer),
                    'sources_used': len(response.sources),
                    'confidence': getattr(response, 'confidence_score', None)
                })

            except Exception as e:
                accuracy_results.append({
                    'query': query,
                    'accuracy': 0.0,
                    'error': str(e)
                })

        # Calculate aggregate metrics
        successful_tests = [r for r in accuracy_results if 'error' not in r]
        if successful_tests:
            accuracies = [r['accuracy'] for r in successful_tests]

            metrics = {
                'total_tests': len(ground_truth_pairs),
                'successful_tests': len(successful_tests),
                'avg_accuracy': statistics.mean(accuracies),
                'min_accuracy': min(accuracies),
                'max_accuracy': max(accuracies),
                'accuracy_variance': statistics.variance(accuracies) if len(accuracies) > 1 else 0,
                'high_accuracy_tests': len([a for a in accuracies if a > 0.8]),
                'low_accuracy_tests': len([a for a in accuracies if a < 0.3]),
                'avg_sources_per_query': statistics.mean([r['sources_used'] for r in successful_tests]),
                'detailed_results': successful_tests
            }
        else:
            metrics = {
                'total_tests': len(ground_truth_pairs),
                'successful_tests': 0,
                'avg_accuracy': 0.0,
                'errors': [r.get('error') for r in accuracy_results if 'error' in r]
            }

        return BenchmarkResult(
            test_name="accuracy_benchmark",
            timestamp=start_time,
            duration=time.time() - start_time,
            success=len(successful_tests) > 0,
            metrics=metrics
        )

    def _create_sample_ground_truth(self) -> List[Dict]:
        """Create sample ground truth data for accuracy testing"""
        return [
            {
                'question': 'What is RAG?',
                'expected_keywords': ['retrieval', 'augmented', 'generation', 'knowledge', 'documents']
            },
            {
                'question': 'How does vector search work?',
                'expected_keywords': ['embedding', 'similarity', 'distance', 'vector', 'search']
            },
            {
                'question': 'What are the benefits of caching?',
                'expected_keywords': ['performance', 'speed', 'cost', 'latency', 'efficiency']
            },
            {
                'question': 'Explain document chunking',
                'expected_keywords': ['chunks', 'split', 'overlap', 'size', 'processing']
            },
            {
                'question': 'What is semantic search?',
                'expected_keywords': ['meaning', 'context', 'semantic', 'understanding', 'similarity']
            }
        ]

class SystemBenchmark:
    """Benchmark system resources and scalability"""

    def __init__(self):
        self.memory_monitor = MemoryMonitor()

    def run_memory_benchmark(self, operations: int = 1000) -> BenchmarkResult:
        """Benchmark memory usage under load"""
        start_time = time.time()

        print(f"🧠 Running memory benchmark ({operations} operations)...")

        self.memory_monitor.start_monitoring()
        initial_memory = self.memory_monitor.get_usage()

        # Simulate heavy operations
        test_data = []
        for i in range(operations):
            # Create some memory load
            test_data.append([1.0] * 1000)  # 1000 floats

            if i % 100 == 0:
                self.memory_monitor.update_peak()
                print(f"   Memory test progress: {i+1}/{operations}")

        final_memory = self.memory_monitor.get_usage()

        # Clean up
        del test_data
        gc.collect()

        metrics = {
            'operations': operations,
            'initial_memory_mb': initial_memory.get('current_mb', 0),
            'peak_memory_mb': final_memory.get('peak_mb', 0),
            'memory_increase_mb': final_memory.get('peak_increase_mb', 0),
            'memory_efficiency': operations / max(1, final_memory.get('peak_increase_mb', 1))  # Ops per MB
        }

        return BenchmarkResult(
            test_name="memory_benchmark",
            timestamp=start_time,
            duration=time.time() - start_time,
            success=True,
            metrics=metrics
        )

    def run_cpu_benchmark(self, duration_seconds: int = 30) -> BenchmarkResult:
        """Benchmark CPU usage"""
        start_time = time.time()
        end_time = start_time + duration_seconds

        print(f"⚙️ Running CPU benchmark ({duration_seconds}s)...")

        cpu_measurements = []
        memory_measurements = []

        def cpu_intensive_task():
            """CPU-intensive task for benchmarking"""
            result = 0
            while time.time() < end_time:
                # Simulate processing
                for i in range(10000):
                    result += i ** 2
                time.sleep(0.01)  # Small break
            return result

        def monitor_system():
            """Monitor system resources"""
            while time.time() < end_time:
                try:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_percent = psutil.virtual_memory().percent

                    cpu_measurements.append({
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory_percent
                    })
                except Exception:
                    pass

        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_system)
        monitor_thread.start()

        # Run CPU intensive task
        task_result = cpu_intensive_task()

        # Wait for monitoring to complete
        monitor_thread.join()

        # Calculate metrics
        if cpu_measurements:
            cpu_values = [m['cpu_percent'] for m in cpu_measurements]
            memory_values = [m['memory_percent'] for m in cpu_measurements]

            metrics = {
                'duration_seconds': duration_seconds,
                'avg_cpu_percent': statistics.mean(cpu_values),
                'max_cpu_percent': max(cpu_values),
                'min_cpu_percent': min(cpu_values),
                'avg_memory_percent': statistics.mean(memory_values),
                'max_memory_percent': max(memory_values),
                'cpu_efficiency': task_result / duration_seconds,  # Operations per second
                'measurements_count': len(cpu_measurements)
            }
        else:
            metrics = {
                'duration_seconds': duration_seconds,
                'error': 'Could not collect system metrics'
            }

        return BenchmarkResult(
            test_name="cpu_benchmark",
            timestamp=start_time,
            duration=time.time() - start_time,
            success=len(cpu_measurements) > 0,
            metrics=metrics
        )

class ComprehensiveBenchmark:
    """Run comprehensive benchmark suite"""

    def __init__(self, retriever=None, enable_caching: bool = True):
        self.retriever = retriever
        self.enable_caching = enable_caching
        self.results = []

    def run_full_benchmark_suite(self) -> BenchmarkSuite:
        """Run complete benchmark suite"""
        suite_start = time.time()

        print("🏁 Starting Comprehensive Benchmark Suite")
        print("=" * 60)

        # Initialize retriever if not provided
        if not self.retriever:
            if self.enable_caching:
                self.retriever = ProductionCohortRAGRetriever(enable_caching=True)
            else:
                self.retriever = CohortRAGRetriever()

        # Check if knowledge base is loaded
        stats = self.retriever.get_stats()
        if not stats.get("vector_store_loaded", False):
            print("❌ No knowledge base loaded. Cannot run benchmarks.")
            return BenchmarkSuite(
                suite_name="comprehensive",
                start_time=suite_start,
                end_time=time.time(),
                total_duration=0,
                results=[],
                summary={'error': 'No knowledge base loaded'},
                system_info=self._get_system_info()
            )

        print(f"✅ Knowledge base ready ({stats['total_chunks']} chunks)")

        # Run benchmarks
        benchmark_results = []

        # 1. Query Benchmarks
        print("\n1️⃣ Query Performance Benchmarks")
        query_benchmark = QueryBenchmark(self.retriever)

        latency_result = query_benchmark.run_latency_benchmark(iterations=50)
        benchmark_results.append(latency_result)

        throughput_result = query_benchmark.run_throughput_benchmark(concurrent_users=5, duration_seconds=30)
        benchmark_results.append(throughput_result)

        accuracy_result = query_benchmark.run_accuracy_benchmark()
        benchmark_results.append(accuracy_result)

        # 2. System Benchmarks
        print("\n2️⃣ System Performance Benchmarks")
        system_benchmark = SystemBenchmark()

        memory_result = system_benchmark.run_memory_benchmark(operations=500)
        benchmark_results.append(memory_result)

        cpu_result = system_benchmark.run_cpu_benchmark(duration_seconds=15)
        benchmark_results.append(cpu_result)

        # 3. Caching Benchmark (if enabled)
        if self.enable_caching and hasattr(self.retriever, 'get_cache_stats'):
            print("\n3️⃣ Caching Performance Benchmark")
            caching_result = self._benchmark_caching()
            benchmark_results.append(caching_result)

        # Generate summary
        suite_end = time.time()
        summary = self._generate_summary(benchmark_results)

        return BenchmarkSuite(
            suite_name="comprehensive",
            start_time=suite_start,
            end_time=suite_end,
            total_duration=suite_end - suite_start,
            results=benchmark_results,
            summary=summary,
            system_info=self._get_system_info()
        )

    def _benchmark_caching(self) -> BenchmarkResult:
        """Benchmark caching performance"""
        start_time = time.time()

        print("   Testing cache performance...")

        test_queries = ["What is RAG?", "How does caching work?", "What are embeddings?"]

        # Clear cache first
        if hasattr(self.retriever, 'clear_cache'):
            self.retriever.clear_cache()

        cache_misses = []
        cache_hits = []

        # First pass - cache misses
        for query in test_queries:
            query_start = time.time()
            response = self.retriever.query(query)
            latency = time.time() - query_start
            cache_misses.append(latency)

        # Second pass - cache hits
        for query in test_queries:
            query_start = time.time()
            response = self.retriever.query(query)
            latency = time.time() - query_start
            cache_hits.append(latency)

        # Calculate cache performance
        avg_miss_latency = statistics.mean(cache_misses)
        avg_hit_latency = statistics.mean(cache_hits)
        cache_speedup = avg_miss_latency / avg_hit_latency if avg_hit_latency > 0 else 0

        cache_stats = self.retriever.get_cache_stats() if hasattr(self.retriever, 'get_cache_stats') else {}

        metrics = {
            'cache_miss_latency': avg_miss_latency,
            'cache_hit_latency': avg_hit_latency,
            'cache_speedup': cache_speedup,
            'cache_hit_rate': cache_stats.get('query_statistics', {}).get('hit_rate', 0),
            'cache_backend': cache_stats.get('cache_backend', {}).get('cache_type', 'unknown')
        }

        return BenchmarkResult(
            test_name="caching_benchmark",
            timestamp=start_time,
            duration=time.time() - start_time,
            success=True,
            metrics=metrics
        )

    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate benchmark summary"""
        successful_tests = [r for r in results if r.success]
        failed_tests = [r for r in results if not r.success]

        # Extract key metrics
        latency_result = next((r for r in results if r.test_name == "latency_benchmark"), None)
        throughput_result = next((r for r in results if r.test_name == "throughput_benchmark"), None)
        accuracy_result = next((r for r in results if r.test_name == "accuracy_benchmark"), None)

        summary = {
            'total_tests': len(results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'overall_success_rate': len(successful_tests) / len(results) if results else 0,
        }

        # Key performance indicators
        if latency_result and latency_result.success:
            summary['avg_query_latency'] = latency_result.metrics.get('avg_latency', 0)
            summary['p95_query_latency'] = latency_result.metrics.get('p95_latency', 0)

        if throughput_result and throughput_result.success:
            summary['max_throughput'] = throughput_result.metrics.get('total_throughput', 0)

        if accuracy_result and accuracy_result.success:
            summary['avg_accuracy'] = accuracy_result.metrics.get('avg_accuracy', 0)

        # Performance grade
        grade = self._calculate_performance_grade(summary)
        summary['performance_grade'] = grade

        return summary

    def _calculate_performance_grade(self, summary: Dict[str, Any]) -> str:
        """Calculate overall performance grade"""
        scores = []

        # Latency score (lower is better)
        avg_latency = summary.get('avg_query_latency', 5.0)
        if avg_latency < 0.5:
            scores.append(95)
        elif avg_latency < 1.0:
            scores.append(85)
        elif avg_latency < 2.0:
            scores.append(75)
        elif avg_latency < 5.0:
            scores.append(65)
        else:
            scores.append(50)

        # Throughput score (higher is better)
        throughput = summary.get('max_throughput', 0)
        if throughput > 100:
            scores.append(95)
        elif throughput > 50:
            scores.append(85)
        elif throughput > 20:
            scores.append(75)
        elif throughput > 5:
            scores.append(65)
        else:
            scores.append(50)

        # Accuracy score
        accuracy = summary.get('avg_accuracy', 0)
        scores.append(accuracy * 100)

        # Overall grade
        avg_score = sum(scores) / len(scores) if scores else 0

        if avg_score >= 90:
            return "A+ (Excellent)"
        elif avg_score >= 80:
            return "A (Good)"
        elif avg_score >= 70:
            return "B (Acceptable)"
        elif avg_score >= 60:
            return "C (Needs Improvement)"
        else:
            return "D (Poor)"

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
                'timestamp': datetime.now().isoformat()
            }
        except Exception:
            return {'error': 'Could not collect system info'}

    def save_results(self, suite: BenchmarkSuite, output_file: str):
        """Save benchmark results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)

        # Convert to serializable format
        serializable_suite = {
            'suite_name': suite.suite_name,
            'start_time': suite.start_time,
            'end_time': suite.end_time,
            'total_duration': suite.total_duration,
            'results': [asdict(result) for result in suite.results],
            'summary': suite.summary,
            'system_info': suite.system_info
        }

        with open(output_path, 'w') as f:
            json.dump(serializable_suite, f, indent=2, default=str)

        print(f"📊 Benchmark results saved to: {output_path}")

def main():
    """Run comprehensive benchmark suite"""
    print("🏁 CohortRAG Performance Benchmark Suite")
    print("=" * 50)

    try:
        # Run comprehensive benchmarks
        benchmark = ComprehensiveBenchmark(enable_caching=True)
        suite = benchmark.run_full_benchmark_suite()

        # Display results
        print(f"\n🎯 BENCHMARK RESULTS SUMMARY")
        print("=" * 50)
        print(f"📊 Performance Grade: {suite.summary.get('performance_grade', 'Unknown')}")
        print(f"⚡ Avg Query Latency: {suite.summary.get('avg_query_latency', 0):.3f}s")
        print(f"🚀 Max Throughput: {suite.summary.get('max_throughput', 0):.1f} queries/sec")
        print(f"🎯 Avg Accuracy: {suite.summary.get('avg_accuracy', 0):.3f}")
        print(f"✅ Success Rate: {suite.summary.get('overall_success_rate', 0):.3f}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"benchmark_results_{timestamp}.json"
        benchmark.save_results(suite, output_file)

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()