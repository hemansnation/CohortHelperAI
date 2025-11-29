#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CohortRAG Engine - Main Interface

This is the main entry point for the CohortRAG Engine.
Run this script to test ingestion and querying functionality.
"""

import sys
import os
from pathlib import Path

# Add the cohortrag_engine to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from core.ingestion import CohortRAGIngestionPipeline
from core.retrieval import CohortRAGRetriever
from core.evaluation import RAGASEvaluator
from core.async_ingestion import AsyncCohortRAGIngestion
from core.cached_retrieval import ProductionCohortRAGRetriever
from utils.cost_modeling import GeminiCostTracker, CostOptimizer
from utils.benchmarks import ComprehensiveBenchmark
import asyncio

def print_header():
    """Print welcome header"""
    print("=" * 60)
    print("  CohortRAG Engine - Educational RAG System")
    print("  Project ID: CHAI-2025-001")
    print("=" * 60)

def print_menu():
    """Print main menu"""
    print("\nDEVELOPMENT OPTIONS:")
    print("1. Run document ingestion")
    print("2. Query the knowledge base")
    print("3. Show system statistics")
    print("4. Run interactive Q&A session")
    print("5. Test with sample questions")
    print("6. Run evaluation framework (RAGAS)")
    print("7. Run unit tests")
    print("\nPRODUCTION READINESS:")
    print("8. Async document ingestion")
    print("9. Production query with caching")
    print("10. Cache management & cost monitoring")
    print("11. Performance benchmarking")
    print("12. Cost optimization analysis")
    print("13. Success metrics validation")
    print("14. Production readiness check")
    print("0. Exit")
    print("-" * 40)

def run_ingestion():
    """Run the document ingestion pipeline"""
    print("\n[BOOKS] Starting Document Ingestion")
    print("-" * 40)

    try:
        pipeline = CohortRAGIngestionPipeline()

        # Show current stats
        stats = pipeline.get_stats()
        print(f"Current knowledge base: {stats['total_chunks']} chunks from {stats['total_files']} files")
        print(f"Data directory: {stats['config']['data_dir']}")
        print(f"Supported formats: {', '.join(stats['supported_formats'])}")

        # Check if data directory has files
        data_path = Path(stats['config']['data_dir'])
        if not data_path.exists():
            print(f"\n[ERROR] Data directory not found: {data_path}")
            print("Please create the data directory and add some documents.")
            return

        files_in_dir = list(data_path.glob("*.*"))
        if not files_in_dir:
            print(f"\n[WARNING] No files found in {data_path}")
            print("Please add some documents to the data directory.")
            return

        print(f"\nFiles in data directory:")
        for file_path in files_in_dir:
            print(f"  - {file_path.name} ({file_path.stat().st_size} bytes)")

        # Run ingestion
        print(f"\n[PROCESSING] Processing documents...")
        success = pipeline.ingest_directory()

        if success:
            # Show updated stats
            final_stats = pipeline.get_stats()
            print(f"\n[SUCCESS] Ingestion complete!")
            print(f"Final stats: {final_stats['total_chunks']} chunks from {final_stats['total_files']} files")
        else:
            print("\n[ERROR] Ingestion failed!")

    except Exception as e:
        print(f"\n[ERROR] Error during ingestion: {e}")

def query_knowledge_base():
    """Query the knowledge base with enhanced two-phase retrieval"""
    print("\n[QUERY] Enhanced Knowledge Base Query")
    print("-" * 40)

    try:
        retriever = CohortRAGRetriever()

        # Check if knowledge base is loaded
        stats = retriever.get_stats()
        if not stats["vector_store_loaded"]:
            print("[ERROR] No knowledge base loaded. Please run ingestion first.")
            return

        print(f"Knowledge base loaded: {stats['total_chunks']} chunks available")
        print(f"Enhanced features: Reranking={stats['reranking_enabled']}, Query Expansion={stats['query_expansion_enabled']}")
        if stats.get('reranker_model'):
            print(f"Reranker: {stats['reranker_model']} (Available: {stats['reranker_available']})")

        # Get user question
        question = input("\nEnter your question: ").strip()
        if not question:
            print("No question entered.")
            return

        print(f"\n[SEARCHING] Enhanced two-phase retrieval for: '{question}'")

        # Query the system with enhanced retrieval
        response = retriever.query(question)

        # Display results
        print(f"\n[ANSWER] Answer:")
        print("-" * 20)
        print(response.answer)

        print(f"\n[METADATA] Enhanced Metadata:")
        print(f"  - Processing time: {response.processing_time:.2f}s")
        print(f"  - Sources used: {len(response.sources)}")
        if response.confidence_score is not None:
            print(f"  - Confidence score: {response.confidence_score:.3f}")

        # Show query expansions if any
        if response.expanded_queries:
            print(f"\n[QUERY EXPANSION] Original query expanded to:")
            for i, expanded in enumerate(response.expanded_queries[:3], 1):
                print(f"  {i}. {expanded}")

        if response.sources:
            print(f"\n[SOURCES] Top sources (Two-phase retrieval):")
            for i, source in enumerate(response.sources[:3]):  # Show top 3 sources
                print(f"  {i+1}. Similarity: {source['similarity']:.3f}")
                if source.get('rerank_score') is not None:
                    print(f"     Rerank score: {source['rerank_score']:.3f}")
                print(f"     Text: {source['text']}")
                if 'file_name' in source['metadata']:
                    print(f"     File: {source['metadata']['file_name']}")
                print()

    except Exception as e:
        print(f"\n[ERROR] Error during query: {e}")

def show_system_stats():
    """Show system statistics"""
    print("\n[STATS] System Statistics")
    print("-" * 40)

    try:
        config = get_config()
        print(f"Configuration:")
        print(f"  - Data directory: {config.data_dir}")
        print(f"  - Vector DB path: {config.chroma_db_path}")
        print(f"  - Embedding model: {config.embedding_model}")
        print(f"  - LLM model: {config.llm_model}")
        print(f"  - Chunk size: {config.chunk_size}")
        print(f"  - Similarity top-k: {config.similarity_top_k}")

        # Try to load retriever stats
        try:
            retriever = CohortRAGRetriever()
            stats = retriever.get_stats()
            print(f"\nKnowledge Base:")
            print(f"  - Total chunks: {stats['total_chunks']}")
            print(f"  - Vector store loaded: {stats['vector_store_loaded']}")

            print(f"\nEnhanced Features:")
            print(f"  - Reranking enabled: {stats['reranking_enabled']}")
            print(f"  - Query expansion enabled: {stats['query_expansion_enabled']}")
            if stats.get('reranker_model'):
                print(f"  - Reranker model: {stats['reranker_model']}")
                print(f"  - Reranker available: {stats['reranker_available']}")
            else:
                print(f"  - Reranker: Using similarity fallback")

        except Exception as e:
            print(f"\nKnowledge Base: Error loading - {e}")

        # Check data directory
        data_path = Path(config.data_dir)
        if data_path.exists():
            files = list(data_path.glob("*.*"))
            print(f"\nData Directory:")
            print(f"  - Path: {data_path}")
            print(f"  - Files: {len(files)}")
            for file_path in files[:5]:  # Show first 5 files
                print(f"    - {file_path.name}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")
        else:
            print(f"\nData Directory: {data_path} (not found)")

    except Exception as e:
        print(f"[ERROR] Error getting stats: {e}")

def interactive_qa():
    """Run interactive Q&A session"""
    print("\n[INTERACTIVE] Interactive Q&A Session")
    print("-" * 40)
    print("Type 'exit' or 'quit' to return to main menu")

    try:
        retriever = CohortRAGRetriever()

        # Check if knowledge base is loaded
        stats = retriever.get_stats()
        if not stats["vector_store_loaded"]:
            print("[ERROR] No knowledge base loaded. Please run ingestion first.")
            return

        print(f"[SUCCESS] Knowledge base ready ({stats['total_chunks']} chunks)")
        print(f"[FEATURES] Reranking: {stats['reranking_enabled']} | Query Expansion: {stats['query_expansion_enabled']}")

        while True:
            question = input("\n[QUESTION] Your question: ").strip()

            if question.lower() in ['exit', 'quit', '']:
                break

            print("[SEARCHING] Enhanced two-phase retrieval...")
            response = retriever.query(question)

            print(f"\n[ANSWER] Answer: {response.answer}")

            # Enhanced information display
            info_parts = [f"Time: {response.processing_time:.2f}s", f"Sources: {len(response.sources)}"]
            if response.confidence_score is not None:
                info_parts.append(f"Confidence: {response.confidence_score:.3f}")
            print(f"[INFO] {' | '.join(info_parts)}")

            # Show query expansion if occurred
            if response.expanded_queries:
                print(f"[EXPANSION] Query expanded to: {', '.join(response.expanded_queries[:2])}")

            # Show ranking scores if available
            if response.sources and len(response.sources) > 0:
                source_info = f"Top source - Similarity: {response.sources[0]['similarity']:.3f}"
                if response.sources[0].get('rerank_score') is not None:
                    source_info += f" | Rerank: {response.sources[0]['rerank_score']:.3f}"
                print(f"[RANKING] {source_info}")

        print("Returning to main menu...")

    except Exception as e:
        print(f"[ERROR] Error: {e}")

def test_sample_questions():
    """Test with predefined sample questions"""
    print("\n[TEST] Testing with Sample Questions")
    print("-" * 40)

    sample_questions = [
        "What is RAG?",
        "How does document indexing work?",
        "What are the best practices for RAG implementation?",
        "How can RAG be used in education?",
        "What are the core components of RAG?"
    ]

    try:
        retriever = CohortRAGRetriever()

        # Check if knowledge base is loaded
        stats = retriever.get_stats()
        if not stats["vector_store_loaded"]:
            print("[ERROR] No knowledge base loaded. Please run ingestion first.")
            return

        print(f"Testing {len(sample_questions)} questions with enhanced retrieval...\n")

        for i, question in enumerate(sample_questions, 1):
            print(f"{i}. Question: {question}")

            response = retriever.query(question)

            print(f"   Answer: {response.answer[:100]}...")

            # Enhanced metrics display
            metrics = [f"Time: {response.processing_time:.2f}s", f"Sources: {len(response.sources)}"]
            if response.confidence_score is not None:
                metrics.append(f"Confidence: {response.confidence_score:.3f}")
            print(f"   {' | '.join(metrics)}")

            if response.sources:
                source_info = f"Similarity: {response.sources[0]['similarity']:.3f}"
                if response.sources[0].get('rerank_score') is not None:
                    source_info += f" | Rerank: {response.sources[0]['rerank_score']:.3f}"
                print(f"   Top source - {source_info}")

            if response.expanded_queries:
                print(f"   Expansion: {response.expanded_queries[0][:50]}...")
            print()

    except Exception as e:
        print(f"[ERROR] Error during testing: {e}")

def run_evaluation():
    """Run RAGAS evaluation framework"""
    print("\n[EVALUATION] RAGAS Evaluation Framework")
    print("-" * 40)

    try:
        # Initialize retriever
        retriever = CohortRAGRetriever()

        # Check if knowledge base is loaded
        stats = retriever.get_stats()
        if not stats.get("vector_store_loaded", False):
            print("[ERROR] No knowledge base loaded. Please run ingestion first.")
            print("   Use option 1 to run document ingestion.")
            return

        print(f"[SUCCESS] Knowledge base ready ({stats['total_chunks']} chunks)")

        # Initialize evaluator
        evaluator = RAGASEvaluator(retriever)

        # Get evaluation options
        print("\nEvaluation Options:")
        print("1. Quick evaluation (5 synthetic Q&A pairs)")
        print("2. Standard evaluation (15 synthetic Q&A pairs)")
        print("3. Comprehensive evaluation (30 synthetic Q&A pairs)")
        print("4. Use ground truth dataset (if available)")

        choice = input("\nSelect evaluation type (1-4): ").strip()

        if choice == "1":
            num_questions = 5
            use_synthetic = True
        elif choice == "2":
            num_questions = 15
            use_synthetic = True
        elif choice == "3":
            num_questions = 30
            use_synthetic = True
        elif choice == "4":
            num_questions = 0
            use_synthetic = False
        else:
            print("[ERROR] Invalid choice. Using quick evaluation.")
            num_questions = 5
            use_synthetic = True

        print(f"\n[PROCESSING] Running {'synthetic' if use_synthetic else 'ground truth'} evaluation...")

        # Run evaluation
        results = evaluator.evaluate_system_comprehensive(
            use_synthetic=use_synthetic,
            num_synthetic=num_questions
        )

        # Display results
        print("\n📊 EVALUATION RESULTS")
        print("=" * 50)

        # Dataset info
        dataset_info = results.get("dataset_info", {})
        print(f"Dataset: {dataset_info.get('type', 'unknown')} ({dataset_info.get('num_samples', 0)} samples)")

        # RAGAS scores
        if results.get("ragas_scores"):
            print("\n🎯 RAGAS Metrics:")
            for metric, score in results["ragas_scores"].items():
                status = "✅" if score > 0.7 else "⚠️" if score > 0.5 else "❌"
                print(f"   {status} {metric}: {score:.3f}")

        # Custom educational metrics
        if results.get("custom_metrics"):
            print("\n🎓 Educational Metrics:")
            for metric, score in results["custom_metrics"].items():
                status = "✅" if score > 0.7 else "⚠️" if score > 0.5 else "❌"
                print(f"   {status} {metric.replace('_', ' ').title()}: {score:.3f}")

        # Performance metrics
        if results.get("performance_metrics"):
            print("\n⚡ Performance Metrics:")
            perf = results["performance_metrics"]
            avg_time = perf.get("avg_processing_time", 0)
            success_rate = perf.get("query_success_rate", 0)
            print(f"   📈 Avg Processing Time: {avg_time:.3f}s")
            print(f"   📈 Query Success Rate: {success_rate:.3f}")
            print(f"   📈 Avg Sources Used: {perf.get('avg_sources_used', 0):.1f}")

        # Overall assessment
        ragas_avg = sum(results.get("ragas_scores", {}).values()) / max(1, len(results.get("ragas_scores", {})))
        custom_avg = sum(results.get("custom_metrics", {}).values()) / max(1, len(results.get("custom_metrics", {})))

        overall_score = (ragas_avg + custom_avg) / 2
        if overall_score > 0.8:
            print(f"\n🏆 Overall Assessment: EXCELLENT ({overall_score:.3f})")
        elif overall_score > 0.6:
            print(f"\n👍 Overall Assessment: GOOD ({overall_score:.3f})")
        elif overall_score > 0.4:
            print(f"\n⚠️ Overall Assessment: NEEDS IMPROVEMENT ({overall_score:.3f})")
        else:
            print(f"\n❌ Overall Assessment: POOR ({overall_score:.3f})")

        print(f"\n📁 Detailed results saved to evaluation_data/results/")

    except Exception as e:
        print(f"[ERROR] Error during evaluation: {e}")

def run_tests():
    """Run unit tests"""
    print("\n[TESTS] Unit Test Suite")
    print("-" * 40)

    try:
        import subprocess
        import os

        # Change to tests directory and run tests
        test_script = os.path.join("tests", "run_tests.py")

        if os.path.exists(test_script):
            print("Running comprehensive test suite...")
            result = subprocess.run(["python", test_script], capture_output=True, text=True)

            print(result.stdout)
            if result.stderr:
                print(f"Errors:\n{result.stderr}")

            if result.returncode == 0:
                print("✅ All tests completed successfully!")
            else:
                print("❌ Some tests failed. Check output above.")
        else:
            print("❌ Test script not found. Please ensure tests/run_tests.py exists.")

    except Exception as e:
        print(f"[ERROR] Error running tests: {e}")

def run_async_ingestion():
    """Run asynchronous document ingestion for production"""
    print("\n[ASYNC] Asynchronous Document Ingestion")
    print("-" * 40)

    async def async_ingestion_main():
        try:
            # Configuration options
            print("Configuration Options:")
            print("1. Fast processing (max_workers=2, batch_size=20)")
            print("2. Balanced processing (max_workers=4, batch_size=50)")
            print("3. Heavy processing (max_workers=8, batch_size=100)")

            choice = input("Select configuration (1-3): ").strip()

            if choice == "1":
                max_workers, batch_size = 2, 20
            elif choice == "2":
                max_workers, batch_size = 4, 50
            elif choice == "3":
                max_workers, batch_size = 8, 100
            else:
                print("Using default configuration (balanced)")
                max_workers, batch_size = 4, 50

            # Initialize async ingestion
            async_ingestion = AsyncCohortRAGIngestion(
                max_workers=max_workers,
                batch_size=batch_size
            )

            config = get_config()
            data_dir = config.data_dir

            if not os.path.exists(data_dir):
                print(f"❌ Data directory not found: {data_dir}")
                return

            # Progress callback
            def progress_callback(current: int, total: int):
                progress = (current / total) * 100
                print(f"⏳ Progress: {current}/{total} ({progress:.1f}%)")

            print(f"\n🚀 Starting async ingestion (workers: {max_workers}, batch: {batch_size})")

            # Run async ingestion
            results = await async_ingestion.ingest_documents_async(
                data_dir=data_dir,
                progress_callback=progress_callback
            )

            # Display results
            print("\n📊 Async Ingestion Results:")
            print(f"  ✅ Success: {results['success']}")
            print(f"  📄 Documents: {results.get('total_documents', 0)}")
            print(f"  📝 Chunks: {results.get('total_chunks', 0)}")
            print(f"  ⏱️  Time: {results.get('processing_time', 0):.2f}s")
            print(f"  📈 Avg time per doc: {results.get('avg_time_per_doc', 0):.2f}s")
            print(f"  🧠 Memory usage: {results.get('memory_usage', {}).get('peak_mb', 0):.1f}MB peak")

            if results.get('failed_documents'):
                print(f"  ⚠️  Failed: {len(results['failed_documents'])} documents")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Run async function
    asyncio.run(async_ingestion_main())

def run_production_query():
    """Query with production caching and cost tracking"""
    print("\n[PRODUCTION] Production Query with Caching")
    print("-" * 40)

    try:
        # Cache configuration
        print("Cache Configuration:")
        print("1. Memory cache (fast, limited)")
        print("2. Redis cache (scalable, requires Redis)")
        print("3. No cache (direct queries)")

        cache_choice = input("Select cache type (1-3): ").strip()

        if cache_choice == "1":
            cache_type = "memory"
            enable_caching = True
        elif cache_choice == "2":
            cache_type = "redis"
            enable_caching = True
            redis_url = input("Redis URL (default: redis://localhost:6379): ").strip()
            if not redis_url:
                redis_url = "redis://localhost:6379"
        else:
            cache_type = "memory"
            enable_caching = False

        # Initialize production retriever
        if cache_choice == "2":
            retriever = ProductionCohortRAGRetriever(
                enable_caching=enable_caching,
                cache_type=cache_type,
                redis_url=redis_url,
                cache_ttl=1800  # 30 minutes
            )
        else:
            retriever = ProductionCohortRAGRetriever(
                enable_caching=enable_caching,
                cache_type=cache_type,
                cache_ttl=1800
            )

        # Check knowledge base
        stats = retriever.get_stats()
        if not stats.get("vector_store_loaded", False):
            print("❌ No knowledge base loaded. Please run ingestion first.")
            return

        print(f"✅ Knowledge base ready ({stats['total_chunks']} chunks)")
        print(f"🔧 Cache: {'Enabled' if enable_caching else 'Disabled'} ({cache_type})")

        # Interactive querying
        print("\nInteractive Production Query (type 'exit' to quit)")
        while True:
            question = input("\n🔍 Your question: ").strip()

            if question.lower() in ['exit', 'quit', '']:
                break

            # Query with timing
            response = retriever.query(question)

            print(f"\n📝 Answer: {response.answer}")
            print(f"⏱️  Time: {response.processing_time:.3f}s")
            print(f"💾 Cached: {'Yes' if response.cached else 'No'}")
            print(f"💰 Cost: ${response.cost_info.get('cost', 0):.6f}")
            print(f"🔢 Tokens: {response.cost_info.get('tokens_used', 0)}")

        # Show cache statistics
        if enable_caching:
            cache_stats = retriever.get_cache_stats()
            query_stats = cache_stats.get('query_statistics', {})
            print(f"\n📊 Cache Performance:")
            print(f"   Hit rate: {query_stats.get('hit_rate', 0):.3f}")
            print(f"   Total queries: {query_stats.get('total_queries', 0)}")
            print(f"   Cache hits: {query_stats.get('cache_hits', 0)}")

    except Exception as e:
        print(f"❌ Error: {e}")

def cache_management_and_monitoring():
    """Cache management and cost monitoring interface"""
    print("\n[CACHE] Cache Management & Cost Monitoring")
    print("-" * 40)

    try:
        # Initialize production retriever with caching
        retriever = ProductionCohortRAGRetriever(
            enable_caching=True,
            cache_type="memory",
            cache_ttl=1800
        )

        while True:
            print("\nCache Management Options:")
            print("1. View cache statistics")
            print("2. Clear cache")
            print("3. Cache optimization recommendations")
            print("4. Cost analysis")
            print("5. Cache warmup")
            print("0. Return to main menu")

            choice = input("Select option (0-5): ").strip()

            if choice == "0":
                break
            elif choice == "1":
                stats = retriever.get_cache_stats()
                print("\n📊 Cache Statistics:")
                print(f"   Cache enabled: {stats['cache_enabled']}")

                query_stats = stats.get('query_statistics', {})
                print(f"   Total queries: {query_stats.get('total_queries', 0)}")
                print(f"   Cache hits: {query_stats.get('cache_hits', 0)}")
                print(f"   Hit rate: {query_stats.get('hit_rate', 0):.3f}")

                cost_stats = stats.get('cost_statistics', {})
                print(f"   Total cost: ${cost_stats.get('total_cost', 0):.6f}")
                print(f"   Avg cost per query: ${cost_stats.get('avg_cost_per_query', 0):.6f}")

            elif choice == "2":
                result = retriever.clear_cache()
                print(f"🗑️  {result['message']}")

            elif choice == "3":
                optimization = retriever.optimize_cache_settings()
                print("\n🎯 Optimization Recommendations:")
                for rec in optimization.get('recommendations', []):
                    print(f"   • {rec}")

                actions = optimization.get('optimization_actions', [])
                if actions:
                    print("\n💡 Suggested Actions:")
                    for action in actions:
                        print(f"   • {action}")

            elif choice == "4":
                # Initialize cost tracker
                cost_tracker = GeminiCostTracker()

                print("\n💰 Cost Analysis:")
                print("   [Simulated cost tracking - integrate with actual usage]")
                print(f"   Current pricing: {cost_tracker.get_current_pricing()}")

                # Sample projection
                daily_queries = input("   Estimated daily queries: ").strip()
                if daily_queries.isdigit():
                    projection = cost_tracker.project_monthly_cost(
                        daily_queries=int(daily_queries),
                        avg_tokens_per_query=500
                    )
                    print(f"   Monthly projection: ${projection:.2f}")

            elif choice == "5":
                common_queries = [
                    "What is RAG?",
                    "How does document indexing work?",
                    "What are the best practices?",
                    "How can RAG be used in education?"
                ]

                print("🔥 Cache warmup with common queries...")
                warmup_result = retriever.warmup_cache(common_queries)
                print(f"   Success: {warmup_result.get('successful_warmups', 0)}")
                print(f"   Time: {warmup_result.get('warmup_time', 0):.2f}s")

    except Exception as e:
        print(f"❌ Error: {e}")

def run_performance_benchmarking():
    """Run comprehensive performance benchmarking"""
    print("\n[BENCHMARK] Performance Benchmarking Suite")
    print("-" * 40)

    try:
        # Check knowledge base
        retriever = CohortRAGRetriever()
        stats = retriever.get_stats()
        if not stats.get("vector_store_loaded", False):
            print("❌ No knowledge base loaded. Please run ingestion first.")
            return

        print("Benchmark Configuration:")
        print("1. Quick benchmark (5 queries)")
        print("2. Standard benchmark (20 queries)")
        print("3. Comprehensive benchmark (50 queries)")
        print("4. Load test (100 concurrent queries)")

        choice = input("Select benchmark type (1-4): ").strip()

        if choice == "1":
            num_queries = 5
        elif choice == "2":
            num_queries = 20
        elif choice == "3":
            num_queries = 50
        elif choice == "4":
            num_queries = 100
        else:
            print("Using standard benchmark")
            num_queries = 20

        print(f"\n🚀 Running benchmark with {num_queries} queries...")

        # Initialize benchmark suite
        benchmark = ComprehensiveBenchmark(retriever)

        # Run benchmark
        results = benchmark.run_comprehensive_benchmark(num_queries=num_queries)

        # Display results
        print("\n📊 Benchmark Results:")
        print("=" * 50)

        # Performance metrics
        perf = results.get('performance_metrics', {})
        print(f"🚀 Average Latency: {perf.get('avg_latency', 0):.3f}s")
        print(f"📈 Throughput: {perf.get('throughput', 0):.1f} queries/second")
        print(f"💾 Memory Usage: {perf.get('memory_usage_mb', 0):.1f}MB")
        print(f"🔥 CPU Usage: {perf.get('cpu_usage_percent', 0):.1f}%")

        # Quality metrics
        quality = results.get('quality_metrics', {})
        if quality:
            print(f"\n🎯 Quality Metrics:")
            print(f"   Response Quality: {quality.get('response_quality', 0):.3f}")
            print(f"   Source Relevance: {quality.get('source_relevance', 0):.3f}")

        # System recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Performance Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")

    except Exception as e:
        print(f"❌ Error: {e}")

def run_cost_optimization():
    """Run cost optimization analysis"""
    print("\n[OPTIMIZATION] Cost Optimization Analysis")
    print("-" * 40)

    try:
        # Initialize cost components
        cost_tracker = GeminiCostTracker()
        optimizer = CostOptimizer()

        # Current system analysis
        print("🔍 Analyzing current system...")

        # Simulate current usage
        daily_queries = input("Enter average daily queries (default: 100): ").strip()
        daily_queries = int(daily_queries) if daily_queries.isdigit() else 100

        avg_tokens = input("Enter average tokens per query (default: 500): ").strip()
        avg_tokens = int(avg_tokens) if avg_tokens.isdigit() else 500

        # Cost projections
        monthly_cost = cost_tracker.project_monthly_cost(daily_queries, avg_tokens)
        yearly_cost = monthly_cost * 12

        print(f"\n💰 Current Cost Analysis:")
        print(f"   Daily queries: {daily_queries}")
        print(f"   Avg tokens per query: {avg_tokens}")
        print(f"   Monthly cost: ${monthly_cost:.2f}")
        print(f"   Yearly cost: ${yearly_cost:.2f}")

        # Optimization strategies
        print(f"\n🎯 Optimization Strategies:")

        # Caching impact
        cache_hit_rates = [0.2, 0.5, 0.8]
        for rate in cache_hit_rates:
            cached_cost = monthly_cost * (1 - rate)
            savings = monthly_cost - cached_cost
            print(f"   {rate*100:.0f}% cache hit rate: ${cached_cost:.2f}/month (Save: ${savings:.2f})")

        # Token optimization
        optimized_strategies = optimizer.suggest_optimizations(
            current_monthly_cost=monthly_cost,
            avg_tokens_per_query=avg_tokens,
            query_volume=daily_queries * 30
        )

        print(f"\n💡 Optimization Suggestions:")
        for strategy in optimized_strategies.get('strategies', []):
            print(f"   • {strategy}")

        # Cost thresholds
        threshold = input(f"\nSet monthly cost alert threshold ($): ").strip()
        if threshold and threshold.replace('.', '').isdigit():
            threshold = float(threshold)
            if monthly_cost > threshold:
                print(f"⚠️  Warning: Current projection (${monthly_cost:.2f}) exceeds threshold (${threshold:.2f})")
            else:
                print(f"✅ Current projection (${monthly_cost:.2f}) is within threshold (${threshold:.2f})")

    except Exception as e:
        print(f"❌ Error: {e}")

def run_success_metrics_validation():
    """Run comprehensive success metrics validation"""
    print("\n[VALIDATION] Success Metrics Validation")
    print("-" * 40)

    try:
        # Check knowledge base
        retriever = CohortRAGRetriever()
        stats = retriever.get_stats()
        if not stats.get("vector_store_loaded", False):
            print("❌ No knowledge base loaded. Please run ingestion first.")
            return

        # Initialize evaluator
        evaluator = RAGASEvaluator(retriever)

        print("🎯 Target Success Metrics:")
        print("   • Context Recall: ≥85% (RAGAS)")
        print("   • Answer Faithfulness: ≥90% (RAGAS)")
        print("   • Answer Relevancy: ≥90% (RAGAS)")
        print("   • Latency: <2 seconds (Benchmarking)")
        print("   • Cost Efficiency: <$0.05 per query (Cost Modeling)")

        # Configuration options
        print("\nValidation Configuration:")
        print("1. Quick validation (10 synthetic questions)")
        print("2. Standard validation (20 synthetic questions)")
        print("3. Comprehensive validation (50 synthetic questions)")
        print("4. Custom test questions")

        choice = input("Select validation type (1-4): ").strip()

        if choice == "1":
            num_questions = 10
            test_questions = None
        elif choice == "2":
            num_questions = 20
            test_questions = None
        elif choice == "3":
            num_questions = 50
            test_questions = None
        elif choice == "4":
            print("Enter test questions (one per line, empty line to finish):")
            test_questions = []
            while True:
                question = input("Question: ").strip()
                if not question:
                    break
                test_questions.append(question)
            num_questions = len(test_questions) if test_questions else 10
        else:
            print("Using standard validation")
            num_questions = 20
            test_questions = None

        print(f"\n🚀 Running success metrics validation...")
        print(f"   Test questions: {num_questions}")
        print(f"   Mode: {'Custom' if test_questions else 'Synthetic'}")

        # Run validation
        report = evaluator.validate_success_metrics(
            test_questions=test_questions,
            num_synthetic=num_questions
        )

        # Export report option
        export = input(f"\nExport detailed report to JSON? (y/n): ").strip().lower()
        if export == 'y':
            from utils.success_metrics import SuccessMetricsValidator
            validator = SuccessMetricsValidator()
            validator.export_report_json(report, "success_validation_report.json")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def run_production_readiness_check():
    """Run quick production readiness assessment"""
    print("\n[READINESS] Production Readiness Check")
    print("-" * 40)

    try:
        # Check knowledge base
        retriever = CohortRAGRetriever()
        stats = retriever.get_stats()
        if not stats.get("vector_store_loaded", False):
            print("❌ No knowledge base loaded. Please run ingestion first.")
            return

        # Initialize evaluator
        evaluator = RAGASEvaluator(retriever)

        print("🚀 Checking production readiness against success criteria...")
        print("   This performs a quick validation with 10 synthetic questions")

        # Configuration
        min_pass_rate = input("Minimum pass rate for production (default 80%): ").strip()
        if min_pass_rate.endswith('%'):
            min_pass_rate = min_pass_rate[:-1]

        try:
            min_pass_rate = float(min_pass_rate) / 100 if min_pass_rate else 0.8
        except ValueError:
            min_pass_rate = 0.8

        print(f"   Using {min_pass_rate:.0%} pass rate threshold")

        # Run production readiness check
        assessment = evaluator.check_production_readiness(min_pass_rate)

        # Additional recommendations based on assessment
        if assessment["production_ready"]:
            print(f"\n🎉 Congratulations! System is ready for production deployment.")
            print(f"\n📋 Pre-Deployment Checklist:")
            print(f"   ✅ All success metrics validated")
            print(f"   ✅ Performance requirements met")
            print(f"   ⏭️  Next: Deploy to production environment")
            print(f"   ⏭️  Next: Setup monitoring and alerting")
            print(f"   ⏭️  Next: Configure auto-scaling (if needed)")
        else:
            print(f"\n⚠️  System requires optimization before production deployment.")
            print(f"\n📋 Remediation Checklist:")

            if assessment["critical_failures"] > 0:
                print(f"   ❌ Fix {assessment['critical_failures']} critical failing metrics")
            if assessment["warnings"] > 0:
                print(f"   ⚠️  Address {assessment['warnings']} metrics with warnings")
            if assessment["pass_rate"] < min_pass_rate:
                print(f"   📈 Improve overall performance from {assessment['pass_rate']:.1%} to {min_pass_rate:.1%}")

            print(f"\n🔄 Suggested Actions:")
            print(f"   • Run 'Success metrics validation' for detailed analysis")
            print(f"   • Use 'Performance benchmarking' to identify bottlenecks")
            print(f"   • Enable caching to improve performance and reduce costs")
            print(f"   • Optimize chunking strategy for better context recall")

        # Save assessment
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        assessment_file = f"production_readiness_{timestamp}.json"

        save = input(f"\nSave readiness assessment? (y/n): ").strip().lower()
        if save == 'y':
            import json
            with open(assessment_file, 'w') as f:
                json.dump(assessment, f, indent=2)
            print(f"📄 Assessment saved to: {assessment_file}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main application loop"""
    print_header()

    # Show initial system info
    try:
        config = get_config()
        print(f"[SUCCESS] Configuration loaded")
        print(f"[DATA] Data directory: {config.data_dir}")
        print(f"[API] API configured: {bool(config.gemini_api_key)}")
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
        return

    while True:
        print_menu()

        try:
            choice = input("Enter your choice (0-14): ").strip()

            if choice == '0':
                print("\n[EXIT] Goodbye!")
                break
            elif choice == '1':
                run_ingestion()
            elif choice == '2':
                query_knowledge_base()
            elif choice == '3':
                show_system_stats()
            elif choice == '4':
                interactive_qa()
            elif choice == '5':
                test_sample_questions()
            elif choice == '6':
                run_evaluation()
            elif choice == '7':
                run_tests()
            elif choice == '8':
                run_async_ingestion()
            elif choice == '9':
                run_production_query()
            elif choice == '10':
                cache_management_and_monitoring()
            elif choice == '11':
                run_performance_benchmarking()
            elif choice == '12':
                run_cost_optimization()
            elif choice == '13':
                run_success_metrics_validation()
            elif choice == '14':
                run_production_readiness_check()
            else:
                print("[ERROR] Invalid choice. Please enter 0-14.")

        except KeyboardInterrupt:
            print("\n\n[EXIT] Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    main()