#!/usr/bin/env python3
"""
CohortRAG Engine - Benchmark CLI
Command-line interface for running performance benchmarks
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.benchmarks import ComprehensiveBenchmark
from core.retrieval import CohortRAGRetriever
from config import get_config


def main():
    """Main benchmark CLI function"""
    parser = argparse.ArgumentParser(
        description="CohortRAG Engine Performance Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cohortrag-benchmark --quick                    # Quick 5-query benchmark
  cohortrag-benchmark --queries 50               # Custom query count
  cohortrag-benchmark --output results.json      # Save to file
  cohortrag-benchmark --config /path/to/.env     # Custom config
        """
    )

    parser.add_argument(
        "--queries",
        type=int,
        default=20,
        help="Number of test queries to run (default: 20)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (5 queries)"
    )

    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive benchmark (100 queries)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON format)"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (.env)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Determine number of queries
    if args.quick:
        num_queries = 5
    elif args.comprehensive:
        num_queries = 100
    else:
        num_queries = args.queries

    try:
        # Load configuration
        if args.config:
            config = get_config(args.config)
        else:
            config = get_config()

        print("🚀 CohortRAG Engine Performance Benchmark")
        print("=" * 50)
        print(f"Configuration: {config}")
        print(f"Test queries: {num_queries}")
        print()

        # Initialize retriever
        print("Initializing RAG system...")
        retriever = CohortRAGRetriever(config)

        # Check if knowledge base is loaded
        stats = retriever.get_stats()
        if not stats.get("vector_store_loaded", False):
            print("❌ No knowledge base loaded. Please run ingestion first:")
            print("   cohortrag ingest /path/to/your/documents")
            sys.exit(1)

        print(f"✅ Knowledge base ready ({stats['total_chunks']} chunks)")
        print()

        # Initialize benchmark
        print("Starting benchmark...")
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

        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Performance Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")

        # Save to file if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n📄 Results saved to: {args.output}")

        print(f"\n✅ Benchmark completed successfully!")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()