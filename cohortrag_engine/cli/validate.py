#!/usr/bin/env python3
"""
CohortRAG Engine - Validation CLI
Command-line interface for success metrics validation
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.evaluation import RAGASEvaluator
from core.retrieval import CohortRAGRetriever
from utils.success_metrics import SuccessMetricsValidator
from config import get_config


def main():
    """Main validation CLI function"""
    parser = argparse.ArgumentParser(
        description="CohortRAG Engine Success Metrics Validation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cohortrag-validate --quick                     # Quick validation (10 questions)
  cohortrag-validate --comprehensive             # Full validation (50 questions)
  cohortrag-validate --readiness                 # Production readiness check
  cohortrag-validate --output validation.json    # Save results to file
  cohortrag-validate --questions file.txt        # Use custom test questions
        """
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (10 synthetic questions)"
    )

    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive validation (50 synthetic questions)"
    )

    parser.add_argument(
        "--readiness",
        action="store_true",
        help="Run production readiness check only"
    )

    parser.add_argument(
        "--questions",
        type=str,
        help="Path to file with custom test questions (one per line)"
    )

    parser.add_argument(
        "--num-questions",
        type=int,
        default=20,
        help="Number of synthetic questions (default: 20)"
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

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Pass rate threshold for production approval (default: 0.8)"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        if args.config:
            config = get_config(args.config)
        else:
            config = get_config()

        print("🎯 CohortRAG Engine Success Metrics Validation")
        print("=" * 55)
        print(f"Configuration: {config}")

        # Load custom questions if provided
        test_questions = None
        if args.questions:
            with open(args.questions, 'r') as f:
                test_questions = [line.strip() for line in f if line.strip()]
            print(f"Custom questions loaded: {len(test_questions)}")

        # Determine number of questions
        if args.quick:
            num_questions = 10
        elif args.comprehensive:
            num_questions = 50
        else:
            num_questions = args.num_questions

        print(f"Test mode: {'Custom' if test_questions else 'Synthetic'}")
        print(f"Test questions: {len(test_questions) if test_questions else num_questions}")
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

        # Initialize evaluator
        evaluator = RAGASEvaluator(retriever, config)

        if args.readiness:
            # Production readiness check only
            print("🚀 Running production readiness check...")
            assessment = evaluator.check_production_readiness(args.threshold)

            print(f"\n📊 PRODUCTION READINESS ASSESSMENT")
            print(f"Status: {'✅' if assessment['production_ready'] else '❌'} {assessment['assessment']}")
            print(f"Pass Rate: {assessment['pass_rate']:.1%} (Threshold: {args.threshold:.0%})")
            print(f"Failed Metrics: {assessment['critical_failures']}")
            print(f"Warning Metrics: {assessment['warnings']}")

            print(f"\n🔄 Next Steps:")
            for step in assessment["next_steps"]:
                print(f"   • {step}")

            # Save results if requested
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    json.dump(assessment, f, indent=2)
                print(f"\n📄 Results saved to: {args.output}")

        else:
            # Full success metrics validation
            print("🎯 Running success metrics validation...")
            print("Target Metrics:")
            print("   • Context Recall: ≥85% (RAGAS)")
            print("   • Answer Faithfulness: ≥90% (RAGAS)")
            print("   • Answer Relevancy: ≥90% (RAGAS)")
            print("   • Latency: <2 seconds (Benchmarking)")
            print("   • Cost Efficiency: <$0.05 per query (Cost Modeling)")
            print()

            # Run validation
            report = evaluator.validate_success_metrics(
                test_questions=test_questions,
                num_synthetic=num_questions
            )

            # Export report if requested
            if args.output:
                validator = SuccessMetricsValidator()
                validator.export_report_json(report, args.output)

        print(f"\n✅ Validation completed successfully!")

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()