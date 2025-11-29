"""
Unit tests for evaluation framework
"""

import unittest
import tempfile
import os
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.evaluation import (
    EvaluationDataset,
    RAGASEvaluator
)

class TestEvaluationDataset(unittest.TestCase):
    """Test EvaluationDataset functionality"""

    def setUp(self):
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.dataset = EvaluationDataset(self.test_dir)

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_dataset_initialization(self):
        """Test dataset initialization"""
        self.assertEqual(self.dataset.dataset_path, self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))

    def test_create_ground_truth_template(self):
        """Test ground truth template creation"""
        self.dataset.create_ground_truth_template()

        template_file = os.path.join(self.test_dir, "ground_truth_template.json")
        self.assertTrue(os.path.exists(template_file))

        # Verify template structure
        with open(template_file, 'r') as f:
            template = json.load(f)

        self.assertIsInstance(template, list)
        self.assertGreater(len(template), 0)
        self.assertIn("question", template[0])
        self.assertIn("ground_truth", template[0])

    def test_load_nonexistent_datasets(self):
        """Test loading non-existent datasets"""
        ground_truth = self.dataset.load_ground_truth_dataset()
        synthetic = self.dataset.load_synthetic_dataset()

        self.assertEqual(len(ground_truth), 0)
        self.assertEqual(len(synthetic), 0)

    def test_save_load_synthetic_dataset(self):
        """Test saving and loading synthetic dataset"""
        # Create mock dataset
        mock_qa_data = [
            {
                "question": "What is RAG?",
                "answer": "RAG is Retrieval Augmented Generation",
                "contexts": ["Context 1", "Context 2"],
                "ground_truth": "RAG is a framework",
                "confidence_score": 0.85,
                "metadata": {"topic": "RAG"}
            }
        ]

        # Save dataset
        with open(self.dataset.synthetic_qa_file, 'w') as f:
            json.dump(mock_qa_data, f)

        # Load and verify
        loaded_data = self.dataset.load_synthetic_dataset()
        self.assertEqual(len(loaded_data), 1)
        self.assertEqual(loaded_data[0]["question"], "What is RAG?")

    def test_create_synthetic_qa_dataset(self):
        """Test synthetic Q&A dataset creation"""
        # Mock retriever
        mock_retriever = MagicMock()
        mock_response = MagicMock()
        mock_response.answer = "This is a test answer about the topic."
        mock_response.sources = [
            {"text": "Source 1 text"},
            {"text": "Source 2 text"}
        ]
        mock_response.confidence_score = 0.8
        mock_response.processing_time = 0.5

        mock_retriever.query.return_value = mock_response

        # Create dataset
        qa_pairs = self.dataset.create_synthetic_qa_dataset(mock_retriever, num_questions=3)

        self.assertGreater(len(qa_pairs), 0)
        self.assertLessEqual(len(qa_pairs), 3)

        # Verify structure
        if qa_pairs:
            qa_pair = qa_pairs[0]
            self.assertIn("question", qa_pair)
            self.assertIn("answer", qa_pair)
            self.assertIn("contexts", qa_pair)
            self.assertIn("metadata", qa_pair)

class TestRAGASEvaluator(unittest.TestCase):
    """Test RAGASEvaluator functionality"""

    def setUp(self):
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()

        # Mock retriever
        self.mock_retriever = MagicMock()
        self.mock_retriever.get_stats.return_value = {
            "total_chunks": 100,
            "vector_store_loaded": True
        }

        # Create evaluator with test directory
        self.evaluator = RAGASEvaluator(self.mock_retriever)
        self.evaluator.dataset_manager.dataset_path = self.test_dir

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator.retriever)
        self.assertIsNotNone(self.evaluator.dataset_manager)
        self.assertIsInstance(self.evaluator.metrics, list)

    def test_prepare_evaluation_dataset(self):
        """Test preparation of evaluation dataset"""
        qa_data = [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence",
                "contexts": ["AI context 1", "AI context 2"],
                "ground_truth": "AI is a field of study"
            }
        ]

        dataset = self.evaluator.prepare_evaluation_dataset(qa_data)

        # Verify dataset structure
        self.assertIn("question", dataset.column_names)
        self.assertIn("answer", dataset.column_names)
        self.assertIn("contexts", dataset.column_names)
        self.assertIn("ground_truth", dataset.column_names)

        self.assertEqual(len(dataset), 1)

    def test_run_ragas_evaluation_no_ragas(self):
        """Test RAGAS evaluation when RAGAS is not available"""
        qa_data = [
            {
                "question": "Test question",
                "answer": "Test answer",
                "contexts": ["Context"],
                "ground_truth": "Ground truth"
            }
        ]

        # Should return mock scores when RAGAS not available
        scores = self.evaluator.run_ragas_evaluation(qa_data)

        self.assertIsInstance(scores, dict)
        self.assertIn("faithfulness", scores)
        self.assertIn("context_recall", scores)

    def test_custom_metrics_evaluation(self):
        """Test custom metrics evaluation"""
        qa_data = [
            {
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of AI that enables computers to learn.",
                "contexts": ["ML context"],
                "ground_truth": "ML is learning from data",
                "confidence_score": 0.8,
                "metadata": {"topic": "ML"}
            }
        ]

        custom_metrics = self.evaluator.evaluate_custom_metrics(qa_data)

        self.assertIsInstance(custom_metrics, dict)
        self.assertIn("curriculum_coverage", custom_metrics)
        self.assertIn("learning_objective_alignment", custom_metrics)
        self.assertIn("knowledge_gap_score", custom_metrics)

        # Verify metrics are in valid range
        for metric_name, score in custom_metrics.items():
            self.assertGreaterEqual(score, 0.0, f"{metric_name} should be >= 0")
            self.assertLessEqual(score, 1.0, f"{metric_name} should be <= 1")

    def test_curriculum_coverage_assessment(self):
        """Test curriculum coverage assessment"""
        qa_data = [
            {"metadata": {"topic": "AI"}},
            {"metadata": {"topic": "ML"}},
            {"metadata": {"topic": "NLP"}},
        ]

        coverage = self.evaluator._assess_curriculum_coverage(qa_data)

        self.assertGreaterEqual(coverage, 0.0)
        self.assertLessEqual(coverage, 1.0)

    def test_learning_objectives_assessment(self):
        """Test learning objectives assessment"""
        qa_data = [
            {"question": "What is AI?"},
            {"question": "How does ML work?"},
            {"question": "Compare AI and ML"},
        ]

        alignment = self.evaluator._assess_learning_objectives(qa_data)

        self.assertGreaterEqual(alignment, 0.0)
        self.assertLessEqual(alignment, 1.0)

    def test_knowledge_gaps_identification(self):
        """Test knowledge gaps identification"""
        qa_data = [
            {"confidence_score": 0.9},
            {"confidence_score": 0.7},
            {"confidence_score": 0.5},
        ]

        gap_score = self.evaluator._identify_knowledge_gaps(qa_data)

        self.assertGreaterEqual(gap_score, 0.0)
        self.assertLessEqual(gap_score, 1.0)

    def test_answer_complexity_assessment(self):
        """Test answer complexity assessment"""
        qa_data = [
            {"answer": "Simple answer with basic vocabulary."},
            {"answer": "Complex answer with sophisticated terminology and comprehensive explanations."},
        ]

        complexity = self.evaluator._assess_answer_complexity(qa_data)

        self.assertGreaterEqual(complexity, 0.0)
        self.assertLessEqual(complexity, 1.0)

    def test_educational_value_calculation(self):
        """Test educational value calculation"""
        qa_data = [
            {
                "answer": "Comprehensive answer with multiple concepts.",
                "contexts": ["Context 1", "Context 2", "Context 3"]
            },
            {
                "answer": "Another detailed educational response.",
                "contexts": ["Context 4", "Context 5"]
            }
        ]

        value = self.evaluator._calculate_educational_value(qa_data)

        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        qa_data = [
            {
                "metadata": {
                    "processing_time": 0.5,
                    "num_sources": 3
                },
                "answer": "Good length answer for testing"
            },
            {
                "metadata": {
                    "processing_time": 0.8,
                    "num_sources": 2
                },
                "answer": "Another reasonable answer"
            }
        ]

        performance = self.evaluator.calculate_performance_metrics(qa_data)

        self.assertIn("avg_processing_time", performance)
        self.assertIn("avg_sources_used", performance)
        self.assertIn("query_success_rate", performance)

        self.assertGreaterEqual(performance["avg_processing_time"], 0.0)
        self.assertGreaterEqual(performance["avg_sources_used"], 0.0)
        self.assertGreaterEqual(performance["query_success_rate"], 0.0)
        self.assertLessEqual(performance["query_success_rate"], 1.0)

    def test_save_evaluation_results(self):
        """Test saving evaluation results"""
        results = {
            "timestamp": "2023-01-01 12:00:00",
            "ragas_scores": {"faithfulness": 0.85},
            "custom_metrics": {"curriculum_coverage": 0.8}
        }

        self.evaluator.save_evaluation_results(results)

        # Check that results directory was created
        results_dir = os.path.join(self.test_dir, "results")
        self.assertTrue(os.path.exists(results_dir))

        # Check that a results file was created
        result_files = os.listdir(results_dir)
        self.assertGreater(len(result_files), 0)

        # Verify file content
        result_file = os.path.join(results_dir, result_files[0])
        with open(result_file, 'r') as f:
            saved_results = json.load(f)

        self.assertEqual(saved_results["timestamp"], "2023-01-01 12:00:00")

    def test_comprehensive_evaluation_workflow(self):
        """Test complete evaluation workflow"""
        # Mock retriever response
        mock_response = MagicMock()
        mock_response.answer = "This is a comprehensive test answer about the educational topic."
        mock_response.sources = [{"text": "Educational source"}]
        mock_response.confidence_score = 0.8
        mock_response.processing_time = 0.6

        self.mock_retriever.query.return_value = mock_response

        # Run evaluation
        results = self.evaluator.evaluate_system_comprehensive(
            use_synthetic=True,
            num_synthetic=2
        )

        # Verify results structure
        self.assertIn("timestamp", results)
        self.assertIn("system_info", results)
        self.assertIn("ragas_scores", results)
        self.assertIn("custom_metrics", results)
        self.assertIn("dataset_info", results)
        self.assertIn("performance_metrics", results)

        # Verify dataset info
        dataset_info = results["dataset_info"]
        self.assertEqual(dataset_info["type"], "synthetic")
        self.assertGreaterEqual(dataset_info["num_samples"], 0)

    def test_empty_dataset_handling(self):
        """Test handling of empty datasets"""
        empty_data = []

        # Should handle empty data gracefully
        ragas_scores = self.evaluator.run_ragas_evaluation(empty_data)
        custom_metrics = self.evaluator.evaluate_custom_metrics(empty_data)
        performance = self.evaluator.calculate_performance_metrics(empty_data)

        self.assertIsInstance(ragas_scores, dict)
        self.assertIsInstance(custom_metrics, dict)
        self.assertIsInstance(performance, dict)

class TestEvaluationIntegration(unittest.TestCase):
    """Integration tests for evaluation framework"""

    def test_evaluation_with_mock_rag_system(self):
        """Test evaluation with a complete mock RAG system"""
        # Create mock RAG system
        mock_retriever = MagicMock()
        mock_retriever.get_stats.return_value = {
            "total_chunks": 50,
            "vector_store_loaded": True,
            "reranking_enabled": True,
            "query_expansion_enabled": True
        }

        # Mock different types of responses
        responses = [
            {
                "answer": "Artificial intelligence is a field of computer science.",
                "sources": [{"text": "AI source 1"}, {"text": "AI source 2"}],
                "confidence_score": 0.9,
                "processing_time": 0.3
            },
            {
                "answer": "Machine learning enables computers to learn from data.",
                "sources": [{"text": "ML source 1"}],
                "confidence_score": 0.7,
                "processing_time": 0.5
            }
        ]

        def mock_query(question):
            response = MagicMock()
            resp_data = responses[len(mock_retriever.query.call_args_list) % len(responses)]
            response.answer = resp_data["answer"]
            response.sources = resp_data["sources"]
            response.confidence_score = resp_data["confidence_score"]
            response.processing_time = resp_data["processing_time"]
            return response

        mock_retriever.query.side_effect = mock_query

        # Run evaluation
        evaluator = RAGASEvaluator(mock_retriever)
        results = evaluator.evaluate_system_comprehensive(num_synthetic=2)

        # Verify comprehensive results
        self.assertIn("system_info", results)
        self.assertEqual(results["system_info"]["total_chunks"], 50)
        self.assertTrue(results["system_info"]["reranking_enabled"])

if __name__ == "__main__":
    unittest.main()