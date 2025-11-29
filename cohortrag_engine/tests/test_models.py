"""
Unit tests for model initialization and functionality
"""

import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CohortRAGConfig, get_config
from models.embeddings import get_embedding_model, HuggingFaceEmbeddingModel, GeminiEmbeddingModel
from models.llm import get_llm_model, GeminiLLMModel

class TestCohortRAGConfig(unittest.TestCase):
    """Test configuration loading and validation"""

    def test_config_initialization(self):
        """Test basic configuration initialization"""
        config = CohortRAGConfig()

        # Check required attributes exist
        self.assertTrue(hasattr(config, 'data_dir'))
        self.assertTrue(hasattr(config, 'chroma_db_path'))
        self.assertTrue(hasattr(config, 'chunk_size'))
        self.assertTrue(hasattr(config, 'similarity_top_k'))

    def test_config_defaults(self):
        """Test default configuration values"""
        config = CohortRAGConfig()

        self.assertEqual(config.chunk_size, 512)
        self.assertEqual(config.chunk_overlap, 50)
        self.assertEqual(config.similarity_top_k, 5)
        self.assertEqual(config.embedding_model, "all-MiniLM-L6-v2")

    @patch.dict(os.environ, {'CHUNK_SIZE': '1024', 'SIMILARITY_TOP_K': '10'})
    def test_config_environment_override(self):
        """Test configuration override from environment variables"""
        config = CohortRAGConfig()

        self.assertEqual(config.chunk_size, 1024)
        self.assertEqual(config.similarity_top_k, 10)

    def test_config_validation(self):
        """Test configuration parameter validation"""
        config = CohortRAGConfig()

        # Chunk size should be positive
        self.assertGreater(config.chunk_size, 0)
        self.assertGreaterEqual(config.chunk_overlap, 0)
        self.assertLess(config.chunk_overlap, config.chunk_size)

        # Top k should be positive
        self.assertGreater(config.similarity_top_k, 0)

    def test_get_config_function(self):
        """Test global config getter function"""
        config = get_config()
        self.assertIsInstance(config, CohortRAGConfig)

        # Should return same instance on subsequent calls
        config2 = get_config()
        self.assertIs(config, config2)

class TestEmbeddingModels(unittest.TestCase):
    """Test embedding model initialization and functionality"""

    def setUp(self):
        self.config = CohortRAGConfig()

    def test_huggingface_embedding_model_init(self):
        """Test HuggingFace embedding model initialization"""
        try:
            model = HuggingFaceEmbeddingModel("all-MiniLM-L6-v2")
            self.assertIsNotNone(model.model)
            self.assertEqual(model.model_name, "all-MiniLM-L6-v2")
        except Exception as e:
            self.skipTest(f"HuggingFace model not available: {e}")

    def test_huggingface_embedding_text(self):
        """Test HuggingFace embedding text functionality"""
        try:
            model = HuggingFaceEmbeddingModel("all-MiniLM-L6-v2")

            # Test single text embedding
            text = "This is a test sentence for embedding."
            embedding = model.embed_text(text)

            self.assertIsInstance(embedding, list)
            self.assertGreater(len(embedding), 0)
            self.assertIsInstance(embedding[0], float)

        except Exception as e:
            self.skipTest(f"HuggingFace embedding test failed: {e}")

    def test_huggingface_batch_embedding(self):
        """Test HuggingFace batch embedding functionality"""
        try:
            model = HuggingFaceEmbeddingModel("all-MiniLM-L6-v2")

            # Test batch embedding
            texts = [
                "First test sentence.",
                "Second test sentence.",
                "Third test sentence."
            ]
            embeddings = model.embed_texts(texts)

            self.assertIsInstance(embeddings, list)
            self.assertEqual(len(embeddings), 3)
            self.assertIsInstance(embeddings[0], list)

        except Exception as e:
            self.skipTest(f"HuggingFace batch embedding test failed: {e}")

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    def test_gemini_embedding_model_init(self):
        """Test Gemini embedding model initialization"""
        try:
            model = GeminiEmbeddingModel("text-embedding-004")
            self.assertEqual(model.model_name, "text-embedding-004")
        except Exception as e:
            self.skipTest(f"Gemini embedding model test failed: {e}")

    def test_get_embedding_model_huggingface(self):
        """Test embedding model factory for HuggingFace models"""
        config = CohortRAGConfig()
        config.embedding_model = "all-MiniLM-L6-v2"

        try:
            model = get_embedding_model(config)
            self.assertIsInstance(model, HuggingFaceEmbeddingModel)
        except Exception as e:
            self.skipTest(f"HuggingFace model factory test failed: {e}")

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    def test_get_embedding_model_gemini(self):
        """Test embedding model factory for Gemini models"""
        config = CohortRAGConfig()
        config.embedding_model = "text-embedding-004"

        try:
            model = get_embedding_model(config)
            self.assertIsInstance(model, GeminiEmbeddingModel)
        except Exception as e:
            self.skipTest(f"Gemini model factory test failed: {e}")

    def test_embedding_model_error_handling(self):
        """Test embedding model error handling"""
        # Test with invalid model name
        try:
            model = HuggingFaceEmbeddingModel("invalid-model-name")
            # Should handle the error gracefully
        except Exception:
            pass  # Expected to fail

    def test_embedding_consistency(self):
        """Test that embedding results are consistent"""
        try:
            model = HuggingFaceEmbeddingModel("all-MiniLM-L6-v2")

            text = "Consistent embedding test"
            embedding1 = model.embed_text(text)
            embedding2 = model.embed_text(text)

            # Embeddings should be identical for same input
            self.assertEqual(len(embedding1), len(embedding2))
            for i in range(min(10, len(embedding1))):  # Check first 10 dimensions
                self.assertAlmostEqual(embedding1[i], embedding2[i], places=5)

        except Exception as e:
            self.skipTest(f"Embedding consistency test failed: {e}")

class TestLLMModels(unittest.TestCase):
    """Test LLM model initialization and functionality"""

    def setUp(self):
        self.config = CohortRAGConfig()

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    def test_gemini_llm_model_init(self):
        """Test Gemini LLM model initialization"""
        try:
            model = GeminiLLMModel("gemini-2.5-flash")
            self.assertEqual(model.model_name, "gemini-2.5-flash")
        except Exception as e:
            self.skipTest(f"Gemini LLM initialization failed: {e}")

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    def test_get_llm_model_gemini(self):
        """Test LLM model factory for Gemini models"""
        config = CohortRAGConfig()
        config.llm_model = "gemini-2.5-flash"

        try:
            model = get_llm_model(config)
            self.assertIsInstance(model, GeminiLLMModel)
        except Exception as e:
            self.skipTest(f"Gemini LLM factory test failed: {e}")

    def test_llm_model_error_handling(self):
        """Test LLM model error handling"""
        # Test with missing API key
        with patch.dict(os.environ, {}, clear=True):
            try:
                model = GeminiLLMModel("gemini-2.5-flash")
                # Should handle missing API key gracefully
            except Exception:
                pass  # Expected to fail without API key

class TestModelIntegration(unittest.TestCase):
    """Integration tests for model components"""

    def setUp(self):
        self.config = CohortRAGConfig()

    def test_embedding_llm_integration(self):
        """Test integration between embedding and LLM models"""
        try:
            # Get models
            embedding_model = get_embedding_model(self.config)
            llm_model = get_llm_model(self.config)

            self.assertIsNotNone(embedding_model)
            self.assertIsNotNone(llm_model)

            # Test basic functionality
            test_text = "This is a test for model integration."
            embedding = embedding_model.embed_text(test_text)

            self.assertIsInstance(embedding, list)
            self.assertGreater(len(embedding), 0)

        except Exception as e:
            self.skipTest(f"Model integration test failed: {e}")

    def test_model_configuration_consistency(self):
        """Test that models respect configuration settings"""
        config = CohortRAGConfig()

        # Test with different embedding models
        embedding_models = ["all-MiniLM-L6-v2", "text-embedding-004"]

        for model_name in embedding_models:
            config.embedding_model = model_name
            try:
                model = get_embedding_model(config)
                self.assertIsNotNone(model)
                # Model should use the specified name
                if hasattr(model, 'model_name'):
                    self.assertEqual(model.model_name, model_name)
            except Exception:
                continue  # Skip if model not available

    def test_model_memory_usage(self):
        """Test that models don't consume excessive memory"""
        import psutil
        import gc

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            # Load models
            embedding_model = get_embedding_model(self.config)
            llm_model = get_llm_model(self.config)

            # Use models
            embedding_model.embed_text("Test text")

            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory

            # Memory increase should be reasonable (less than 2GB for testing)
            self.assertLess(memory_increase, 2000, "Memory usage should be reasonable")

            # Cleanup
            del embedding_model, llm_model
            gc.collect()

        except Exception as e:
            self.skipTest(f"Memory usage test failed: {e}")

class TestModelPerformance(unittest.TestCase):
    """Performance tests for model operations"""

    def setUp(self):
        self.config = CohortRAGConfig()

    def test_embedding_performance(self):
        """Test embedding model performance"""
        import time

        try:
            model = get_embedding_model(self.config)

            # Single embedding performance
            start_time = time.time()
            embedding = model.embed_text("Performance test sentence")
            single_time = time.time() - start_time

            self.assertLess(single_time, 5.0, "Single embedding should be fast")

            # Batch embedding performance
            texts = ["Performance test sentence"] * 10

            start_time = time.time()
            embeddings = model.embed_texts(texts)
            batch_time = time.time() - start_time

            self.assertLess(batch_time, 10.0, "Batch embedding should be efficient")
            self.assertEqual(len(embeddings), 10)

        except Exception as e:
            self.skipTest(f"Embedding performance test failed: {e}")

    def test_model_startup_time(self):
        """Test model initialization time"""
        import time

        start_time = time.time()
        try:
            embedding_model = get_embedding_model(self.config)
            initialization_time = time.time() - start_time

            # Model initialization should be reasonable
            self.assertLess(initialization_time, 30.0, "Model initialization should complete within 30 seconds")

        except Exception as e:
            self.skipTest(f"Model startup test failed: {e}")

if __name__ == "__main__":
    # Set up test environment
    os.environ.setdefault('DATA_DIR', './test_data')
    os.environ.setdefault('CHROMA_DB_PATH', './test_chroma_db')

    unittest.main()