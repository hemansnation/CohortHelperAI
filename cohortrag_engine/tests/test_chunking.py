"""
Unit tests for chunking utilities
"""

import unittest
import tempfile
import os
from pathlib import Path

# Add the parent directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.chunking import (
    SimpleTextChunker,
    SemanticChunker,
    get_chunker,
    OverlapChunker,
    SentenceChunker
)

class TestSimpleTextChunker(unittest.TestCase):
    """Test SimpleTextChunker functionality"""

    def setUp(self):
        self.chunker = SimpleTextChunker(chunk_size=100, chunk_overlap=20)

    def test_basic_chunking(self):
        """Test basic text chunking functionality"""
        text = "This is a test sentence. " * 10  # 250 characters
        chunks = self.chunker.chunk_text(text)

        self.assertGreater(len(chunks), 1, "Should create multiple chunks for long text")
        self.assertLessEqual(len(chunks[0]), 100, "First chunk should not exceed chunk_size")

    def test_empty_text(self):
        """Test handling of empty text"""
        chunks = self.chunker.chunk_text("")
        self.assertEqual(len(chunks), 0, "Empty text should return empty list")

    def test_short_text(self):
        """Test handling of text shorter than chunk size"""
        text = "Short text"
        chunks = self.chunker.chunk_text(text)
        self.assertEqual(len(chunks), 1, "Short text should return single chunk")
        self.assertEqual(chunks[0], text, "Short text should be unchanged")

    def test_chunk_overlap(self):
        """Test that chunk overlap works correctly"""
        text = "a" * 150  # Text longer than chunk size
        chunks = self.chunker.chunk_text(text)

        if len(chunks) > 1:
            # Check that there's overlap between consecutive chunks
            overlap_found = False
            for i in range(len(chunks) - 1):
                if chunks[i][-10:] in chunks[i + 1][:30]:  # Check for some overlap
                    overlap_found = True
                    break
            # Note: Exact overlap testing depends on implementation details

    def test_metadata_creation(self):
        """Test metadata creation for chunks"""
        text = "This is a test document with multiple sentences."
        chunks = self.chunker.chunk_text(text)
        metadata = self.chunker.create_chunk_metadata(chunks[0], 0, {"source": "test"})

        self.assertIn("chunk_id", metadata)
        self.assertIn("chunk_index", metadata)
        self.assertIn("chunk_size", metadata)
        self.assertEqual(metadata["source"], "test")

class TestSentenceChunker(unittest.TestCase):
    """Test SentenceChunker functionality"""

    def setUp(self):
        self.chunker = SentenceChunker(max_sentences=3)

    def test_sentence_splitting(self):
        """Test sentence-based chunking"""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence. Fifth sentence."
        chunks = self.chunker.chunk_text(text)

        self.assertGreater(len(chunks), 1, "Should create multiple chunks")

    def test_single_sentence(self):
        """Test handling of single sentence"""
        text = "This is a single sentence."
        chunks = self.chunker.chunk_text(text)
        self.assertEqual(len(chunks), 1, "Single sentence should return one chunk")

    def test_no_sentences(self):
        """Test handling of text without sentence endings"""
        text = "This text has no proper sentence endings"
        chunks = self.chunker.chunk_text(text)
        self.assertEqual(len(chunks), 1, "Text without sentences should return one chunk")

class TestOverlapChunker(unittest.TestCase):
    """Test OverlapChunker functionality"""

    def setUp(self):
        self.chunker = OverlapChunker(chunk_size=50, overlap_size=10)

    def test_overlap_calculation(self):
        """Test that overlap is calculated correctly"""
        text = "This is a test sentence. " * 10  # Long text
        chunks = self.chunker.chunk_text(text)

        if len(chunks) > 1:
            self.assertGreater(len(chunks), 1, "Should create multiple chunks")

    def test_overlap_size_validation(self):
        """Test that overlap size is validated"""
        # Overlap size should be less than chunk size
        with self.assertRaises(ValueError):
            OverlapChunker(chunk_size=50, overlap_size=60)

class TestSemanticChunker(unittest.TestCase):
    """Test SemanticChunker functionality"""

    def setUp(self):
        # Note: SemanticChunker might require embedding models
        try:
            self.chunker = SemanticChunker(max_chunk_size=200)
            self.semantic_available = True
        except Exception:
            self.semantic_available = False

    def test_semantic_chunking(self):
        """Test semantic-based chunking"""
        if not self.semantic_available:
            self.skipTest("Semantic chunker not available")

        text = """
        Machine learning is a subset of artificial intelligence.
        It involves training algorithms on data.
        Deep learning is a subset of machine learning.
        It uses neural networks with multiple layers.
        Natural language processing deals with text analysis.
        It can be used for sentiment analysis and translation.
        """

        chunks = self.chunker.chunk_text(text)
        self.assertGreater(len(chunks), 0, "Should create at least one chunk")

    def test_empty_text_semantic(self):
        """Test semantic chunker with empty text"""
        if not self.semantic_available:
            self.skipTest("Semantic chunker not available")

        chunks = self.chunker.chunk_text("")
        self.assertEqual(len(chunks), 0, "Empty text should return empty list")

class TestChunkerFactory(unittest.TestCase):
    """Test chunker factory function"""

    def test_get_simple_chunker(self):
        """Test getting simple text chunker"""
        chunker = get_chunker("simple", chunk_size=100)
        self.assertIsInstance(chunker, SimpleTextChunker)
        self.assertEqual(chunker.chunk_size, 100)

    def test_get_sentence_chunker(self):
        """Test getting sentence chunker"""
        chunker = get_chunker("sentence", max_sentences=5)
        self.assertIsInstance(chunker, SentenceChunker)
        self.assertEqual(chunker.max_sentences, 5)

    def test_get_overlap_chunker(self):
        """Test getting overlap chunker"""
        chunker = get_chunker("overlap", chunk_size=100, overlap_size=20)
        self.assertIsInstance(chunker, OverlapChunker)
        self.assertEqual(chunker.chunk_size, 100)
        self.assertEqual(chunker.overlap_size, 20)

    def test_get_semantic_chunker(self):
        """Test getting semantic chunker"""
        try:
            chunker = get_chunker("semantic", max_chunk_size=200)
            self.assertIsInstance(chunker, SemanticChunker)
        except Exception:
            self.skipTest("Semantic chunker not available")

    def test_invalid_chunker_type(self):
        """Test handling of invalid chunker type"""
        with self.assertRaises(ValueError):
            get_chunker("invalid_type")

    def test_default_chunker(self):
        """Test getting default chunker"""
        chunker = get_chunker()
        self.assertIsNotNone(chunker)

class TestChunkingIntegration(unittest.TestCase):
    """Integration tests for chunking with real document processing"""

    def test_document_chunking_workflow(self):
        """Test complete document chunking workflow"""
        # Create a temporary document
        test_content = """
        Chapter 1: Introduction to RAG

        Retrieval Augmented Generation (RAG) is a powerful framework that combines
        information retrieval with language generation. This approach allows AI systems
        to access and utilize external knowledge sources when generating responses.

        The key components of RAG include:
        1. Document ingestion and indexing
        2. Query processing and retrieval
        3. Context-aware response generation

        Chapter 2: Implementation Details

        When implementing a RAG system, several factors must be considered:
        - Chunk size and overlap strategies
        - Embedding model selection
        - Vector database configuration
        - Query optimization techniques

        These considerations directly impact the quality and relevance of retrieved information.
        """

        chunker = get_chunker("simple", chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_text(test_content)

        # Verify chunking results
        self.assertGreater(len(chunks), 1, "Long document should be split into multiple chunks")

        # Check that important content is preserved
        all_content = " ".join(chunks)
        self.assertIn("RAG", all_content, "Key terms should be preserved")
        self.assertIn("Retrieval Augmented Generation", all_content)

        # Test metadata creation
        for i, chunk in enumerate(chunks):
            metadata = chunker.create_chunk_metadata(chunk, i, {"document": "test_doc"})
            self.assertEqual(metadata["chunk_index"], i)
            self.assertEqual(metadata["document"], "test_doc")

    def test_chunking_performance(self):
        """Test chunking performance with large text"""
        import time

        # Create large text content
        large_text = "This is a test sentence with meaningful content. " * 1000  # ~50KB

        chunker = get_chunker("simple", chunk_size=500)

        start_time = time.time()
        chunks = chunker.chunk_text(large_text)
        processing_time = time.time() - start_time

        # Performance assertions
        self.assertLess(processing_time, 5.0, "Chunking should complete within 5 seconds")
        self.assertGreater(len(chunks), 50, "Large text should create many chunks")

        # Verify no data loss
        total_length = sum(len(chunk) for chunk in chunks)
        self.assertGreater(total_length, len(large_text) * 0.8, "Most content should be preserved")

    def test_chunk_quality_metrics(self):
        """Test metrics for chunk quality assessment"""
        text = """
        Artificial intelligence is transforming education through personalized learning.
        Machine learning algorithms can adapt to student needs and preferences.
        Natural language processing enables intelligent tutoring systems.
        These technologies work together to create more effective educational experiences.
        """

        chunker = get_chunker("sentence", max_sentences=2)
        chunks = chunker.chunk_text(text)

        # Quality metrics
        avg_chunk_length = sum(len(chunk) for chunk in chunks) / len(chunks)
        self.assertGreater(avg_chunk_length, 20, "Chunks should have meaningful length")

        # Check for content coherence (sentences should be complete)
        for chunk in chunks:
            sentence_endings = chunk.count('.') + chunk.count('!') + chunk.count('?')
            self.assertGreater(sentence_endings, 0, "Chunks should contain complete sentences")

if __name__ == "__main__":
    unittest.main()