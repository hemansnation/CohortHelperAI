"""
Asynchronous ingestion pipeline for production-ready document processing
======================================================================

This module provides async document ingestion with performance optimization,
batch processing, and scalable architecture for large document collections.
"""

import asyncio
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
import logging

# Local imports
try:
    from ..config import get_config
    from ..models.embeddings import get_embedding_model
    from ..utils.chunking import get_chunker
    from ..utils.performance import (
        AsyncIngestionProcessor,
        PerformanceMonitor,
        MemoryMonitor,
        async_timer,
        performance_monitor
    )
    from .ingestion import Document, SimpleVectorStore
except ImportError:
    # Fallback for direct execution
    from config import get_config
    from models.embeddings import get_embedding_model
    from utils.chunking import get_chunker
    from utils.performance import (
        AsyncIngestionProcessor,
        PerformanceMonitor,
        MemoryMonitor,
        async_timer,
        performance_monitor
    )
    from core.ingestion import Document, SimpleVectorStore

class AsyncCohortRAGIngestion:
    """Asynchronous ingestion pipeline for large-scale document processing"""

    def __init__(self, config=None, max_workers: int = 4, batch_size: int = 50):
        """
        Initialize async ingestion pipeline

        Args:
            config: Configuration object
            max_workers: Maximum number of worker threads
            batch_size: Number of documents to process in each batch
        """
        self.config = config or get_config()
        self.max_workers = max_workers
        self.batch_size = batch_size

        # Initialize components
        self.embedding_model = get_embedding_model(self.config)
        self.chunker = get_chunker(
            self.config.chunker_type,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        # Initialize async processor and monitoring
        self.async_processor = AsyncIngestionProcessor(max_workers, batch_size)
        self.memory_monitor = MemoryMonitor()

        # Vector store
        persist_path = os.path.join(self.config.chroma_db_path, "simple_store.pkl")
        self.vector_store = SimpleVectorStore(persist_path)

    @async_timer("async_document_ingestion")
    async def ingest_documents_async(self, data_dir: str,
                                   file_patterns: List[str] = None,
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Asynchronously ingest documents from directory

        Args:
            data_dir: Directory containing documents
            file_patterns: List of file patterns to match (e.g., ['*.pdf', '*.txt'])
            progress_callback: Callback function for progress updates

        Returns:
            Ingestion results and statistics
        """
        self.memory_monitor.start_monitoring()
        start_time = time.time()

        print(f"🚀 Starting async ingestion from: {data_dir}")

        # Discover documents
        documents = await self._discover_documents_async(data_dir, file_patterns)
        print(f"📁 Found {len(documents)} documents to process")

        if not documents:
            return {
                "success": False,
                "message": "No documents found",
                "stats": self._get_empty_stats()
            }

        # Process documents in batches
        processed_results = await self._process_documents_in_batches(
            documents, progress_callback
        )

        # Aggregate results
        results = await self._aggregate_results(processed_results)

        # Save to vector store
        if results['chunks']:
            await self._save_to_vector_store_async(results['chunks'])

        # Calculate final statistics
        end_time = time.time()
        total_time = end_time - start_time

        final_stats = {
            "success": True,
            "total_documents": len(documents),
            "total_chunks": len(results['chunks']),
            "total_tokens": results.get('total_tokens', 0),
            "processing_time": total_time,
            "avg_time_per_doc": total_time / len(documents) if documents else 0,
            "memory_usage": self.memory_monitor.get_usage(),
            "failed_documents": results.get('failed_documents', []),
            "performance_breakdown": self._get_performance_breakdown()
        }

        print(f"✅ Async ingestion completed in {total_time:.2f}s")
        print(f"📊 Processed {final_stats['total_documents']} documents → {final_stats['total_chunks']} chunks")

        return final_stats

    async def _discover_documents_async(self, data_dir: str,
                                      file_patterns: List[str] = None) -> List[Path]:
        """Asynchronously discover documents in directory"""
        if file_patterns is None:
            file_patterns = ['*.pdf', '*.txt', '*.md', '*.docx']

        loop = asyncio.get_event_loop()

        def discover_sync():
            data_path = Path(data_dir)
            documents = []
            for pattern in file_patterns:
                documents.extend(data_path.glob(f"**/{pattern}"))
            return list(set(documents))  # Remove duplicates

        with ThreadPoolExecutor() as executor:
            documents = await loop.run_in_executor(executor, discover_sync)

        return documents

    async def _process_documents_in_batches(self, documents: List[Path],
                                          progress_callback: Optional[Callable] = None) -> List[Any]:
        """Process documents in batches with async coordination"""

        async def batch_progress(current_batch: int, total_batches: int, processed_docs: int):
            """Internal progress tracking"""
            if progress_callback:
                await asyncio.get_event_loop().run_in_executor(
                    None, progress_callback, processed_docs, len(documents)
                )

            print(f"📈 Batch {current_batch}/{total_batches} completed ({processed_docs}/{len(documents)} docs)")
            self.memory_monitor.update_peak()

        # Use the async processor
        results = await self.async_processor.process_documents_async(
            documents,
            self._process_single_document_sync,
            batch_progress
        )

        return results

    def _process_single_document_sync(self, document_path: Path) -> Dict[str, Any]:
        """
        Process a single document synchronously (called from async context)

        This is the core document processing function that runs in thread pool
        """
        try:
            # Load document
            document = self._load_document(document_path)
            if not document:
                return {"error": f"Failed to load {document_path}", "chunks": []}

            # Chunk document
            chunks = self.chunker.chunk_text(document.content)

            # Process chunks
            processed_chunks = []
            total_tokens = 0

            for i, chunk_text in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = self.embedding_model.embed_text(chunk_text)

                    # Create chunk metadata
                    metadata = self.chunker.create_chunk_metadata(
                        chunk_text, i, document.metadata
                    )

                    chunk_data = {
                        "text": chunk_text,
                        "embedding": embedding,
                        "metadata": metadata
                    }

                    processed_chunks.append(chunk_data)
                    total_tokens += len(chunk_text.split())

                except Exception as e:
                    logging.warning(f"Failed to process chunk {i} from {document_path}: {e}")
                    continue

            return {
                "document_path": str(document_path),
                "chunks": processed_chunks,
                "tokens": total_tokens,
                "chunk_count": len(processed_chunks)
            }

        except Exception as e:
            logging.error(f"Error processing document {document_path}: {e}")
            return {
                "error": str(e),
                "document_path": str(document_path),
                "chunks": []
            }

    def _load_document(self, file_path: Path) -> Optional[Document]:
        """Load document from file (synchronous helper)"""
        try:
            if file_path.suffix.lower() == '.pdf':
                return self._load_pdf(file_path)
            elif file_path.suffix.lower() in ['.txt', '.md']:
                return self._load_text(file_path)
            elif file_path.suffix.lower() == '.docx':
                return self._load_docx(file_path)
            else:
                logging.warning(f"Unsupported file type: {file_path}")
                return None
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            return None

    def _load_pdf(self, file_path: Path) -> Optional[Document]:
        """Load PDF document"""
        try:
            import pypdf
            content = ""
            with open(file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                for page in reader.pages:
                    content += page.extract_text() + "\n"

            metadata = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": "pdf",
                "page_count": len(reader.pages) if 'reader' in locals() else 0,
                "file_size": file_path.stat().st_size
            }

            return Document(content=content.strip(), metadata=metadata)

        except Exception as e:
            logging.error(f"Error loading PDF {file_path}: {e}")
            return None

    def _load_text(self, file_path: Path) -> Optional[Document]:
        """Load text document"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            metadata = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix[1:],  # Remove dot
                "file_size": file_path.stat().st_size
            }

            return Document(content=content, metadata=metadata)

        except Exception as e:
            logging.error(f"Error loading text file {file_path}: {e}")
            return None

    def _load_docx(self, file_path: Path) -> Optional[Document]:
        """Load DOCX document"""
        try:
            import python_docx
            doc = python_docx.Document(file_path)
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])

            metadata = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": "docx",
                "paragraph_count": len(doc.paragraphs),
                "file_size": file_path.stat().st_size
            }

            return Document(content=content, metadata=metadata)

        except Exception as e:
            logging.error(f"Error loading DOCX {file_path}: {e}")
            return None

    async def _aggregate_results(self, processed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from all processed documents"""
        all_chunks = []
        total_tokens = 0
        failed_documents = []

        for result in processed_results:
            if isinstance(result, dict) and 'error' not in result:
                all_chunks.extend(result.get('chunks', []))
                total_tokens += result.get('tokens', 0)
            else:
                # Handle errors
                if isinstance(result, dict) and 'error' in result:
                    failed_documents.append(result)
                else:
                    failed_documents.append({'error': str(result)})

        return {
            'chunks': all_chunks,
            'total_tokens': total_tokens,
            'failed_documents': failed_documents
        }

    async def _save_to_vector_store_async(self, chunks: List[Dict[str, Any]]):
        """Save processed chunks to vector store asynchronously"""
        loop = asyncio.get_event_loop()

        def save_sync():
            # Clear existing data
            self.vector_store.texts = []
            self.vector_store.embeddings = []
            self.vector_store.metadatas = []

            # Add new data
            for chunk in chunks:
                self.vector_store.texts.append(chunk['text'])
                self.vector_store.embeddings.append(chunk['embedding'])
                self.vector_store.metadatas.append(chunk['metadata'])

            # Save to disk
            success = self.vector_store.save()
            return success

        with ThreadPoolExecutor() as executor:
            success = await loop.run_in_executor(executor, save_sync)

        if success:
            print(f"💾 Saved {len(chunks)} chunks to vector store")
        else:
            print("❌ Failed to save to vector store")

        return success

    def _get_performance_breakdown(self) -> Dict[str, Any]:
        """Get detailed performance breakdown"""
        return performance_monitor.get_summary("async_document_ingestion")

    def _get_empty_stats(self) -> Dict[str, Any]:
        """Get empty statistics structure"""
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_tokens": 0,
            "processing_time": 0,
            "avg_time_per_doc": 0,
            "memory_usage": {},
            "failed_documents": [],
            "performance_breakdown": {}
        }

    async def stream_process_documents(self, documents: List[Path]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream document processing results as they become available

        Args:
            documents: List of document paths to process

        Yields:
            Individual document processing results
        """
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_and_yield(doc_path: Path):
            async with semaphore:
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor, self._process_single_document_sync, doc_path
                    )
                    return result

        # Create tasks for all documents
        tasks = [process_and_yield(doc) for doc in documents]

        # Yield results as they complete
        for task in asyncio.as_completed(tasks):
            result = await task
            yield result

    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get current ingestion statistics"""
        return {
            "configuration": {
                "max_workers": self.max_workers,
                "batch_size": self.batch_size,
                "chunker_type": self.config.chunker_type,
                "chunk_size": self.config.chunk_size,
                "embedding_model": self.config.embedding_model
            },
            "vector_store": {
                "total_chunks": len(self.vector_store.texts),
                "persist_path": self.vector_store.persist_path,
                "loaded": len(self.vector_store.texts) > 0
            },
            "performance": performance_monitor.get_summary(),
            "memory_usage": self.memory_monitor.get_usage()
        }

async def main():
    """Test async ingestion pipeline"""
    print("🧪 Testing Async Ingestion Pipeline")
    print("=" * 40)

    try:
        # Initialize async ingestion
        async_ingestion = AsyncCohortRAGIngestion(max_workers=2, batch_size=5)

        # Test with data directory
        config = get_config()
        data_dir = config.data_dir

        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            return

        # Run async ingestion
        def progress_callback(current: int, total: int):
            progress = (current / total) * 100
            print(f"⏳ Progress: {current}/{total} ({progress:.1f}%)")

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
        print(f"  🧠 Memory: {results.get('memory_usage', {}).get('peak_mb', 0):.1f}MB peak")

        if results.get('failed_documents'):
            print(f"  ⚠️  Failed: {len(results['failed_documents'])} documents")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())