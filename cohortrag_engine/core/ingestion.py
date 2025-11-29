import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import json
import pickle

# Document processors
try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# Local imports
try:
    from ..config import get_config
    from ..models.embeddings import get_embedding_model
    from ..utils.chunking import get_chunker
except ImportError:
    # Fallback for direct execution
    from config import get_config
    from models.embeddings import get_embedding_model
    from utils.chunking import get_chunker

class Document:
    """Represents a processed document"""

    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata
        self.chunks = []

    def __repr__(self):
        source = self.metadata.get("source", "unknown")
        return f"Document(source='{source}', chunks={len(self.chunks)})"

class SimpleVectorStore:
    """Simple in-memory vector store for basic functionality"""

    def __init__(self, persist_path: Optional[str] = None):
        self.embeddings = []
        self.texts = []
        self.metadata = []
        self.persist_path = persist_path

    def add(self, texts: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        """Add texts, embeddings, and metadata to the store"""
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)

    def similarity_search(self, query_embedding: List[float], k: int = 3) -> List[Dict[str, Any]]:
        """Find k most similar texts to the query embedding"""
        if not self.embeddings:
            return []

        # Simple cosine similarity
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((similarity, i))

        # Sort by similarity (descending)
        similarities.sort(reverse=True)

        # Return top k results
        results = []
        for similarity, idx in similarities[:k]:
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": similarity
            })

        return results

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def persist(self):
        """Save the vector store to disk"""
        if self.persist_path:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            with open(self.persist_path, 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings,
                    'texts': self.texts,
                    'metadata': self.metadata
                }, f)

    def load(self):
        """Load the vector store from disk"""
        if self.persist_path and os.path.exists(self.persist_path):
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.texts = data['texts']
                self.metadata = data['metadata']
            return True
        return False

class DocumentProcessor:
    """Processes various document formats"""

    def __init__(self):
        self.supported_formats = ['.txt', '.md']
        if HAS_PYPDF:
            self.supported_formats.append('.pdf')

    def load_document(self, file_path: str) -> Optional[Document]:
        """Load a document from file"""
        path = Path(file_path)

        if not path.exists():
            print(f"File not found: {file_path}")
            return None

        if path.suffix.lower() not in self.supported_formats:
            print(f"Unsupported format: {path.suffix}")
            return None

        try:
            if path.suffix.lower() == '.pdf':
                return self._load_pdf(path)
            else:
                return self._load_text_file(path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _load_pdf(self, path: Path) -> Optional[Document]:
        """Load PDF document"""
        if not HAS_PYPDF:
            print("PyPDF not available. Install with: pip install pypdf")
            return None

        try:
            text = ""
            with open(path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n\n--- Page {page_num + 1} ---\n\n"
                        text += page_text

            metadata = {
                "source": str(path),
                "format": "pdf",
                "pages": len(pdf_reader.pages),
                "file_size": path.stat().st_size
            }

            return Document(text.strip(), metadata)

        except Exception as e:
            print(f"Error reading PDF {path}: {e}")
            return None

    def _load_text_file(self, path: Path) -> Optional[Document]:
        """Load text file (txt, md)"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()

            metadata = {
                "source": str(path),
                "format": path.suffix.lower()[1:],  # Remove the dot
                "file_size": path.stat().st_size
            }

            return Document(text, metadata)

        except Exception as e:
            print(f"Error reading text file {path}: {e}")
            return None

class CohortRAGIngestionPipeline:
    """Main ingestion pipeline for CohortRAG"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.embedding_model = get_embedding_model(self.config)
        self.chunker = get_chunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.document_processor = DocumentProcessor()

        # Initialize vector store
        persist_path = os.path.join(self.config.chroma_db_path, "simple_store.pkl")
        self.vector_store = SimpleVectorStore(persist_path)

        # Try to load existing data
        if self.vector_store.load():
            print(f"Loaded existing vector store with {len(self.vector_store.texts)} chunks")

    def ingest_directory(self, directory_path: str = None) -> bool:
        """Ingest all supported documents from a directory"""
        if directory_path is None:
            directory_path = self.config.data_dir

        path = Path(directory_path)
        if not path.exists():
            print(f"Directory not found: {directory_path}")
            return False

        # Find all supported files
        files_to_process = []
        for ext in self.document_processor.supported_formats:
            files_to_process.extend(path.glob(f"*{ext}"))

        if not files_to_process:
            print(f"No supported files found in {directory_path}")
            print(f"Supported formats: {', '.join(self.document_processor.supported_formats)}")
            return False

        print(f"Found {len(files_to_process)} files to process:")
        for file_path in files_to_process:
            print(f"  - {file_path.name}")

        # Process each file
        total_chunks = 0
        for file_path in tqdm(files_to_process, desc="Processing files"):
            chunks_added = self._process_file(file_path)
            total_chunks += chunks_added

        # Persist the vector store
        self.vector_store.persist()

        print(f"Ingestion complete! Added {total_chunks} chunks from {len(files_to_process)} files")
        return True

    def _process_file(self, file_path: Path) -> int:
        """Process a single file"""
        # Load document
        document = self.document_processor.load_document(str(file_path))
        if not document:
            return 0

        # Create chunks
        chunks = self.chunker.chunk_text(document.text, document.metadata)
        if not chunks:
            print(f"No chunks created for {file_path}")
            return 0

        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        try:
            embeddings = self.embedding_model.embed_batch(texts)
        except Exception as e:
            print(f"Error generating embeddings for {file_path}: {e}")
            return 0

        # Prepare metadata
        metadata_list = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = chunk["metadata"].copy()
            chunk_metadata.update({
                "file_name": file_path.name,
                "chunk_id": f"{file_path.stem}_{i}"
            })
            metadata_list.append(chunk_metadata)

        # Add to vector store
        self.vector_store.add(texts, embeddings, metadata_list)

        return len(chunks)

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        return {
            "total_chunks": len(self.vector_store.texts),
            "total_files": len(set(meta.get("file_name", "") for meta in self.vector_store.metadata)),
            "supported_formats": self.document_processor.supported_formats,
            "config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "data_dir": self.config.data_dir
            }
        }

def main():
    """Test the ingestion pipeline"""
    try:
        pipeline = CohortRAGIngestionPipeline()
        print("CohortRAG Ingestion Pipeline")
        print("=" * 40)

        # Show current stats
        stats = pipeline.get_stats()
        print(f"Current stats: {stats['total_chunks']} chunks from {stats['total_files']} files")

        # Run ingestion
        success = pipeline.ingest_directory()

        if success:
            # Show updated stats
            stats = pipeline.get_stats()
            print(f"Final stats: {stats['total_chunks']} chunks from {stats['total_files']} files")
        else:
            print("Ingestion failed!")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()