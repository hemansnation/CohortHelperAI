from typing import List, Dict, Any
import re

class TextChunker:
    """Basic text chunking utility"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks

        Args:
            text: Input text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []

        # Clean the text
        text = self._clean_text(text)

        # Split into sentences first for better boundaries
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = ""
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    "chunk_size": current_length,
                    "chunk_index": len(chunks)
                })

                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": chunk_metadata
                })

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_length = len(current_chunk)
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length

        # Add final chunk if not empty
        if current_chunk.strip():
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_size": len(current_chunk),
                "chunk_index": len(chunks)
            })

            chunks.append({
                "text": current_chunk.strip(),
                "metadata": chunk_metadata
            })

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\']+', ' ', text)

        return text.strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last overlap_size characters for chunk overlap"""
        if len(text) <= overlap_size:
            return text

        # Try to find a good breaking point (end of word)
        overlap_text = text[-overlap_size:]
        space_index = overlap_text.find(' ')

        if space_index > 0:
            return overlap_text[space_index:].strip()
        else:
            return overlap_text

def get_chunker(chunker_type: str = "basic", **kwargs) -> TextChunker:
    """Factory function to get the appropriate chunker"""
    return TextChunker(**kwargs)