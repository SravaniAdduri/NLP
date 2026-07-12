"""
Text Chunking Module
Splits documents into smaller chunks for embedding and retrieval.
Supports multiple chunking strategies: fixed-size, sentence-based, and recursive.
"""

from typing import List
from dataclasses import dataclass, field
import re

from loguru import logger


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata about its position."""
    content: str
    chunk_id: int
    source: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """
    Splits text into overlapping chunks for embedding.
    Supports fixed-size chunking with overlap and sentence-aware boundaries.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    def chunk_text(self, text: str, source: str = "") -> List[TextChunk]:
        """
        Split text into chunks using sentence-aware boundaries.
        
        Tries to split at sentence boundaries within the chunk_size limit.
        Falls back to character-level splitting if sentences are too long.
        
        Args:
            text: Text to split into chunks.
            source: Source identifier for the text.
            
        Returns:
            List of TextChunk objects.
        """
        if not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_start = 0
        char_position = 0

        for sentence in sentences:
            sentence_with_space = sentence if not current_chunk else " " + sentence

            if len(current_chunk) + len(sentence_with_space) <= self.chunk_size:
                current_chunk += sentence_with_space
            else:
                if current_chunk:
                    chunks.append(TextChunk(
                        content=current_chunk.strip(),
                        chunk_id=len(chunks),
                        source=source,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                        metadata={"chunk_method": "sentence_aware"},
                    ))
                    # Calculate overlap start position
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    # Single sentence longer than chunk_size - force split
                    sub_chunks = self._force_split(sentence, source, len(chunks), current_start)
                    chunks.extend(sub_chunks)
                    current_start = current_start + len(sentence)
                    current_chunk = ""

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(TextChunk(
                content=current_chunk.strip(),
                chunk_id=len(chunks),
                source=source,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata={"chunk_method": "sentence_aware"},
            ))

        logger.debug(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks

    def chunk_document(self, content: str, source: str, metadata: dict = None) -> List[TextChunk]:
        """
        Chunk a full document, preserving document-level metadata in each chunk.
        
        Args:
            content: Document text content.
            source: Document source path or identifier.
            metadata: Additional metadata to attach to each chunk.
            
        Returns:
            List of TextChunk objects with document metadata.
        """
        chunks = self.chunk_text(content, source)

        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex pattern."""
        sentences = self._sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _force_split(self, text: str, source: str, start_id: int, char_offset: int) -> List[TextChunk]:
        """Force split a long text that exceeds chunk_size at word boundaries."""
        chunks = []
        words = text.split()
        current = ""
        current_start = char_offset

        for word in words:
            if len(current) + len(word) + 1 <= self.chunk_size:
                current = current + " " + word if current else word
            else:
                if current:
                    chunks.append(TextChunk(
                        content=current.strip(),
                        chunk_id=start_id + len(chunks),
                        source=source,
                        start_char=current_start,
                        end_char=current_start + len(current),
                        metadata={"chunk_method": "force_split"},
                    ))
                    current_start += len(current)
                current = word

        if current:
            chunks.append(TextChunk(
                content=current.strip(),
                chunk_id=start_id + len(chunks),
                source=source,
                start_char=current_start,
                end_char=current_start + len(current),
                metadata={"chunk_method": "force_split"},
            ))

        return chunks
