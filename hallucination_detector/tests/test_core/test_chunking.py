"""
Tests for core modules: chunking, document processor.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from core.chunking import TextChunker, TextChunk
from core.document_processor import DocumentProcessor, Document


class TestTextChunker:

    @pytest.fixture
    def chunker(self):
        return TextChunker(chunk_size=100, chunk_overlap=20)

    def test_basic_chunking(self, chunker):
        """Test basic text chunking produces non-empty chunks."""
        text = "This is a test sentence. " * 20
        chunks = chunker.chunk_text(text, source="test")
        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_empty_text(self, chunker):
        """Test that empty text returns no chunks."""
        chunks = chunker.chunk_text("", source="test")
        assert chunks == []

    def test_short_text_single_chunk(self, chunker):
        """Test that short text produces a single chunk."""
        text = "Short text."
        chunks = chunker.chunk_text(text, source="test")
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_chunk_metadata(self, chunker):
        """Test that chunks contain correct metadata."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk_text(text, source="my_source")
        assert all(c.source == "my_source" for c in chunks)
        assert all(c.chunk_id >= 0 for c in chunks)

    def test_chunk_size_limit(self, chunker):
        """Test that no chunk exceeds the size limit (approximately)."""
        text = "This is a moderately long sentence that might be chunked. " * 50
        chunks = chunker.chunk_text(text, source="test")
        # Allow some tolerance for sentence-boundary splitting
        for chunk in chunks:
            assert len(chunk.content) <= chunker.chunk_size * 2  # 2x tolerance for edge cases

    def test_chunk_document(self, chunker):
        """Test chunk_document preserves metadata."""
        content = "Test content for document. Another sentence here. And one more."
        chunks = chunker.chunk_document(content, source="doc.pdf", metadata={"type": "pdf"})
        assert len(chunks) > 0
        assert all("type" in c.metadata for c in chunks)


class TestDocumentProcessor:

    @pytest.fixture
    def processor(self):
        return DocumentProcessor()

    def test_process_text_input(self, processor):
        """Test direct text processing."""
        doc = processor.process_text_input("Hello world, this is a test.", "test_source")
        assert isinstance(doc, Document)
        assert doc.content == "Hello world, this is a test."
        assert doc.source == "test_source"
        assert doc.doc_type == "text"

    def test_process_nonexistent_file(self, processor):
        """Test that processing a nonexistent file raises an error."""
        with pytest.raises(FileNotFoundError):
            processor.process_file("/nonexistent/path/file.txt")

    def test_unsupported_extension(self, processor):
        """Test that unsupported file types raise ValueError."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            f.flush()
            with pytest.raises(ValueError):
                processor.process_file(f.name)
            os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
