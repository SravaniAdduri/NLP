"""
Retrieval Agent
Responsible for:
1. Retrieving relevant documents from the FAISS vector store.
2. Supporting multiple retrieval strategies.
3. Handling different document types (PDF, text, web).
"""

from typing import List, Optional
from dataclasses import dataclass

from loguru import logger

from core.embeddings import EmbeddingEngine
from core.vector_store import FAISSVectorStore, SearchResult
from core.document_processor import DocumentProcessor, Document
from core.chunking import TextChunker, TextChunk


@dataclass
class RetrievalResult:
    """Complete retrieval result with source documents and metadata."""
    query: str
    results: List[SearchResult]
    num_results: int
    retrieval_method: str


class RetrievalAgent:
    """
    Retrieves relevant documents from the vector store.
    
    Handles:
    - Document ingestion (PDF, text, web)
    - Chunking and embedding
    - Similarity search
    - Multi-query retrieval for better recall
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: FAISSVectorStore,
        chunker: TextChunker,
        document_processor: DocumentProcessor,
        top_k: int = 20,
    ):
        """
        Initialize the Retrieval Agent.
        
        Args:
            embedding_engine: Engine for generating embeddings.
            vector_store: FAISS vector store instance.
            chunker: Text chunking utility.
            document_processor: Document ingestion processor.
            top_k: Number of documents to retrieve.
        """
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.chunker = chunker
        self.document_processor = document_processor
        self.top_k = top_k
        logger.info(f"Retrieval Agent initialized (top_k={top_k})")

    def ingest_document(self, file_path: str) -> int:
        """
        Ingest a single document into the vector store.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            Number of chunks added to the vector store.
        """
        # Process the document
        document = self.document_processor.process_file(file_path)
        return self._add_document_to_store(document)

    def ingest_directory(self, directory_path: str) -> int:
        """
        Ingest all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory.
            
        Returns:
            Total number of chunks added.
        """
        documents = self.document_processor.process_directory(directory_path)
        total_chunks = 0
        for doc in documents:
            total_chunks += self._add_document_to_store(doc)
        logger.info(f"Ingested {len(documents)} documents, {total_chunks} total chunks")
        return total_chunks

    def ingest_url(self, url: str) -> int:
        """
        Ingest a web page into the vector store.
        
        Args:
            url: URL to fetch and ingest.
            
        Returns:
            Number of chunks added.
        """
        document = self.document_processor.process_url(url)
        return self._add_document_to_store(document)

    def ingest_text(self, text: str, source_name: str = "direct_input") -> int:
        """
        Ingest raw text directly into the vector store.
        
        Args:
            text: Raw text content.
            source_name: Identifier for the text source.
            
        Returns:
            Number of chunks added.
        """
        document = self.document_processor.process_text_input(text, source_name)
        return self._add_document_to_store(document)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query string.
            top_k: Override default top_k if provided.
            
        Returns:
            RetrievalResult with matching documents.
        """
        k = top_k or self.top_k

        # Generate query embedding
        query_embedding = self.embedding_engine.embed_query(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=k)

        logger.info(f"Retrieved {len(results)} results for query: '{query[:50]}...'")

        return RetrievalResult(
            query=query,
            results=results,
            num_results=len(results),
            retrieval_method="dense_retrieval",
        )

    def multi_query_retrieve(self, queries: List[str], top_k: Optional[int] = None) -> RetrievalResult:
        """
        Retrieve documents using multiple query variations for better recall.
        Merges results from all queries and deduplicates.
        
        Args:
            queries: List of query variations.
            top_k: Number of total results to return.
            
        Returns:
            Merged and deduplicated RetrievalResult.
        """
        k = top_k or self.top_k
        all_results = []
        seen_contents = set()

        for query in queries:
            query_embedding = self.embedding_engine.embed_query(query)
            results = self.vector_store.search(query_embedding, top_k=k)

            for result in results:
                content_key = result.content[:100]  # Use first 100 chars as key
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    all_results.append(result)

        # Sort by score and take top_k
        all_results.sort(key=lambda x: x.score, reverse=True)
        final_results = all_results[:k]

        return RetrievalResult(
            query=queries[0],
            results=final_results,
            num_results=len(final_results),
            retrieval_method="multi_query_retrieval",
        )

    def _add_document_to_store(self, document: Document) -> int:
        """
        Chunk a document, generate embeddings, and add to vector store.
        
        Args:
            document: Processed Document object.
            
        Returns:
            Number of chunks added.
        """
        # Chunk the document
        chunks = self.chunker.chunk_document(
            content=document.content,
            source=document.source,
            metadata=document.metadata,
        )

        if not chunks:
            logger.warning(f"No chunks generated from document: {document.source}")
            return 0

        # Generate embeddings for all chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_engine.embed_texts(chunk_texts)

        # Add to vector store
        self.vector_store.add_chunks(chunks, embeddings)

        logger.info(f"Added {len(chunks)} chunks from {document.source}")
        return len(chunks)

    def save_index(self, index_name: str = "default") -> None:
        """Save the current vector store index to disk."""
        self.vector_store.save(index_name)

    def load_index(self, index_name: str = "default") -> bool:
        """Load a vector store index from disk."""
        return self.vector_store.load(index_name)

    @property
    def index_size(self) -> int:
        """Return the number of vectors in the store."""
        return self.vector_store.size
