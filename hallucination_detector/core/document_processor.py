"""
Document Processor Module
Handles ingestion of PDFs, text files, and web pages.
Extracts text content and metadata from various document formats.
"""

import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from loguru import logger


@dataclass
class Document:
    """Represents a processed document with content and metadata."""
    content: str
    source: str
    doc_type: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = f"{self.doc_type}_{hash(self.source) % 10**8}"


class DocumentProcessor:
    """
    Processes documents from multiple sources: PDF, text files, and web pages.
    Returns structured Document objects with extracted text and metadata.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}

    def process_file(self, file_path: str) -> Document:
        """
        Process a single file and return a Document object.
        
        Args:
            file_path: Path to the file to process.
            
        Returns:
            Document object with extracted content.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}. Supported: {self.SUPPORTED_EXTENSIONS}")

        logger.info(f"Processing file: {file_path}")

        if extension == ".pdf":
            return self._process_pdf(path)
        elif extension in {".txt", ".md"}:
            return self._process_text(path)
        elif extension in {".html", ".htm"}:
            return self._process_html_file(path)
        else:
            raise ValueError(f"No processor for extension: {extension}")

    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Process all supported files in a directory recursively.
        
        Args:
            directory_path: Path to the directory to process.
            
        Returns:
            List of Document objects.
        """
        path = Path(directory_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        documents = []
        for file_path in path.rglob("*"):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.process_file(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

        logger.info(f"Processed {len(documents)} documents from {directory_path}")
        return documents

    def process_url(self, url: str) -> Document:
        """
        Fetch and process a web page.
        
        Args:
            url: URL of the web page to process.
            
        Returns:
            Document object with extracted text content.
        """
        logger.info(f"Fetching URL: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Research Bot) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = "\n".join(lines)

        title = soup.title.string if soup.title else url

        return Document(
            content=cleaned_text,
            source=url,
            doc_type="web",
            metadata={
                "title": title,
                "url": url,
                "content_length": len(cleaned_text),
            },
        )

    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Process multiple URLs.
        
        Args:
            urls: List of URLs to process.
            
        Returns:
            List of Document objects.
        """
        documents = []
        for url in urls:
            try:
                doc = self.process_url(url)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
        return documents

    def _process_pdf(self, path: Path) -> Document:
        """Extract text from a PDF file."""
        reader = PdfReader(str(path))
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(text)

        full_text = "\n\n".join(pages_text)

        return Document(
            content=full_text,
            source=str(path),
            doc_type="pdf",
            metadata={
                "filename": path.name,
                "num_pages": len(reader.pages),
                "content_length": len(full_text),
            },
        )

    def _process_text(self, path: Path) -> Document:
        """Extract text from a plain text or markdown file."""
        content = path.read_text(encoding="utf-8")

        return Document(
            content=content,
            source=str(path),
            doc_type="text",
            metadata={
                "filename": path.name,
                "content_length": len(content),
            },
        )

    def _process_html_file(self, path: Path) -> Document:
        """Extract text from a local HTML file."""
        content = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(content, "html.parser")

        for element in soup(["script", "style"]):
            element.decompose()

        text = soup.get_text(separator="\n", strip=True)

        return Document(
            content=text,
            source=str(path),
            doc_type="html",
            metadata={
                "filename": path.name,
                "content_length": len(text),
            },
        )

    def process_text_input(self, text: str, source_name: str = "direct_input") -> Document:
        """
        Process raw text input directly.
        
        Args:
            text: Raw text content.
            source_name: Name to identify this text source.
            
        Returns:
            Document object.
        """
        return Document(
            content=text,
            source=source_name,
            doc_type="text",
            metadata={
                "content_length": len(text),
                "source_type": "direct_input",
            },
        )
