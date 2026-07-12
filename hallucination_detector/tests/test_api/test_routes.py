"""
Tests for the FastAPI endpoints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:

    def test_health_check(self, client):
        """Test health endpoint returns valid response."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "index_size" in data
        assert "message" in data


class TestUploadEndpoints:

    def test_upload_text(self, client):
        """Test text upload endpoint."""
        payload = {
            "text": "The Earth revolves around the Sun. This is a fundamental fact of astronomy.",
            "source_name": "test_source",
        }
        response = client.post("/api/upload/text", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "num_chunks" in data
        assert data["num_chunks"] > 0

    def test_upload_empty_text(self, client):
        """Test that empty text upload is rejected."""
        payload = {"text": "", "source_name": "test"}
        response = client.post("/api/upload/text", json=payload)
        assert response.status_code == 422  # Validation error


class TestQueryEndpoint:

    def test_query_without_documents(self, client):
        """Test query returns appropriate error when no documents exist."""
        payload = {"query": "What is AI?", "llm_response": ""}
        response = client.post("/api/query", json=payload)
        # May return 400 if no docs, or 200 with empty results
        assert response.status_code in (200, 400)

    def test_query_validation(self, client):
        """Test query validation rejects empty queries."""
        payload = {"query": "", "llm_response": ""}
        response = client.post("/api/query", json=payload)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
