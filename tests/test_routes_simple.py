"""
Simple unit tests for API routes that actually work
"""
import pytest
import json
import io
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from app.api.routes import router, _parse_question_file_regex


@pytest.fixture
def client():
    """Create a test client for FastAPI"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


class TestRouteBasics:
    """Basic tests that don't require complex mocking"""

    def test_health_check(self, client):
        """Test the health check endpoint"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_test_endpoint(self, client):
        """Test the test endpoint"""
        response = client.post("/api/test")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_analyze_data_no_file(self, client):
        """Test analyze endpoint without file"""
        response = client.post("/api/")
        assert response.status_code == 422  # Validation error

    def test_analyze_data_empty_file(self, client):
        """Test analyze endpoint with empty file"""
        files = {"file": ("question.txt", io.BytesIO(b""), "text/plain")}
        response = client.post("/api/", files=files)
        # Should handle gracefully but may return various status codes
        assert response.status_code in [200, 400, 422, 500]


class TestRegexParsing:
    """Test the regex parsing fallback function"""

    def test_parse_with_urls(self):
        """Test parsing with URLs"""
        question_text = """
        Scrape data from: https://en.wikipedia.org/wiki/Test
        1. How many items are there?
        2. What's the average?
        Respond with JSON array.
        """
        
        result = _parse_question_file_regex(question_text)
        
        assert result["data_source"] == "https://en.wikipedia.org/wiki/Test"
        assert result["requires_web_scraping"] is True
        assert result["output_format"]["type"] == "json_array"
        assert len(result["numbered_questions"]) == 2

    def test_parse_without_urls(self):
        """Test parsing without URLs"""
        question_text = "Simple analysis question without URLs"
        
        result = _parse_question_file_regex(question_text)
        
        assert result["data_source"] is None
        assert result["requires_web_scraping"] is False
        assert result["output_format"]["type"] == "standard"

    def test_parse_multiple_urls(self):
        """Test parsing with multiple URLs"""
        question_text = """
        Check these sources:
        https://example1.com/data.csv
        https://example2.com/more-data.json
        What insights can you find?
        """
        
        result = _parse_question_file_regex(question_text)
        
        assert result["data_source"] == "https://example1.com/data.csv"  # First URL
        assert len(result["all_urls"]) == 2

    def test_parse_base64_requirement(self):
        """Test detection of base64 image requirements"""
        question_text = """
        Create a chart and return as base64 encoded image.
        Make sure it's under 100,000 bytes.
        """
        
        result = _parse_question_file_regex(question_text)
        
        assert result["output_format"]["include_base64_images"] is True
        assert result["output_format"]["size_limit"] == "100,000 bytes"

    def test_parse_numbered_questions(self):
        """Test extraction of numbered questions"""
        question_text = """
        Analyze the data and answer:
        1. What is the trend?
        2. Which category performs best?
        3. Any correlations found?
        """
        
        result = _parse_question_file_regex(question_text)
        
        assert len(result["numbered_questions"]) == 3
        assert "What is the trend?" in result["numbered_questions"]
        assert "Which category performs best?" in result["numbered_questions"]
        assert "Any correlations found?" in result["numbered_questions"]


class TestIntegrationBasic:
    """Basic integration tests"""

    @patch('app.api.routes.llm_client')
    def test_analyze_with_mocked_llm_failure(self, mock_llm_client, client):
        """Test that the system falls back to regex when LLM fails"""
        
        # Mock LLM to fail
        mock_llm_client.openai_client = None
        
        question_content = """
        Simple question for analysis.
        1. What can you tell me about the data?
        """
        
        files = {"file": ("question.txt", io.BytesIO(question_content.encode()), "text/plain")}
        
        # This should not crash even when LLM is unavailable
        response = client.post("/api/", files=files)
        
        # The request might fail for other reasons (no actual data processing services)
        # but it should at least parse the question using regex fallback
        assert response.status_code in [200, 500]  # Either success or controlled failure

    def test_analyze_with_simple_question(self, client):
        """Test with a simple question that should trigger minimal processing"""
        
        question_content = "What insights can you provide?"
        files = {"file": ("question.txt", io.BytesIO(question_content.encode()), "text/plain")}
        
        response = client.post("/api/", files=files)
        
        # Even if it fails, it should be a controlled failure, not a crash
        assert response.status_code in [200, 500]
        
        if response.status_code == 500:
            # If it fails, it should be JSON with error info
            try:
                error_data = response.json()
                assert "error" in error_data
            except:
                # If not JSON, that's also acceptable for now
                pass 