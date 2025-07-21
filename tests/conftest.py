"""
Shared pytest fixtures for Data Analyst Agent tests
"""
import pytest
import io
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.api.routes import router


@pytest.fixture
def app():
    """Create a FastAPI app for testing"""
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return app


@pytest.fixture
def client(app):
    """Create a test client for FastAPI"""
    return TestClient(app)


@pytest.fixture
def sample_question_file():
    """Create a sample question file for testing"""
    content = """
    Analyze the sales data from the following source:
    https://example.com/sales-data.csv
    
    1. What are the top 5 products by revenue?
    2. Show monthly sales trends
    3. Create a visualization showing product performance
    
    Respond with a JSON array of strings.
    """
    return io.BytesIO(content.encode('utf-8'))


@pytest.fixture
def complex_question_file():
    """Sample with multiple requirements"""
    content = """
    Scrape data from: https://en.wikipedia.org/wiki/List_of_countries_by_GDP
    
    1. Which are the top 10 countries by GDP?
    2. What's the average GDP of European countries?
    3. Create a bar chart showing top 10 countries
       Return as base64 encoded image under 100,000 bytes
    """
    return io.BytesIO(content.encode('utf-8'))


@pytest.fixture
def simple_question_file():
    """Simple question file without URLs"""
    content = """
    Simple analysis question without URLs.
    What insights can you provide from the data?
    """
    return io.BytesIO(content.encode('utf-8')) 