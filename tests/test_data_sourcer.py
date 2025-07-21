"""
Test data sourcing functionality
"""
import pytest
import pandas as pd
import requests
from unittest.mock import Mock, patch, AsyncMock
from app.services.data_sourcer import DataSourcer


class TestDataSourcer:
    """Test data sourcing functionality"""
    
    def test_init(self):
        """Test DataSourcer initialization"""
        sourcer = DataSourcer()
        assert sourcer is not None
    
    @pytest.mark.asyncio
    async def test_load_from_url_csv(self):
        """Test loading CSV from URL"""
        sourcer = DataSourcer()
        
        # Mock pandas read_csv since the implementation uses pd.read_csv(url) directly for CSV
        with patch('pandas.read_csv') as mock_read_csv, \
             patch('requests.get') as mock_get:
            
            # Mock requests.get for initial request
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'text/csv'}
            mock_get.return_value = mock_response
            
            # Mock pandas.read_csv to return a DataFrame
            mock_df = pd.DataFrame({
                'name': ['John', 'Jane'],
                'age': [25, 30]
            })
            mock_read_csv.return_value = mock_df
            
            result = await sourcer.load_from_url("https://example.com/data.csv")
            
            # Check actual return structure: "dataframe" and "metadata"
            assert "dataframe" in result
            assert "metadata" in result
            assert result["metadata"]["shape"] == (2, 2)
            assert "name" in result["metadata"]["columns"]
            assert "age" in result["metadata"]["columns"]
    
    @pytest.mark.asyncio
    async def test_load_from_url_failure(self):
        """Test handling of URL loading failures"""
        sourcer = DataSourcer()
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            with pytest.raises(Exception):
                await sourcer.load_from_url("https://invalid-url.com")
    
    @pytest.mark.asyncio
    async def test_scrape_web_data(self):
        """Test web scraping functionality"""
        sourcer = DataSourcer()
        
        parsed_request = {
            'data_source': 'https://en.wikipedia.org/wiki/Test',
            'questions': ['What are the main topics?']
        }
        
        # Mock web scraping - the implementation uses pd.read_html for HTML content
        with patch('pandas.read_html') as mock_read_html, \
             patch('requests.get') as mock_get:
            
            # Mock requests.get for initial request
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_get.return_value = mock_response
            
            # Mock pandas.read_html to return list of DataFrames
            mock_df = pd.DataFrame({
                'Topic': ['Topic 1', 'Topic 2', 'Topic 3'],
                'Description': ['Desc 1', 'Desc 2', 'Desc 3'],
                'Count': [10, 20, 15]
            })
            mock_read_html.return_value = [mock_df]  # pd.read_html returns a list
            
            result = await sourcer.scrape_web_data(parsed_request)
            
            # Check actual return structure
            assert "dataframe" in result
            assert "metadata" in result
            assert result["metadata"]["shape"] == (3, 3)
            assert "Topic" in result["metadata"]["columns"]
    
    @pytest.mark.asyncio
    async def test_load_from_url_json(self):
        """Test loading JSON from URL"""
        sourcer = DataSourcer()
        
        with patch('requests.get') as mock_get:
            # Mock JSON response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'application/json'}
            mock_response.json.return_value = [
                {"name": "Alice", "age": 25, "city": "NYC"},
                {"name": "Bob", "age": 30, "city": "LA"}
            ]
            mock_get.return_value = mock_response
            
            result = await sourcer.load_from_url("https://example.com/data.json")
            
            assert "dataframe" in result
            assert "metadata" in result
            assert result["metadata"]["shape"] == (2, 3)
            assert "name" in result["metadata"]["columns"]
    
    @pytest.mark.asyncio
    async def test_load_from_url_invalid_url(self):
        """Test handling of invalid URLs"""
        sourcer = DataSourcer()
        
        with pytest.raises(ValueError, match="Invalid URL format"):
            await sourcer.load_from_url("not-a-valid-url")
    
    @pytest.mark.asyncio
    async def test_load_from_url_http_error(self):
        """Test handling of HTTP errors"""
        sourcer = DataSourcer()
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.RequestException("Connection failed")
            
            with pytest.raises(ValueError, match="Error loading data from URL"):
                await sourcer.load_from_url("https://example.com/data.csv")
    
    @pytest.mark.asyncio
    async def test_scrape_web_data_no_source(self):
        """Test web scraping with no data source"""
        sourcer = DataSourcer()
        
        parsed_request = {
            'questions': ['What are the main topics?']
            # Missing 'data_source'
        }
        
        with pytest.raises(ValueError, match="No data source URL provided"):
            await sourcer.scrape_web_data(parsed_request)
    
    def test_create_data_info(self):
        """Test data info creation"""
        sourcer = DataSourcer()
        
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', None],
            'age': [25, 30, 35],
            'salary': [50000, None, 70000]
        })
        
        result = sourcer._create_data_info(df, "test_data")
        
        assert "dataframe" in result
        assert "metadata" in result
        
        metadata = result["metadata"]
        assert metadata["source_name"] == "test_data"
        assert metadata["shape"] == (3, 3)
        assert set(metadata["columns"]) == {"name", "age", "salary"}
        assert metadata["missing_values"]["name"] == 1
        assert metadata["missing_values"]["salary"] == 1
        assert len(metadata["numeric_columns"]) == 2  # age, salary
        assert "name" in metadata["categorical_columns"]
