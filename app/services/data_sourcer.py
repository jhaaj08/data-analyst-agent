"""
Data sourcing service for loading data from various sources
"""
import pandas as pd
import requests
import json
import os
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from ..core.config import settings


class DataSourcer:
    """Service for sourcing data from files and external sources"""
    
    async def load_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load data from a file
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Dictionary containing data and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext == '.json':
                df = pd.read_json(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_ext == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_ext == '.txt':
                # Try to read as CSV with different delimiters
                try:
                    df = pd.read_csv(file_path, delimiter='\t')
                except:
                    df = pd.read_csv(file_path, delimiter='|')
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            return self._create_data_info(df, os.path.basename(file_path))
            
        except Exception as e:
            raise ValueError(f"Error loading file {file_path}: {str(e)}")
    
    async def load_from_url(self, url: str) -> Dict[str, Any]:
        """
        Load data from a URL
        
        Args:
            url: URL to fetch data from
            
        Returns:
            Dictionary containing data and metadata
        """
        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid URL format")
            
            # Fetch data
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Determine data format from content type or URL
            content_type = response.headers.get('content-type', '').lower()
            
            if 'json' in content_type or url.endswith('.json'):
                data = response.json()
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.json_normalize(data)
            elif 'csv' in content_type or url.endswith('.csv'):
                df = pd.read_csv(url)
            elif 'html' in content_type or 'wikipedia.org' in url:
                # Handle HTML table scraping (especially Wikipedia)
                try:
                    # pandas can read HTML tables directly
                    tables = pd.read_html(url)
                    
                    # For Wikipedia, get the largest table (usually the main data table)
                    df = max(tables, key=len) if tables else pd.DataFrame()
                    
                    # Clean column names
                    if not df.empty:
                        df.columns = df.columns.astype(str)
                        
                except Exception as e:
                    print(f"Error reading HTML tables: {e}")
                    # Fallback to empty DataFrame
                    df = pd.DataFrame()
            else:
                # Try to parse as JSON first, then CSV, then HTML tables
                try:
                    data = response.json()
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.json_normalize(data)
                except:
                    try:
                        df = pd.read_csv(url)
                    except:
                        # Last resort: try HTML tables
                        try:
                            tables = pd.read_html(url)
                            df = max(tables, key=len) if tables else pd.DataFrame()
                        except:
                            raise ValueError(f"Could not parse data from URL: {url}")
            
            return self._create_data_info(df, f"url_data_{url.split('/')[-1]}")
            
        except Exception as e:
            raise ValueError(f"Error loading data from URL {url}: {str(e)}")
    
    def _create_data_info(self, df: pd.DataFrame, source_name: str) -> Dict[str, Any]:
        """
        Create data information dictionary
        
        Args:
            df: Pandas DataFrame
            source_name: Name of the data source
            
        Returns:
            Data information dictionary
        """
        return {
            "dataframe": df,
            "metadata": {
                "source_name": source_name,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "datetime_columns": df.select_dtypes(include=['datetime']).columns.tolist()
            }
        }
    
    async def scrape_web_data(self, parsed_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle web scraping based on parsed request
        
        Args:
            parsed_request: Parsed request containing URLs and requirements
            
        Returns:
            Dictionary containing scraped data and metadata
        """
        data_source = parsed_request.get('data_source')
        if not data_source:
            raise ValueError("No data source URL provided for web scraping")
        
        # Use the existing load_from_url method which now handles HTML tables
        return await self.load_from_url(data_source)
    
    async def get_sample_data(self, data_info: Dict[str, Any], n_rows: int = 5) -> Dict[str, Any]:
        """
        Get a sample of the data for preview
        
        Args:
            data_info: Data information dictionary
            n_rows: Number of rows to sample
            
        Returns:
            Sample data information
        """
        df = data_info["dataframe"]
        sample_df = df.head(n_rows)
        
        return {
            "sample_data": sample_df.to_dict('records'),
            "sample_size": len(sample_df),
            "total_size": len(df)
        } 