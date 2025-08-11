"""
Data Scraping Module
Takes data source URLs and creates DataFrames
"""
import pandas as pd
import requests
from typing import List, Dict, Any, Optional
import time


class DataScraper:
    """
    Simple data scraper that takes URLs and creates DataFrames
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def scrape_sources(self, data_sources: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape multiple data sources and return DataFrames info
        
        Args:
            data_sources: List of URLs to scrape
            
        Returns:
            List of dictionaries with DataFrame info for each source
        """
        results = []
        
        print(f"🔍 Starting to scrape {len(data_sources)} data sources...")
        
        for i, url in enumerate(data_sources):
            print(f"\n📡 Scraping source {i+1}/{len(data_sources)}: {url}")
            
            try:
                start_time = time.time()
                
                # Scrape the URL
                df = self._scrape_single_url(url)
                
                scrape_time = time.time() - start_time
                
                if df is not None and not df.empty:
                    # Create summary
                    result = {
                        "source_url": url,
                        "source_index": i,
                        "success": True,
                        "shape": [int(df.shape[0]), int(df.shape[1])],
                        "columns": df.columns.tolist(),
                        "sample_data": df.head(3).to_dict('records'),
                        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "memory_usage_mb": round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 3),
                        "scrape_time_seconds": round(scrape_time, 2),
                        "dataframe": df  # Include actual DataFrame for further processing
                    }
                    
                    print(f"✅ Success: {df.shape[0]} rows × {df.shape[1]} columns")
                    print(f"📊 Columns: {df.columns.tolist()[:5]}{'...' if len(df.columns) > 5 else ''}")
                    
                else:
                    result = {
                        "source_url": url,
                        "source_index": i,
                        "success": False,
                        "error": "No data found or empty DataFrame",
                        "shape": [0, 0],
                        "scrape_time_seconds": round(scrape_time, 2)
                    }
                    print(f"❌ No data found")
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ Error scraping {url}: {e}")
                results.append({
                    "source_url": url,
                    "source_index": i,
                    "success": False,
                    "error": str(e),
                    "shape": [0, 0]
                })
        
        return results
    
    def _scrape_single_url(self, url: str) -> Optional[pd.DataFrame]:
        """
        Scrape a single URL and return DataFrame
        """
        try:
            # Determine scraping method based on URL
            if 'wikipedia.org' in url or '.html' in url:
                return self._scrape_html_tables(url)
            elif '.csv' in url:
                return self._scrape_csv(url)
            elif '.json' in url:
                return self._scrape_json(url)
            elif 's3' in url:
                return 
            else:
                # Default: try HTML tables first, then CSV
                df = self._scrape_html_tables(url)
                if df is None or df.empty:
                    df = self._scrape_csv(url)
                return df
                
        except Exception as e:
            print(f"⚠️  Scraping error: {e}")
            return None


    def _scrape_files_from_s3(self,url : str) -> Optional[pd.DataFrame] :
        """
        Scrape files from s3 path 
        """        
    
    def _scrape_html_tables(self, url: str) -> Optional[pd.DataFrame]:
        """
        Scrape HTML tables from URL (especially Wikipedia)
        """
        try:
            print(f"🌐 Reading HTML tables from: {url}")
            
            # Read all tables
            tables = pd.read_html(url)
            print(f"📋 Found {len(tables)} tables")
            
            if not tables:
                return None
            
            # For Wikipedia, find the best table
            if 'wikipedia.org' in url:
                best_table = self._find_best_wikipedia_table(tables, url)
            else:
                # Just take the largest table
                best_table = max(tables, key=len)
            
            if best_table is not None and not best_table.empty:
                # Clean column names
                best_table.columns = [str(col).strip() for col in best_table.columns]
                print(f"📊 Selected table: {best_table.shape[0]} rows × {best_table.shape[1]} columns")
                return best_table
            
            return None
            
        except Exception as e:
            print(f"❌ HTML table scraping failed: {e}")
            return None
    
    def _find_best_wikipedia_table(self, tables: List[pd.DataFrame], url: str) -> Optional[pd.DataFrame]:
        """
        Find the best table from Wikipedia page based on content and structure
        """
        if not tables:
            return None
        
        # Look for tables with proper movie/film data
        for i, table in enumerate(tables):
            if table.empty:
                continue
                
            columns = [str(col).lower() for col in table.columns]
            
            # Check if this looks like a movie data table
            movie_keywords = ['rank', 'title', 'gross', 'year', 'peak', 'film', 'movie']
            keyword_matches = sum(1 for keyword in movie_keywords if any(keyword in col for col in columns))
            
            # Also check the content
            has_good_data = (
                table.shape[0] > 10 and  # Has reasonable number of rows
                table.shape[1] >= 4 and  # Has multiple columns
                keyword_matches >= 2     # Has movie-related column names
            )
            
            if has_good_data:
                print(f"🎯 Found good movie table at index {i}: {table.shape[0]} rows, {keyword_matches} keyword matches")
                return table
        
        # Fallback: return the largest table
        print("📋 No perfect match, using largest table")
        return max(tables, key=len)
    
    def _scrape_csv(self, url: str) -> Optional[pd.DataFrame]:
        """
        Scrape CSV data from URL
        """
        try:
            print(f"📄 Reading CSV from: {url}")
            df = pd.read_csv(url)
            return df
        except Exception as e:
            print(f"❌ CSV scraping failed: {e}")
            return None
    
    def _scrape_json(self, url: str) -> Optional[pd.DataFrame]:
        """
        Scrape JSON data from URL
        """
        try:
            print(f"🔗 Reading JSON from: {url}")
            response = self.session.get(url, timeout=30)
            data = response.json()
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                return None
                
            return df
        except Exception as e:
            print(f"❌ JSON scraping failed: {e}")
            return None


# Simple function interface
def scrape_data_sources(data_sources: List[str]) -> List[Dict[str, Any]]:
    """
    Simple function to scrape data sources
    
    Args:
        data_sources: List of URLs to scrape
        
    Returns:
        List of dictionaries with scraping results
    """
    scraper = DataScraper()
    return scraper.scrape_sources(data_sources)


# Test function
def test_scraping():
    """
    Test the scraping with Wikipedia URL
    """
    test_sources = [
        "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    ]
    
    print("🧪 Testing scraping...")
    results = scrape_data_sources(test_sources)
    
    for result in results:
        print(f"\n📊 Result for {result['source_url']}:")
        print(f"   Success: {result['success']}")
        if result['success']:
            print(f"   Shape: {result['shape']}")
            print(f"   Columns: {result['columns'][:5]}{'...' if len(result['columns']) > 5 else ''}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    test_scraping() 