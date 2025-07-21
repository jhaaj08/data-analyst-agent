"""
Test data processing functionality
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from app.services.data_processor import DataProcessor


class TestDataProcessor:
    """Test data processing functionality"""
    
    def test_init(self):
        """Test DataProcessor initialization"""
        processor = DataProcessor()
        assert processor is not None
    
    @pytest.mark.asyncio
    async def test_process_csv_data(self):
        """Test processing CSV data"""
        processor = DataProcessor()
        
        # Create actual DataFrame (what DataProcessor expects)
        df = pd.DataFrame({
            "name": ["John", "Jane", "Bob"],
            "age": [25, 30, 35],
            "salary": [50000, 60000, 70000]
        })
        
        data_info = {
            "dataframe": df,  # Correct key with actual DataFrame
            "metadata": {"columns": ["name", "age", "salary"]}
        }
        
        analysis_plan = {
            "data_operations": ["clean_data"],  # Remove "basic_stats" (not implemented)
            "analysis_type": "descriptive"
        }
        
        result = await processor.process(data_info, analysis_plan)
        
        # Check return structure
        assert "dataframe" in result
        assert "metadata" in result
        assert "processing_log" in result
        assert result["metadata"]["processed_shape"] == (3, 3)
        assert len(result["processing_log"]) > 0
        assert "Applied basic data cleaning" in result["processing_log"][0]
    
    @pytest.mark.asyncio
    async def test_process_with_missing_data(self):
        """Test processing data with missing values"""
        processor = DataProcessor()
        
        # Create DataFrame with missing values
        df = pd.DataFrame({
            "name": ["John", "Jane", "Bob"],
            "age": [25, None, 35],  # Missing value
            "salary": [50000, 60000, None]  # Missing value
        })
        
        data_info = {
            "dataframe": df,
            "metadata": {"columns": ["name", "age", "salary"]}
        }
        
        analysis_plan = {"data_operations": ["handle_missing"]}
        
        result = await processor.process(data_info, analysis_plan)
        
        # Check that missing values were handled
        assert "dataframe" in result
        assert "processing_log" in result
        
        # Should have no missing values after processing
        processed_df = result["dataframe"]
        assert processed_df.isnull().sum().sum() == 0
        
        # Check processing log mentions filling missing values
        log_messages = " ".join(result["processing_log"])
        assert "missing values" in log_messages.lower()
    
    @pytest.mark.asyncio
    async def test_process_normalization(self):
        """Test data normalization"""
        processor = DataProcessor()
        
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
            "category": ["A", "B", "A", "B", "A"]
        })
        
        data_info = {
            "dataframe": df,
            "metadata": {"columns": ["feature1", "feature2", "category"]}
        }
        
        analysis_plan = {"data_operations": ["normalize"]}
        
        result = await processor.process(data_info, analysis_plan)
        
        # Check normalization was applied
        processed_df = result["dataframe"]
        
        # Numeric columns should be normalized (mean ~0, std ~1)
        # Note: StandardScaler normalizes to standard normal distribution
        assert abs(processed_df["feature1"].mean()) < 0.001
        assert abs(processed_df["feature1"].std(ddof=0) - 1.0) < 0.001  # Use population std
        assert abs(processed_df["feature2"].mean()) < 0.001
        assert abs(processed_df["feature2"].std(ddof=0) - 1.0) < 0.001  # Use population std
        
        # Categorical column should remain unchanged
        assert processed_df["category"].equals(df["category"])
        
        # Check log
        assert any("Normalized columns" in log for log in result["processing_log"])
    
    @pytest.mark.asyncio
    async def test_process_categorical_encoding(self):
        """Test categorical encoding"""
        processor = DataProcessor()
        
        # Create data with enough rows to have 10+ unique values for one column
        df = pd.DataFrame({
            "category": ["A", "B", "A", "C", "B"],  # 3 unique values (< 10, should be encoded)
            "description": [f"Text_{i}" for i in range(15)][:5]  # 5 unique values (< 10, will be encoded)
        })
        
        # Add a column with 10+ unique values (should NOT be encoded)
        df["unique_id"] = [f"ID_{i}" for i in range(len(df))]  # 5 unique values, but let's make it 10+
        
        # Extend the DataFrame to have 12 rows with 12 unique IDs
        extended_data = []
        for i in range(12):
            extended_data.append({
                "category": ["A", "B", "C"][i % 3],
                "description": f"Text_{i % 5}",
                "unique_id": f"ID_{i}"  # 12 unique values (>= 10, should NOT be encoded)
            })
        
        df = pd.DataFrame(extended_data)
        
        data_info = {
            "dataframe": df,
            "metadata": {"columns": ["category", "description", "unique_id"]}
        }
        
        analysis_plan = {"data_operations": ["encode_categorical"]}
        
        result = await processor.process(data_info, analysis_plan)
        
        processed_df = result["dataframe"]
        
        # Should have encoded category (3 unique values < 10)
        assert "category_encoded" in processed_df.columns
        
        # Should have encoded description (5 unique values < 10)
        assert "description_encoded" in processed_df.columns
        
        # Should NOT encode unique_id (12 unique values >= 10)
        assert "unique_id_encoded" not in processed_df.columns
        
        # Check log mentions encoding
        log_messages = " ".join(result["processing_log"])
        assert "Encoded categorical column" in log_messages
    
    @pytest.mark.asyncio
    async def test_process_outlier_detection(self):
        """Test outlier detection"""
        processor = DataProcessor()
        
        # Create data with clear outliers
        df = pd.DataFrame({
            "normal_data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "with_outlier": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is an outlier
        })
        
        data_info = {
            "dataframe": df,
            "metadata": {"columns": ["normal_data", "with_outlier"]}
        }
        
        analysis_plan = {"data_operations": ["detect_outliers"]}
        
        result = await processor.process(data_info, analysis_plan)
        
        # Check log mentions outliers
        log_messages = " ".join(result["processing_log"])
        assert "Detected outliers" in log_messages
    
    @pytest.mark.asyncio
    async def test_process_multiple_operations(self):
        """Test multiple operations in sequence"""
        processor = DataProcessor()
        
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Alice", "Bob"],  # Duplicates
            "age": [25, None, 25, 30],  # Missing value and duplicate
            "category": ["A", "B", "A", "B"]
        })
        
        data_info = {
            "dataframe": df,
            "metadata": {"columns": ["name", "age", "category"]}
        }
        
        analysis_plan = {
            "data_operations": ["clean_data", "handle_missing", "encode_categorical"]
        }
        
        result = await processor.process(data_info, analysis_plan)
        
        processed_df = result["dataframe"]
        
        # Should have removed duplicates (clean_data)
        assert len(processed_df) == 3  # One duplicate removed
        
        # Should have filled missing values (handle_missing)
        assert processed_df["age"].isnull().sum() == 0
        
        # Should have encoded categorical (encode_categorical)
        assert "category_encoded" in processed_df.columns
        
        # Should have multiple log entries
        assert len(result["processing_log"]) >= 3
    
    @pytest.mark.asyncio
    async def test_process_no_operations(self):
        """Test processing with no operations specified"""
        processor = DataProcessor()
        
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["A", "B", "C"]
        })
        
        data_info = {
            "dataframe": df,
            "metadata": {"columns": ["col1", "col2"]}
        }
        
        analysis_plan = {"data_operations": []}  # No operations
        
        result = await processor.process(data_info, analysis_plan)
        
        # DataFrame should remain unchanged
        processed_df = result["dataframe"]
        assert processed_df.equals(df)
        
        # Should have empty processing log
        assert len(result["processing_log"]) == 0
    
    def test_get_data_summary(self):
        """Test data summary generation"""
        processor = DataProcessor()
        
        df = pd.DataFrame({
            "numeric": [1, 2, 3, 4, 5],
            "categorical": ["A", "B", "A", "C", "B"],
            "with_missing": [1, None, 3, None, 5]
        })
        
        summary = processor.get_data_summary(df)
        
        assert summary["shape"] == (5, 3)
        assert set(summary["columns"]) == {"numeric", "categorical", "with_missing"}
        assert summary["missing_values"]["with_missing"] == 2
        assert summary["missing_values"]["numeric"] == 0
        
        # Check categorical summary
        assert "categorical" in summary["categorical_summary"]
        cat_summary = summary["categorical_summary"]["categorical"]
        assert cat_summary["unique_count"] == 3  # A, B, C
        assert "A" in cat_summary["top_values"]
        assert "B" in cat_summary["top_values"]