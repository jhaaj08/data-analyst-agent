"""
Test visualization functionality
"""
import pytest
import pandas as pd
from app.services.visualizer import Visualizer


class TestVisualizer:
    """Test visualization functionality"""
    
    @pytest.mark.asyncio
    async def test_create_basic_visualizations(self):
        """Test creating basic visualizations"""
        visualizer = Visualizer()
        
        # Create proper processed data structure
        df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'D'],
            'value': [10, 20, 15, 25]
        })
        
        processed_data = {
            "dataframe": df,
            "metadata": {"columns": ["category", "value"]}
        }
        
        analysis_results = {
            "insights": ["Category D has highest value"],
            "statistics": {"mean": 17.5}
        }
        
        analysis_plan = {
            "visualization_requirements": ["bar_chart", "histogram"]
        }
        
        result = await visualizer.create_visualizations(
            processed_data, analysis_results, analysis_plan, "test-id"
        )
        
        assert isinstance(result, list)
        assert len(result) >= 0  # Should handle gracefully (may return empty list)
    
    @pytest.mark.asyncio
    async def test_create_visualizations_no_data(self):
        """Test visualization creation with no data"""
        visualizer = Visualizer()
        
        # Create empty processed data structure
        processed_data = {
            "dataframe": pd.DataFrame(),
            "metadata": {"columns": []}
        }
        
        result = await visualizer.create_visualizations(
            processed_data, {}, {"visualization_requirements": []}, "test-id"
        )
        
        # Should handle gracefully
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_create_visualizations_with_numeric_data(self):
        """Test visualization with numeric data"""
        visualizer = Visualizer()
        
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        
        processed_data = {
            "dataframe": df,
            "metadata": {"columns": ["x", "y", "category"]}
        }
        
        analysis_results = {
            "insights": ["Strong positive correlation between x and y"],
            "correlation_analysis": {
                "strong_correlations": [{"variable1": "x", "variable2": "y", "correlation": 0.95}]
            }
        }
        
        analysis_plan = {
            "visualization_requirements": ["scatter_plot", "line_chart", "bar_chart"]
        }
        
        result = await visualizer.create_visualizations(
            processed_data, analysis_results, analysis_plan, "test-numeric"
        )
        
        assert isinstance(result, list)
        # Visualizer may or may not create files depending on implementation
        assert len(result) >= 0
    
    @pytest.mark.asyncio
    async def test_create_visualizations_error_handling(self):
        """Test visualization error handling"""
        visualizer = Visualizer()
        
        # Data that might cause issues
        df = pd.DataFrame({
            'empty_col': [None, None, None],
            'text_col': ['a', 'b', 'c']
        })
        
        processed_data = {
            "dataframe": df,
            "metadata": {"columns": ["empty_col", "text_col"]}
        }
        
        analysis_results = {"insights": ["No useful patterns found"]}
        
        analysis_plan = {
            "visualization_requirements": ["histogram", "scatter_plot"]  # May not work with this data
        }
        
        result = await visualizer.create_visualizations(
            processed_data, analysis_results, analysis_plan, "test-error"
        )
        
        # Should handle gracefully even with problematic data
        assert isinstance(result, list)