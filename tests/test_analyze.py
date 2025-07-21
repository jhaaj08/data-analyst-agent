"""
Test analysis functionality
"""
import pytest
import pandas as pd
import numpy as np
from app.services.analyzer import Analyzer


class TestAnalyzer:
    """Test analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_analyze_descriptive(self):
        """Test descriptive analysis"""
        analyzer = Analyzer()
        
        # Create proper processed data structure (matches what DataProcessor returns)
        df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000]
        })
        
        processed_data = {
            "dataframe": df,
            "metadata": {"columns": ["age", "salary"]}
        }
        
        analysis_plan = {
            "analysis_type": "descriptive",
            "analysis_steps": ["summary_statistics", "correlation_analysis"]
        }
        
        result = await analyzer.analyze(processed_data, analysis_plan)
        
        # Check actual return structure
        assert "insights" in result
        assert "data_summary" in result
        assert "summary_statistics" in result
        assert "correlation_analysis" in result
        assert result["analysis_type"] == "descriptive"
        
        # Check data summary
        assert result["data_summary"]["total_rows"] == 5
        assert result["data_summary"]["total_columns"] == 2
        assert result["data_summary"]["numeric_columns"] == 2
        
        # Check summary statistics
        assert "descriptive_stats" in result["summary_statistics"]
        assert "age" in result["summary_statistics"]["descriptive_stats"]
        assert "salary" in result["summary_statistics"]["descriptive_stats"]
    
    @pytest.mark.asyncio
    async def test_analyze_with_no_data(self):
        """Test analysis when no data is provided"""
        analyzer = Analyzer()
        
        # Create empty processed data structure
        processed_data = {
            "dataframe": pd.DataFrame(),  # Empty DataFrame
            "metadata": {"columns": []}
        }
        
        analysis_plan = {"analysis_type": "descriptive"}
        
        result = await analyzer.analyze(processed_data, analysis_plan)
        
        # Should handle gracefully
        assert result is not None
        assert "data_summary" in result
        assert result["data_summary"]["total_rows"] == 0
        assert result["data_summary"]["total_columns"] == 0
    
    @pytest.mark.asyncio
    async def test_analyze_correlation_analysis(self):
        """Test correlation analysis"""
        analyzer = Analyzer()
        
        # Create data with clear correlation
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'y': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],  # Perfect correlation with x
            'z': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]  # Negative correlation
        })
        
        processed_data = {
            "dataframe": df,
            "metadata": {"columns": ["x", "y", "z"]}
        }
        
        analysis_plan = {
            "analysis_type": "descriptive",
            "analysis_steps": ["correlation_analysis"]
        }
        
        result = await analyzer.analyze(processed_data, analysis_plan)
        
        assert "correlation_analysis" in result
        assert "correlation_matrix" in result["correlation_analysis"]
        assert "strong_correlations" in result["correlation_analysis"]
        
        # Should find strong correlations (>0.7)
        strong_corrs = result["correlation_analysis"]["strong_correlations"]
        assert len(strong_corrs) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_distribution_analysis(self):
        """Test distribution analysis"""
        analyzer = Analyzer()
        
        # Create normally distributed data
        np.random.seed(42)
        df = pd.DataFrame({
            'normal_data': np.random.normal(100, 15, 1000),
            'uniform_data': np.random.uniform(0, 100, 1000)
        })
        
        processed_data = {
            "dataframe": df,
            "metadata": {"columns": ["normal_data", "uniform_data"]}
        }
        
        analysis_plan = {
            "analysis_type": "descriptive",
            "analysis_steps": ["distribution_analysis"]
        }
        
        result = await analyzer.analyze(processed_data, analysis_plan)
        
        assert "distribution_analysis" in result
        assert "normal_data" in result["distribution_analysis"]
        assert "uniform_data" in result["distribution_analysis"]
        
        # Check distribution statistics
        normal_stats = result["distribution_analysis"]["normal_data"]
        assert "is_normal" in normal_stats
        assert "normality_p_value" in normal_stats
        assert "mean" in normal_stats
        assert "std" in normal_stats
    
    @pytest.mark.asyncio
    async def test_analyze_regression_analysis(self):
        """Test regression analysis"""
        analyzer = Analyzer()
        
        # Create data suitable for regression
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.random(100) * 100,
            'feature2': np.random.random(100) * 50,
            'target': np.random.random(100) * 200 + 50  # Target variable
        })
        
        processed_data = {
            "dataframe": df,
            "metadata": {"columns": ["feature1", "feature2", "target"]}
        }
        
        analysis_plan = {
            "analysis_type": "predictive",
            "analysis_steps": ["regression_analysis"]
        }
        
        result = await analyzer.analyze(processed_data, analysis_plan)
        
        assert "regression_analysis" in result
        assert "target_variable" in result["regression_analysis"]
        assert "feature_variables" in result["regression_analysis"]
        assert "linear_regression" in result["regression_analysis"]
        assert "random_forest" in result["regression_analysis"]
        
        # Check regression metrics
        lr_results = result["regression_analysis"]["linear_regression"]
        assert "r2_score" in lr_results
        assert "mse" in lr_results
        assert "coefficients" in lr_results
    
    @pytest.mark.asyncio
    async def test_analyze_clustering_analysis(self):
        """Test clustering analysis"""
        analyzer = Analyzer()
        
        # Create clusterable data
        np.random.seed(42)
        cluster1 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], 50)
        cluster2 = np.random.multivariate_normal([8, 8], [[1, 0], [0, 1]], 50)
        
        data = np.vstack([cluster1, cluster2])
        df = pd.DataFrame(data, columns=['x', 'y'])
        
        processed_data = {
            "dataframe": df,
            "metadata": {"columns": ["x", "y"]}
        }
        
        analysis_plan = {
            "analysis_type": "diagnostic",
            "analysis_steps": ["clustering_analysis"]
        }
        
        result = await analyzer.analyze(processed_data, analysis_plan)
        
        assert "clustering_analysis" in result
        assert "optimal_clusters" in result["clustering_analysis"]
        assert "cluster_centers" in result["clustering_analysis"]
        assert "cluster_counts" in result["clustering_analysis"]
    
    @pytest.mark.asyncio
    async def test_analyze_insufficient_data(self):
        """Test analysis with insufficient data"""
        analyzer = Analyzer()
        
        # Single column data (insufficient for correlation)
        df = pd.DataFrame({'single_col': [1, 2, 3]})
        
        processed_data = {
            "dataframe": df,
            "metadata": {"columns": ["single_col"]}
        }
        
        analysis_plan = {
            "analysis_type": "descriptive",
            "analysis_steps": ["correlation_analysis", "regression_analysis"]
        }
        
        result = await analyzer.analyze(processed_data, analysis_plan)
        
        # Should handle gracefully with messages
        assert "correlation_analysis" in result
        assert "message" in result["correlation_analysis"]
        assert "Need at least 2 numeric columns" in result["correlation_analysis"]["message"]
    
    @pytest.mark.asyncio
    async def test_analyze_non_numeric_data(self):
        """Test analysis with non-numeric data"""
        analyzer = Analyzer()
        
        # Non-numeric data
        df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'text': ['hello', 'world', 'test', 'data', 'analysis']
        })
        
        processed_data = {
            "dataframe": df,
            "metadata": {"columns": ["category", "text"]}
        }
        
        analysis_plan = {
            "analysis_type": "descriptive",
            "analysis_steps": ["summary_statistics", "correlation_analysis"]
        }
        
        result = await analyzer.analyze(processed_data, analysis_plan)
        
        # Should handle gracefully
        assert "summary_statistics" in result
        assert "message" in result["summary_statistics"]
        assert "No numeric columns" in result["summary_statistics"]["message"]
    
    @pytest.mark.asyncio
    async def test_analyze_multiple_steps(self):
        """Test analysis with multiple steps"""
        analyzer = Analyzer()
        
        # Rich dataset for multiple analyses
        np.random.seed(42)
        df = pd.DataFrame({
            'age': np.random.randint(20, 80, 100),
            'income': np.random.normal(50000, 15000, 100),
            'experience': np.random.randint(0, 40, 100),
            'education_score': np.random.normal(75, 10, 100)
        })
        
        processed_data = {
            "dataframe": df,
            "metadata": {"columns": ["age", "income", "experience", "education_score"]}
        }
        
        analysis_plan = {
            "analysis_type": "comprehensive",
            "analysis_steps": [
                "summary_statistics", 
                "correlation_analysis", 
                "distribution_analysis",
                "regression_analysis"
            ]
        }
        
        result = await analyzer.analyze(processed_data, analysis_plan)
        
        # Should have all requested analyses
        assert "summary_statistics" in result
        assert "correlation_analysis" in result
        assert "distribution_analysis" in result
        assert "regression_analysis" in result
        assert "insights" in result
        
        # Check insights were generated
        assert len(result["insights"]) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_no_steps(self):
        """Test analysis with no specific steps"""
        analyzer = Analyzer()
        
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        
        processed_data = {
            "dataframe": df,
            "metadata": {"columns": ["col1", "col2"]}
        }
        
        analysis_plan = {
            "analysis_type": "basic"
            # No analysis_steps specified
        }
        
        result = await analyzer.analyze(processed_data, analysis_plan)
        
        # Should still return basic structure
        assert "data_summary" in result
        assert "insights" in result
        assert result["analysis_type"] == "basic"