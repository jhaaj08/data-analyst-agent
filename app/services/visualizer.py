"""
Visualization service for generating charts and plots
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import os
from typing import Dict, Any, List

from ..core.config import settings
from ..utils.file_handler import FileHandler


class Visualizer:
    """Service for creating data visualizations"""
    
    def __init__(self):
        self.file_handler = FileHandler()
        # Set matplotlib backend
        plt.style.use('default')
        sns.set_palette("husl")
    
    async def create_visualizations(
        self, 
        processed_data: Dict[str, Any], 
        analysis_results: Dict[str, Any], 
        analysis_plan: Dict[str, Any], 
        analysis_id: str
    ) -> List[str]:
        """
        Create visualizations based on analysis results
        
        Args:
            processed_data: Processed data information
            analysis_results: Results from analysis
            analysis_plan: Analysis plan from LLM
            analysis_id: Unique analysis identifier
            
        Returns:
            List of visualization file paths
        """
        df = processed_data["dataframe"]
        viz_requirements = analysis_plan.get("visualization_requirements", [])
        
        # Create output directory
        output_dir = self.file_handler.create_output_dir(analysis_id)
        visualizations = []
        
        for viz_type in viz_requirements:
            try:
                if viz_type == "histogram":
                    viz_path = await self._create_histogram(df, output_dir)
                    visualizations.append(viz_path)
                elif viz_type == "correlation_heatmap":
                    viz_path = await self._create_correlation_heatmap(df, output_dir)
                    visualizations.append(viz_path)
                elif viz_type == "scatter_plot":
                    viz_path = await self._create_scatter_plot(df, output_dir)
                    visualizations.append(viz_path)
                elif viz_type == "box_plot":
                    viz_path = await self._create_box_plot(df, output_dir)
                    visualizations.append(viz_path)
                elif viz_type == "bar_chart":
                    viz_path = await self._create_bar_chart(df, output_dir)
                    visualizations.append(viz_path)
                elif viz_type == "line_chart":
                    viz_path = await self._create_line_chart(df, output_dir)
                    visualizations.append(viz_path)
                elif viz_type == "distribution_plot":
                    viz_path = await self._create_distribution_plot(df, output_dir)
                    visualizations.append(viz_path)
            except Exception as e:
                print(f"Error creating {viz_type}: {e}")
                continue
        
        # Create summary dashboard if multiple visualizations
        if len(visualizations) > 1:
            try:
                dashboard_path = await self._create_dashboard(df, analysis_results, output_dir)
                visualizations.append(dashboard_path)
            except Exception as e:
                print(f"Error creating dashboard: {e}")
        
        return [os.path.basename(viz) for viz in visualizations]
    
    async def _create_histogram(self, df: pd.DataFrame, output_dir: str) -> str:
        """Create histogram for numeric columns"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns for histogram")
        
        # Create subplots
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, "histogram.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
    
    async def _create_correlation_heatmap(self, df: pd.DataFrame, output_dir: str) -> str:
        """Create correlation heatmap"""
        numeric_df = df.select_dtypes(include=['number'])
        
        if len(numeric_df.columns) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation heatmap")
        
        correlation_matrix = numeric_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
    
    async def _create_scatter_plot(self, df: pd.DataFrame, output_dir: str) -> str:
        """Create scatter plot for first two numeric columns"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for scatter plot")
        
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_col], df[y_col], alpha=0.6)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{y_col} vs {x_col}')
        
        # Add trend line
        z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
        p = np.poly1d(z)
        plt.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, "scatter_plot.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
    
    async def _create_box_plot(self, df: pd.DataFrame, output_dir: str) -> str:
        """Create box plot for numeric columns"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns for box plot")
        
        # Limit to first 5 columns to avoid overcrowding
        cols_to_plot = numeric_cols[:5]
        
        plt.figure(figsize=(12, 6))
        df[cols_to_plot].boxplot()
        plt.title('Box Plot of Numeric Variables')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, "box_plot.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
    
    async def _create_bar_chart(self, df: pd.DataFrame, output_dir: str) -> str:
        """Create bar chart for categorical columns"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            raise ValueError("No categorical columns for bar chart")
        
        # Use first categorical column
        col = categorical_cols[0]
        value_counts = df[col].value_counts().head(10)  # Top 10 values
        
        plt.figure(figsize=(12, 6))
        value_counts.plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, "bar_chart.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
    
    async def _create_line_chart(self, df: pd.DataFrame, output_dir: str) -> str:
        """Create line chart for time series or index-based data"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns for line chart")
        
        plt.figure(figsize=(12, 6))
        
        # Use first numeric column
        col = numeric_cols[0]
        plt.plot(df.index, df[col], marker='o', markersize=3)
        plt.title(f'{col} Over Index')
        plt.xlabel('Index')
        plt.ylabel(col)
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, "line_chart.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
    
    async def _create_distribution_plot(self, df: pd.DataFrame, output_dir: str) -> str:
        """Create distribution plot with KDE"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns for distribution plot")
        
        # Create subplots for first few numeric columns
        cols_to_plot = numeric_cols[:4]
        n_cols = min(2, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(cols_to_plot):
            if i < len(axes):
                sns.histplot(data=df, x=col, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
        
        # Hide empty subplots
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, "distribution_plot.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return file_path
    
    async def _create_dashboard(self, df: pd.DataFrame, analysis_results: Dict[str, Any], output_dir: str) -> str:
        """Create an interactive dashboard using Plotly"""
        # Create a simple HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Analysis Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .insights {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; }}
                .visualization {{ margin: 20px 0; text-align: center; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Data Analysis Dashboard</h1>
            
            <div class="summary">
                <h2>Data Summary</h2>
                <p><strong>Total Rows:</strong> {analysis_results['data_summary']['total_rows']}</p>
                <p><strong>Total Columns:</strong> {analysis_results['data_summary']['total_columns']}</p>
                <p><strong>Numeric Columns:</strong> {analysis_results['data_summary']['numeric_columns']}</p>
                <p><strong>Categorical Columns:</strong> {analysis_results['data_summary']['categorical_columns']}</p>
            </div>
            
            <div class="insights">
                <h2>Key Insights</h2>
                <ul>
        """
        
        for insight in analysis_results.get('insights', []):
            html_content += f"<li>{insight}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <h2>Visualizations</h2>
        """
        
        # Add visualizations to dashboard
        viz_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        for viz_file in viz_files:
            viz_name = viz_file.replace('.png', '').replace('_', ' ').title()
            html_content += f"""
            <div class="visualization">
                <h3>{viz_name}</h3>
                <img src="{viz_file}" alt="{viz_name}">
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        dashboard_path = os.path.join(output_dir, "dashboard.html")
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        return dashboard_path 