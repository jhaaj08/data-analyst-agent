"""
Analysis service for performing statistical analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


class Analyzer:
    """Service for performing data analysis"""
    
    async def analyze(self, processed_data: Dict[str, Any], analysis_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform analysis based on the analysis plan
        
        Args:
            processed_data: Processed data information
            analysis_plan: Analysis plan from LLM
            
        Returns:
            Analysis results
        """
        df = processed_data["dataframe"]
        analysis_type = analysis_plan.get("analysis_type", "descriptive")
        analysis_steps = analysis_plan.get("analysis_steps", [])
        
        results = {
            "analysis_type": analysis_type,
            "data_summary": self._get_basic_summary(df)
        }
        
        # Perform analysis based on steps
        for step in analysis_steps:
            if step == "summary_statistics":
                results["summary_statistics"] = self._summary_statistics(df)
            elif step == "correlation_analysis":
                results["correlation_analysis"] = self._correlation_analysis(df)
            elif step == "distribution_analysis":
                results["distribution_analysis"] = self._distribution_analysis(df)
            elif step == "regression_analysis":
                results["regression_analysis"] = self._regression_analysis(df)
            elif step == "clustering_analysis":
                results["clustering_analysis"] = self._clustering_analysis(df)
            elif step == "time_series_analysis":
                results["time_series_analysis"] = self._time_series_analysis(df)
        
        # Add insights
        results["insights"] = self._generate_insights(df, results)
        
        return results
    
    def _get_basic_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic data summary"""
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=['number']).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "missing_values_total": df.isnull().sum().sum(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
    
    def _summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics"""
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return {"message": "No numeric columns found for summary statistics"}
        
        return {
            "descriptive_stats": numeric_df.describe().to_dict(),
            "skewness": numeric_df.skew().to_dict(),
            "kurtosis": numeric_df.kurtosis().to_dict()
        }
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform correlation analysis"""
        numeric_df = df.select_dtypes(include=['number'])
        
        if len(numeric_df.columns) < 2:
            return {"message": "Need at least 2 numeric columns for correlation analysis"}
        
        correlation_matrix = numeric_df.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        "variable1": correlation_matrix.columns[i],
                        "variable2": correlation_matrix.columns[j],
                        "correlation": corr_value
                    })
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "strong_correlations": strong_correlations
        }
    
    def _distribution_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric variables"""
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return {"message": "No numeric columns found for distribution analysis"}
        
        distribution_stats = {}
        
        for col in numeric_df.columns:
            data = numeric_df[col].dropna()
            
            # Normality test
            _, p_value = stats.normaltest(data)
            is_normal = p_value > 0.05
            
            distribution_stats[col] = {
                "mean": data.mean(),
                "median": data.median(),
                "std": data.std(),
                "min": data.min(),
                "max": data.max(),
                "is_normal": is_normal,
                "normality_p_value": p_value
            }
        
        return distribution_stats
    
    def _regression_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform regression analysis"""
        numeric_df = df.select_dtypes(include=['number'])
        
        if len(numeric_df.columns) < 2:
            return {"message": "Need at least 2 numeric columns for regression analysis"}
        
        # Use the last column as target, others as features
        target_col = numeric_df.columns[-1]
        feature_cols = numeric_df.columns[:-1]
        
        X = numeric_df[feature_cols]
        y = numeric_df[target_col]
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:  # Need minimum data points
            return {"message": "Insufficient data for regression analysis"}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Linear regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        # Random Forest regression
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        return {
            "target_variable": target_col,
            "feature_variables": feature_cols.tolist(),
            "linear_regression": {
                "r2_score": r2_score(y_test, lr_pred),
                "mse": mean_squared_error(y_test, lr_pred),
                "coefficients": dict(zip(feature_cols, lr_model.coef_))
            },
            "random_forest": {
                "r2_score": r2_score(y_test, rf_pred),
                "mse": mean_squared_error(y_test, rf_pred),
                "feature_importance": dict(zip(feature_cols, rf_model.feature_importances_))
            }
        }
    
    def _clustering_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis"""
        from sklearn.cluster import KMeans
        
        numeric_df = df.select_dtypes(include=['number']).dropna()
        
        if len(numeric_df.columns) < 2 or len(numeric_df) < 10:
            return {"message": "Insufficient data for clustering analysis"}
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, min(10, len(numeric_df) // 2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(numeric_df)
            inertias.append(kmeans.inertia_)
        
        # Use k=3 as default if not enough data for elbow method
        optimal_k = 3 if len(k_range) == 0 else k_range[np.argmin(np.diff(inertias))]
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(numeric_df)
        
        return {
            "optimal_clusters": optimal_k,
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "cluster_counts": np.bincount(clusters).tolist(),
            "inertia": kmeans.inertia_
        }
    
    def _time_series_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform time series analysis if datetime columns exist"""
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        
        if len(datetime_cols) == 0:
            # Try to find columns that might be dates
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                        datetime_cols = [col]
                        break
                    except:
                        continue
        
        if len(datetime_cols) == 0:
            return {"message": "No datetime columns found for time series analysis"}
        
        date_col = datetime_cols[0]
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            return {"message": "No numeric columns found for time series analysis"}
        
        # Sort by date
        df_sorted = df.sort_values(date_col)
        
        # Basic time series statistics
        results = {
            "date_column": date_col,
            "date_range": {
                "start": df_sorted[date_col].min().isoformat(),
                "end": df_sorted[date_col].max().isoformat()
            },
            "total_periods": len(df_sorted)
        }
        
        # Trend analysis for numeric columns
        trends = {}
        for col in numeric_cols:
            data = df_sorted[[date_col, col]].dropna()
            if len(data) > 2:
                # Simple linear trend
                x = np.arange(len(data))
                y = data[col].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                trends[col] = {
                    "slope": slope,
                    "r_squared": r_value ** 2,
                    "trend": "increasing" if slope > 0 else "decreasing"
                }
        
        results["trends"] = trends
        return results
    
    def _generate_insights(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate insights based on analysis results"""
        insights = []
        
        # Data quality insights
        if analysis_results["data_summary"]["missing_values_total"] > 0:
            missing_pct = (analysis_results["data_summary"]["missing_values_total"] / 
                          (df.shape[0] * df.shape[1])) * 100
            insights.append(f"Dataset has {missing_pct:.1f}% missing values")
        
        # Correlation insights
        if "correlation_analysis" in analysis_results:
            strong_corrs = analysis_results["correlation_analysis"].get("strong_correlations", [])
            if strong_corrs:
                insights.append(f"Found {len(strong_corrs)} strong correlations between variables")
        
        # Distribution insights
        if "distribution_analysis" in analysis_results:
            normal_vars = [var for var, stats in analysis_results["distribution_analysis"].items() 
                          if isinstance(stats, dict) and stats.get("is_normal")]
            if normal_vars:
                insights.append(f"Variables {normal_vars} follow normal distribution")
        
        # Regression insights
        if "regression_analysis" in analysis_results:
            r2 = analysis_results["regression_analysis"].get("linear_regression", {}).get("r2_score")
            if r2 and r2 > 0.7:
                insights.append(f"Strong predictive relationship found (R² = {r2:.3f})")
        
        return insights 