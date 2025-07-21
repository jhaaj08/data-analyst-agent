"""
Data processing service for cleaning and preparing data
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataProcessor:
    """Service for processing and cleaning data"""
    
    async def process(self, data_info: Dict[str, Any], analysis_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data based on analysis plan
        
        Args:
            data_info: Data information dictionary
            analysis_plan: Analysis plan from LLM
            
        Returns:
            Processed data information
        """
        df = data_info["dataframe"].copy()
        operations = analysis_plan.get("data_operations", [])
        
        processing_log = []
        
        for operation in operations:
            if operation == "clean_data":
                df, log = self._clean_data(df)
                processing_log.extend(log)
            elif operation == "handle_missing":
                df, log = self._handle_missing_values(df)
                processing_log.extend(log)
            elif operation == "normalize":
                df, log = self._normalize_data(df)
                processing_log.extend(log)
            elif operation == "encode_categorical":
                df, log = self._encode_categorical(df)
                processing_log.extend(log)
            elif operation == "detect_outliers":
                outliers_info = self._detect_outliers(df)
                processing_log.append(f"Detected outliers: {outliers_info}")
        
        # Update data info with processed data
        processed_info = data_info.copy()
        processed_info["dataframe"] = df
        processed_info["processing_log"] = processing_log
        processed_info["metadata"]["processed_shape"] = df.shape
        
        return processed_info
    
    def _clean_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """Clean data by removing duplicates and handling basic issues"""
        log = []
        initial_shape = df.shape
        
        # Remove duplicates
        df = df.drop_duplicates()
        if df.shape[0] < initial_shape[0]:
            log.append(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")
        
        # Remove empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            df = df.drop(columns=empty_cols)
            log.append(f"Removed empty columns: {empty_cols}")
        
        # Strip whitespace from string columns
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].astype(str).str.strip()
        
        log.append("Applied basic data cleaning")
        return df, log
    
    def _handle_missing_values(self, df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """Handle missing values based on column types"""
        log = []
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    df[col] = df[col].fillna(df[col].median())
                    log.append(f"Filled {missing_count} missing values in {col} with median")
                else:
                    # Fill categorical columns with mode
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    df[col] = df[col].fillna(mode_value)
                    log.append(f"Filled {missing_count} missing values in {col} with mode")
        
        return df, log
    
    def _normalize_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """Normalize numeric columns"""
        log = []
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            log.append(f"Normalized columns: {numeric_cols.tolist()}")
        
        return df, log
    
    def _encode_categorical(self, df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """Encode categorical variables"""
        log = []
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if df[col].nunique() < 10:  # Only encode if few unique values
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                log.append(f"Encoded categorical column: {col}")
        
        return df, log
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        outliers_info = {}
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_info[col] = {
                "count": len(outliers),
                "percentage": (len(outliers) / len(df)) * 100,
                "bounds": {"lower": lower_bound, "upper": upper_bound}
            }
        
        return outliers_info
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "numeric_summary": df.describe().to_dict(),
            "categorical_summary": {
                col: {
                    "unique_count": df[col].nunique(),
                    "top_values": df[col].value_counts().head().to_dict()
                }
                for col in df.select_dtypes(include=['object']).columns
            }
        } 