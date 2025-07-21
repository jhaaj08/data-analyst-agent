"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum


class AnalysisType(str, Enum):
    """Types of data analysis"""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"


class AnalysisRequest(BaseModel):
    """Request model for data analysis"""
    question: str
    format: Optional[str] = "json"
    data_source: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Response model for data analysis results"""
    success: bool
    analysis_id: str
    analysis_type: AnalysisType
    results: Dict[str, Any]
    visualizations: List[str] = []
    execution_time: float
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    details: Optional[str] = None


class DataInfo(BaseModel):
    """Model for data information"""
    filename: Optional[str] = None
    file_size: Optional[int] = None
    columns: List[str] = []
    shape: Optional[tuple] = None
    data_types: Dict[str, str] = {}
    missing_values: Dict[str, int] = {}


class AnalysisPlan(BaseModel):
    """Model for analysis plan"""
    analysis_type: AnalysisType
    data_operations: List[str]
    analysis_steps: List[str]
    visualization_requirements: List[str]
    expected_outputs: str 