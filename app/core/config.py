"""
Configuration settings for the Data Analyst Agent
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    api_title: str = "Data Analyst Agent API"
    api_description: str = "API that uses LLMs to source, prepare, analyze, and visualize data"
    api_version: str = "1.0.0"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None  # Replace anthropic with gemini
    
    # File paths
    upload_dir: str = "data/uploads"
    output_dir: str = "data/outputs"
    
    # Processing limits
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    analysis_timeout: int = 300  # 5 minutes
    
    class Config:
        env_file = ".env"


settings = Settings() 