"""
File handling utilities
"""
import os
import aiofiles
from fastapi import UploadFile, HTTPException
from typing import Optional
import uuid

from ..core.config import settings


class FileHandler:
    """Handle file operations for uploads and outputs"""
    
    def __init__(self):
        self.upload_dir = settings.upload_dir
        self.output_dir = settings.output_dir
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure upload and output directories exist"""
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def save_upload(self, file: UploadFile, analysis_id: str) -> str:
        """
        Save uploaded file to disk
        
        Args:
            file: Uploaded file
            analysis_id: Unique analysis identifier
            
        Returns:
            Path to saved file
        """
        # Validate file size
        if file.size and file.size > settings.max_file_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File size exceeds limit of {settings.max_file_size} bytes"
            )
        
        # Validate file type
        allowed_types = {'.csv', '.json', '.xlsx', '.xls', '.txt', '.parquet'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Allowed types: {allowed_types}"
            )
        
        # Create analysis-specific directory
        analysis_dir = os.path.join(self.upload_dir, analysis_id)
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(analysis_dir, file.filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return file_path
    
    def create_output_dir(self, analysis_id: str) -> str:
        """
        Create output directory for analysis results
        
        Args:
            analysis_id: Unique analysis identifier
            
        Returns:
            Path to output directory
        """
        output_path = os.path.join(self.output_dir, analysis_id)
        os.makedirs(output_path, exist_ok=True)
        return output_path
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Get basic file information
        
        Args:
            file_path: Path to file
            
        Returns:
            File information dictionary
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = os.stat(file_path)
        filename = os.path.basename(file_path)
        
        return {
            "filename": filename,
            "size": stat.st_size,
            "extension": os.path.splitext(filename)[1].lower(),
            "path": file_path
        } 