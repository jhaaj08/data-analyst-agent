#!/usr/bin/env python3
"""
Data Analyst Agent - Main Entry Point

A Python-based data analyst agent API for automated data analysis tasks.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes import router

        

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router, prefix="/api")
    
    # Mount static files for serving visualizations
    app.mount("/static", StaticFiles(directory=settings.output_dir), name="static")
    
    return app


# Create the application instance
app = create_app()


@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "Data Analyst Agent API",
        "version": settings.api_version,
        "docs": "/docs",
        "status": "active"
    }


def main():
    """Main function to run the data analyst agent API"""
    print(f"Starting {settings.api_title} v{settings.api_version}")
    print(f"API Documentation: http://{settings.host}:{settings.port}/docs")
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        access_log=True
    )


if __name__ == "__main__":
    main() 