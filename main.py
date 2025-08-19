#!/usr/bin/env python3
"""
Data Analyst Agent - Main Entry Point

A Python-based data analyst agent API for automated data analysis tasks.
"""

import uvicorn
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes import router
import time 
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RequestLoggerMiddleware(BaseHTTPMiddleware):
        async def set_body(self,request: Request, body: bytes):
            async def receive():
                return {"type": "http.request", "body": body}
            request._receive = receive
        async def dispatch(self, request: Request, call_next):
            start_time = time.time()
            # Log request details
            body  = await request.body()
            logger.info(f"raw request body : {body.decode()}")
            await self.set_body(request, body)
            logger.info(f"Incoming Request: {request.method} {request.url} - {request.headers} {request.body}")

            response = await call_next(request)

            # Log response details (optional)
            process_time = time.time() - start_time
            logger.info(f"Outgoing Response: {response.status_code} for {request.method} {request.url} - Processed in {process_time:.4f}s")
            return response
        

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    app.add_middleware(RequestLoggerMiddleware)


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

@app.post("/")  # Add this back to handle direct root calls
async def root_post_endpoint(request: Request):
    """
    Root POST endpoint - handles calls to the base URL
    Forwards to the main API logic to ensure compatibility
    """
    # Import here to avoid circular imports  
    from app.api.routes import analyze_complete_pipeline
    return await analyze_complete_pipeline(request)


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
