"""
Main API module for the Vector Database Backend.
Defines the FastAPI application and includes routers.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import query_router, document_router
from app.core.config import settings

# Create FastAPI application
app = FastAPI(
    title="Vector Database API",
    description="API for querying vector database with document embeddings",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query_router.router, prefix="/api/query", tags=["Query"])
app.include_router(document_router.router, prefix="/api/documents", tags=["Documents"])

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Vector Database API",
        "docs": "/docs",
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
