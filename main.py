"""
Vector Database Backend API
Main application entry point
"""
import os
import uvicorn
from dotenv import load_dotenv
from app.api.api import app
from app.core.config import settings
import numpy

# Load environment variables
load_dotenv()

print("Numpy version in server:", numpy.__version__)

if __name__ == "__main__":
    # Run the FastAPI application with uvicorn
    uvicorn.run(
        "app.api.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
