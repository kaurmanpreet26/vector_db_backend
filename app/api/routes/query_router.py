"""
Router for vector database queries.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
import logging

from app.models.query import QueryRequest, QueryResponse, QueryResult
from app.db.vector_db import get_vector_db

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/", response_model=QueryResponse)
async def query_vector_db(
    query_request: QueryRequest,
    limit: int = Query(5, ge=1, le=20, description="Maximum number of results to return"),
):
    """
    Query the vector database with a text prompt.
    
    Returns similar documents based on vector similarity and a conversational response.
    
    Example request without filters:
    ```json
    {
        "query": "your search query",
        "filters": null
    }
    ```
    
    Example request with filters:
    ```json
    {
        "query": "your search query",
        "filters": {
            "filename": {
                "$eq": "document.pdf"
            }
        }
    }
    ```
    """
    try:
        # Get the vector database instance
        vector_db = get_vector_db()
        
        # Query the vector database
        results = vector_db.query(
            query_text=query_request.query,
            limit=limit,
            filters=query_request.filters
        )
        
        # The results are already in the correct format from vector_db.query
        return QueryResponse(
            query=results["query"],
            response=results["response"],
            results=results["results"]
        )
    except Exception as e:
        logger.error(f"Error in query_vector_db: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error querying vector database: {str(e)}"
        )
