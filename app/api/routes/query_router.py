"""
Router for vector database queries.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional

from app.models.query import QueryRequest, QueryResponse, QueryResult
from app.db.vector_db import get_vector_db

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
        
        # Convert results to QueryResult objects
        query_results = [
            QueryResult(
                document_id=result["document_id"],
                content=result["content"],
                metadata=result["metadata"],
                score=result["score"]
            )
            for result in results["results"]
        ]
        
        return QueryResponse(
            query=query_request.query,
            response=results["response"],
            results=query_results
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying vector database: {str(e)}"
        )
