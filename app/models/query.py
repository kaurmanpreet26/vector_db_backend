"""
Models for vector database queries.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    """
    Request model for vector database queries.
    """
    query: str = Field(..., description="The query text to search for")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional filters to apply to the query. Set to null to search without filters. When using filters, each filter must have exactly one operator (e.g., {'field': {'$eq': 'value'}})",
        example=None,
        json_schema_extra={"example": None}
    )

class QueryResult(BaseModel):
    """
    A single result from a vector database query.
    """
    document_id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="The content snippet from the document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Metadata associated with the document"
    )
    score: float = Field(..., description="Similarity score (higher is more similar)")

class QueryResponse(BaseModel):
    """
    Response model for vector database queries.
    """
    query: str = Field(..., description="The original query text")
    response: str = Field(..., description="The conversational response from Gemini")
    results: List[QueryResult] = Field(
        default_factory=list, 
        description="List of query results"
    )
