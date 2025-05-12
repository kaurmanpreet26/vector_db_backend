"""
Models for document management.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class Document(BaseModel):
    """
    Model representing a processed document.
    """
    id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename")
    content: str = Field(..., description="Extracted text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Metadata associated with the document"
    )
    chunks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Document chunks for vector storage"
    )

class DocumentResponse(BaseModel):
    """
    Response model for document operations.
    """
    filename: str = Field(..., description="Document filename")
    status: str = Field(..., description="Status of the operation (e.g., 'processed', 'pending')")
    message: str = Field(..., description="Message describing the result")

class DocumentInfo(BaseModel):
    """
    Information about a document in storage.
    """
    filename: str = Field(..., description="Document filename")
    size: int = Field(..., description="File size in bytes")
    last_modified: float = Field(..., description="Last modified timestamp")
    
    @property
    def last_modified_datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.last_modified)

class DocumentList(BaseModel):
    """
    List of documents in storage.
    """
    documents: List[DocumentInfo] = Field(
        default_factory=list,
        description="List of documents"
    )
