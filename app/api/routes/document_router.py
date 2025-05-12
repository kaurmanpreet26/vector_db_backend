"""
Router for document management.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import List, Optional
import os
import shutil
from pathlib import Path

from app.models.document import DocumentResponse, DocumentList
from app.parsers.document_parser import parse_document
from app.db.vector_db import get_vector_db
from app.core.config import settings

router = APIRouter()

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_immediately: bool = Form(False),
):
    """
    Upload a document to be processed and added to the vector database.
    
    The document will be saved to the data directory and optionally processed immediately.
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        
        # Save the uploaded file
        file_path = Path(settings.DATA_DIR) / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document in the background or immediately
        if process_immediately:
            # Parse the document and add to vector database
            document = parse_document(file_path)
            vector_db = get_vector_db()
            vector_db.add_document(document)
            status = "processed"
        else:
            # Schedule background processing
            background_tasks.add_task(process_document, file_path)
            status = "pending"
        
        return DocumentResponse(
            filename=file.filename,
            status=status,
            message=f"Document uploaded successfully. Status: {status}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )

@router.get("/list", response_model=DocumentList)
async def list_documents():
    """
    List all documents in the data directory.
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        
        # Get list of files in the data directory
        files = []
        for file_path in Path(settings.DATA_DIR).glob("*"):
            if file_path.is_file():
                files.append({
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "last_modified": file_path.stat().st_mtime
                })
        
        return DocumentList(documents=files)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )

async def process_document(file_path: Path):
    """
    Process a document and add it to the vector database.
    
    This function is meant to be run as a background task.
    """
    try:
        document = parse_document(file_path)
        vector_db = get_vector_db()
        vector_db.add_document(document)
    except Exception as e:
        print(f"Error processing document {file_path}: {str(e)}")
