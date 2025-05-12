"""
File utility functions.
"""
import os
import shutil
from pathlib import Path
from typing import List, Set, Optional
import mimetypes

def get_supported_extensions() -> Set[str]:
    """
    Get a set of supported file extensions.
    
    Returns:
        Set of supported file extensions (with dot prefix)
    """
    return {
        ".pdf", ".docx", ".txt", ".md", ".html", ".htm"
    }

def is_supported_file(file_path: Path) -> bool:
    """
    Check if a file is supported for parsing.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is supported, False otherwise
    """
    return file_path.suffix.lower() in get_supported_extensions()

def get_mime_type(file_path: Path) -> str:
    """
    Get the MIME type of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type of the file
    """
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or "application/octet-stream"

def list_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    List all files in a directory.
    
    Args:
        directory: Path to the directory
        recursive: Whether to list files recursively
        
    Returns:
        List of file paths
    """
    if not directory.exists() or not directory.is_dir():
        return []
    
    if recursive:
        return [p for p in directory.glob("**/*") if p.is_file()]
    else:
        return [p for p in directory.glob("*") if p.is_file()]

def list_supported_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    List all supported files in a directory.
    
    Args:
        directory: Path to the directory
        recursive: Whether to list files recursively
        
    Returns:
        List of supported file paths
    """
    all_files = list_files(directory, recursive)
    return [f for f in all_files if is_supported_file(f)]

def ensure_directory(directory: Path) -> None:
    """
    Ensure a directory exists.
    
    Args:
        directory: Path to the directory
    """
    os.makedirs(directory, exist_ok=True)

def safe_delete_file(file_path: Path) -> bool:
    """
    Safely delete a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file was deleted, False otherwise
    """
    try:
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            return True
    except Exception as e:
        print(f"Error deleting file {file_path}: {str(e)}")
    
    return False
