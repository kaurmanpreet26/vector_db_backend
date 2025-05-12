"""
Document parser module for extracting text from various file types.
"""
import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import pypdf
import docx2txt
import markdown
from bs4 import BeautifulSoup

from app.models.document import Document

def parse_document(file_path: Path) -> Document:
    """
    Parse a document file and extract its text content.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Document object with extracted content
    """
    # Generate a unique ID for the document
    doc_id = str(uuid.uuid4())
    
    # Get the file extension
    file_ext = file_path.suffix.lower()
    
    # Extract metadata
    metadata = extract_metadata(file_path)
    
    # Parse the document based on its type
    if file_ext == ".pdf":
        content = parse_pdf(file_path)
    elif file_ext == ".docx":
        content = parse_docx(file_path)
    elif file_ext == ".txt":
        content = parse_text(file_path)
    elif file_ext == ".md":
        content = parse_markdown(file_path)
    elif file_ext in [".html", ".htm"]:
        content = parse_html(file_path)
    else:
        # Default to treating as text
        content = parse_text(file_path)
    
    # Create the document object
    document = Document(
        id=doc_id,
        filename=file_path.name,
        content=content,
        metadata=metadata,
        chunks=[]  # Will be chunked by the vector database
    )
    
    return document

def extract_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary of metadata
    """
    metadata = {
        "source": str(file_path),
        "filename": file_path.name,
        "extension": file_path.suffix.lower(),
        "size_bytes": file_path.stat().st_size,
        "created_at": file_path.stat().st_ctime,
        "modified_at": file_path.stat().st_mtime,
    }
    
    return metadata

def parse_pdf(file_path: Path) -> str:
    """
    Parse a PDF file and extract its text content.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    text = ""
    
    try:
        # Open the PDF file
        with open(file_path, "rb") as file:
            # Create a PDF reader object
            pdf_reader = pypdf.PdfReader(file)
            
            # Extract text from each page
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
    except Exception as e:
        print(f"Error parsing PDF {file_path}: {str(e)}")
        text = f"Error parsing PDF: {str(e)}"
    
    return text

def parse_docx(file_path: Path) -> str:
    """
    Parse a DOCX file and extract its text content.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        Extracted text content
    """
    try:
        # Extract text from the DOCX file
        text = docx2txt.process(file_path)
    except Exception as e:
        print(f"Error parsing DOCX {file_path}: {str(e)}")
        text = f"Error parsing DOCX: {str(e)}"
    
    return text

def parse_text(file_path: Path) -> str:
    """
    Parse a text file and extract its content.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Extracted text content
    """
    try:
        # Read the text file
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    except UnicodeDecodeError:
        # Try with a different encoding
        try:
            with open(file_path, "r", encoding="latin-1") as file:
                text = file.read()
        except Exception as e:
            print(f"Error parsing text file {file_path}: {str(e)}")
            text = f"Error parsing text file: {str(e)}"
    except Exception as e:
        print(f"Error parsing text file {file_path}: {str(e)}")
        text = f"Error parsing text file: {str(e)}"
    
    return text

def parse_markdown(file_path: Path) -> str:
    """
    Parse a Markdown file and extract its text content.
    
    Args:
        file_path: Path to the Markdown file
        
    Returns:
        Extracted text content
    """
    try:
        # Read the Markdown file
        with open(file_path, "r", encoding="utf-8") as file:
            md_text = file.read()
        
        # Convert Markdown to HTML
        html = markdown.markdown(md_text)
        
        # Extract text from HTML
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
    except Exception as e:
        print(f"Error parsing Markdown {file_path}: {str(e)}")
        text = f"Error parsing Markdown: {str(e)}"
    
    return text

def parse_html(file_path: Path) -> str:
    """
    Parse an HTML file and extract its text content.
    
    Args:
        file_path: Path to the HTML file
        
    Returns:
        Extracted text content
    """
    try:
        # Read the HTML file
        with open(file_path, "r", encoding="utf-8") as file:
            html = file.read()
        
        # Extract text from HTML
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        text = "\n".join(chunk for chunk in chunks if chunk)
    except Exception as e:
        print(f"Error parsing HTML {file_path}: {str(e)}")
        text = f"Error parsing HTML: {str(e)}"
    
    return text
