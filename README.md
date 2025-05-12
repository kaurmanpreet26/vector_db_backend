# Vector Database Backend

A Python backend service that parses files, stores the data in a vector database, and provides an API for querying the database.

## Features

- File parsing for various formats (PDF, DOCX, TXT, MD, HTML)
- Vector database storage using Chroma DB
- FastAPI-based REST API
- Document management endpoints
- Vector similarity search

## Project Structure

```
vector_db_backend/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── query_router.py
│   │   │   └── document_router.py
│   │   └── api.py
│   ├── core/
│   │   └── config.py
│   ├── db/
│   │   └── vector_db.py
│   ├── models/
│   │   ├── document.py
│   │   └── query.py
│   ├── parsers/
│   │   └── document_parser.py
│   └── utils/
│       └── file_utils.py
├── data/
│   ├── documents/
│   └── vector_db/
├── tests/
├── .env
├── main.py
└── requirements.txt
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables in `.env` file (or use the default values)

## Usage

### Running the API Server

```bash
python main.py
```

The API will be available at http://localhost:8000

### API Documentation

Once the server is running, you can access the API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### API Endpoints

#### Document Management

- `POST /api/documents/upload`: Upload a document
- `GET /api/documents/list`: List all documents

#### Vector Database Queries

- `POST /api/query/`: Query the vector database

## Example Usage

### Upload a Document

```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "process_immediately=true"
```

### Query the Vector Database

```bash
curl -X POST "http://localhost:8000/api/query/" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "filters": null}'
```

## Development

### Running Tests

```bash
pytest
```

### Adding Support for New File Types

To add support for a new file type:

1. Add a new parser function in `app/parsers/document_parser.py`
2. Update the `parse_document` function to use the new parser
3. Add the new file extension to `get_supported_extensions` in `app/utils/file_utils.py`
