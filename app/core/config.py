"""
Configuration settings for the application.
Loads settings from environment variables.
"""
import os
import logging
from pydantic_settings import BaseSettings
from dotenv import load_dotenv, find_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
logger.info("=== Environment Variables Debug ===")
logger.info(f"Current working directory: {os.getcwd()}")
dotenv_path = find_dotenv()
logger.info(f"Found .env file at: {dotenv_path}")
logger.info(f".env file exists: {os.path.exists(dotenv_path)}")

if os.path.exists(dotenv_path):
    with open(dotenv_path, 'r') as f:
        env_contents = f.read()
        logger.info("Contents of .env file:")
        for line in env_contents.splitlines():
            if 'GEMINI_API_KEY' in line:
                logger.info("GEMINI_API_KEY line found (value masked)")
            else:
                logger.info(line)

logger.info("Loading environment variables from .env file...")
load_dotenv(dotenv_path=dotenv_path)
logger.info(f"GEMINI_API_KEY present in environment: {'GEMINI_API_KEY' in os.environ}")
if 'GEMINI_API_KEY' in os.environ:
    api_key = os.environ['GEMINI_API_KEY']
    logger.info(f"GEMINI_API_KEY length: {len(api_key)}")
    logger.info(f"GEMINI_API_KEY first 4 chars: {api_key[:4] if len(api_key) >= 4 else 'too short'}")
else:
    logger.error("GEMINI_API_KEY not found in environment variables!")
    logger.error("Please make sure your .env file contains a line like: GEMINI_API_KEY=your_api_key_here")
logger.info("=== End Environment Variables Debug ===")

class Settings(BaseSettings):
    """Application settings."""
    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Vector Database Settings
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # File Parsing Settings
    DATA_DIR: str = os.getenv("DATA_DIR", "./data/documents")
    
    # Gemini API Settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("=== Settings Initialization Debug ===")
        logger.info(f"Settings initialized. GEMINI_API_KEY present: {bool(self.GEMINI_API_KEY)}")
        logger.info(f"GEMINI_API_KEY length: {len(self.GEMINI_API_KEY) if self.GEMINI_API_KEY else 0}")
        logger.info(f"All settings: {self.dict()}")
        logger.info("=== End Settings Initialization Debug ===")
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = True

# Create a settings instance
settings = Settings()
