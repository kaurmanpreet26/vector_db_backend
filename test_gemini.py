import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gemini_connection():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    logger.info(f"API key found: {'Yes' if api_key else 'No'}")
    if api_key:
        logger.info(f"API key length: {len(api_key)}")
        logger.info(f"API key starts with: {api_key[:8]}...")
    
    try:
        # Configure Gemini
        logger.info("Configuring Gemini...")
        genai.configure(api_key=api_key)
        
        # List available models
        logger.info("Listing available models...")
        for m in genai.list_models():
            logger.info(f"Model: {m.name}")
        
        # Try different model names
        model_names = ['gemini-pro', 'gemini-1.0-pro', 'gemini-1.0-pro-latest']
        
        for model_name in model_names:
            try:
                logger.info(f"\nTrying model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Say hello!")
                if response and response.text:
                    logger.info(f"Success with {model_name}! Response received:")
                    logger.info(response.text)
                    break
            except Exception as e:
                logger.error(f"Error with {model_name}: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e)}")

if __name__ == "__main__":
    test_gemini_connection() 