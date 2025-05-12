"""
Vector database module for document storage and retrieval.
"""
import os
from typing import List, Dict, Any, Optional
import uuid
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import numpy as np
import logging
import traceback

from app.core.config import settings
from app.models.document import Document
from app.models.query import QueryResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global vector database instance
_vector_db_instance = None

class VectorDB:
    """
    Vector database for document storage and retrieval.
    """
    
    def __init__(self, db_path: str, embedding_model: str):
        """
        Initialize the vector database.
        
        Args:
            db_path: Path to the vector database
            embedding_model: Name of the embedding model to use
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        
        try:
            # Create embeddings
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            
            # Create or load the vector database
            os.makedirs(db_path, exist_ok=True)
            self.db = Chroma(
                persist_directory=db_path,
                embedding_function=self.embeddings
            )
            
            # Log database state
            self._log_database_state()
        except Exception as e:
            raise Exception(f"Error initializing vector database: {str(e)}")
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
        )
        
        # Initialize Gemini
        logger.info("=== Gemini Initialization Debug ===")
        logger.info(f"Settings object type: {type(settings)}")
        logger.info(f"Settings attributes: {dir(settings)}")
        logger.info(f"Raw GEMINI_API_KEY from settings: {settings.GEMINI_API_KEY}")
        logger.info(f"GEMINI_API_KEY length: {len(settings.GEMINI_API_KEY) if settings.GEMINI_API_KEY else 0}")
        
        api_key = settings.GEMINI_API_KEY
        if api_key and api_key.strip():
            logger.info("GEMINI_API_KEY found and not empty")
            try:
                logger.info("Configuring Gemini with API key...")
                # Log the first few characters of the API key for verification
                logger.info(f"API key starts with: {api_key[:8]}...")
                genai.configure(api_key=api_key)
                logger.info("Successfully configured Gemini")
                
                logger.info("Creating Gemini model...")
                self.gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
                logger.info("Successfully created Gemini model")
                
                # Test Gemini connection
                logger.info("Testing Gemini connection...")
                test_response = self.gemini_model.generate_content("Hello, this is a test message.")
                if test_response and test_response.text:
                    logger.info("Successfully connected to Gemini API")
                    logger.info(f"Test response: {test_response.text[:100]}...")
                else:
                    logger.error("Failed to get response from Gemini API")
                    logger.error(f"Response object: {test_response}")
                    self.gemini_model = None
            except Exception as e:
                logger.error(f"Error initializing Gemini: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                self.gemini_model = None
        else:
            logger.warning("No valid GEMINI_API_KEY found in settings")
            logger.warning(f"API key value: '{api_key}'")
            self.gemini_model = None
        logger.info("=== End Gemini Initialization Debug ===")

    def _log_database_state(self):
        """Log the current state of the database."""
        try:
            # Get collection info
            collection = self.db._collection
            count = collection.count()
            logger.info(f"Database state - Total documents: {count}")
            
            # Get a sample of documents if any exist
            if count > 0:
                sample = collection.get(limit=1)
                logger.info(f"Sample document metadata: {sample['metadatas'][0] if sample['metadatas'] else 'No metadata'}")
                logger.info(f"Sample document content: {sample['documents'][0][:200] if sample['documents'] else 'No content'}")
        except Exception as e:
            logger.error(f"Error logging database state: {str(e)}")

    def _process_with_gemini(self, query_text: str, results: List[QueryResult]) -> Dict[str, Any]:
        """
        Process query results through Gemini to get structured response.
        
        Args:
            query_text: Original query text
            results: List of query results
            
        Returns:
            Dictionary containing query results and structured response
        """
        if not self.gemini_model:
            logger.warning("Gemini model not available, returning raw results")
            return {
                "query": query_text,
                "response": "I apologize, but I'm currently unable to process your request. Please try again later.",
                "results": [r.dict() for r in results]
            }
        
        if results:
            # Prepare context from results
            context = "\n\n".join([
                f"Document {i+1}:\n{r.content}\nRelevance: {r.score:.2f}"
                for i, r in enumerate(results)
            ])
            # Create prompt for Gemini
            prompt = f"""You are a helpful HR assistant. Based on the following query and retrieved documents, provide a friendly and conversational response.
            DO NOT return the information in a bullet-point or technical format. Instead, write it as if you're having a natural conversation with the user.

Query: {query_text}

Retrieved Documents:
{context}

Instructions:
1. Start with a friendly greeting like \"Hi!\" or \"Hello!\"
2. Acknowledge their question in a natural way
3. Provide the information in a conversational tone, as if you're explaining it to a friend
4. If there are multiple pieces of information, connect them naturally in sentences
5. Include any contact information in a natural way within the conversation
6. End with a helpful closing statement or offer to provide more information

Example format (DO NOT use bullet points or technical formatting):
\"Hi! I'd be happy to help you with that. [Natural explanation of the information]. If you need any help, you can reach out to [contact person] at [contact info]. Would you like me to explain anything else about this?\"

Remember:
- Write in a natural, conversational style
- Avoid bullet points, numbered lists, or technical formatting
- Make it sound like a friendly chat
- Include all the important information but present it in a flowing, natural way"""
        else:
            # No results prompt
            prompt = f"""You are a helpful HR assistant. The user asked: \"{query_text}\", but there were no relevant documents found in the database. Respond in a friendly, conversational way, letting them know you couldn't find an answer, and suggest they contact HR for more help. Offer to assist with anything else they might need."""
        
        try:
            logger.info("Sending request to Gemini")
            response = self.gemini_model.generate_content(prompt)
            logger.info(f"Received response from Gemini: {response.text[:200]}...")
            
            if not response.text:
                logger.warning("Empty response from Gemini")
                return {
                    "query": query_text,
                    "response": "I apologize, but I couldn't generate a response at this time. Please try again later.",
                    "results": [r.dict() for r in results]
                }
                
            return {
                "query": query_text,
                "response": response.text,
                "results": [r.dict() for r in results]
            }
        except Exception as e:
            logger.error(f"Error processing with Gemini: {str(e)}")
            return {
                "query": query_text,
                "response": f"I apologize, but I encountered an error while processing your request. Please try again later.",
                "results": [r.dict() for r in results]
            }
    
    def add_document(self, document: Document) -> str:
        """
        Add a document to the vector database.
        
        Args:
            document: The document to add
            
        Returns:
            The document ID
        """
        # If document has pre-defined chunks, use those
        if document.chunks:
            texts = [chunk["content"] for chunk in document.chunks]
            metadatas = [
                {
                    "document_id": document.id,
                    "filename": document.filename,
                    "chunk_id": i,
                    **chunk.get("metadata", {}),
                    **document.metadata
                }
                for i, chunk in enumerate(document.chunks)
            ]
        else:
            # Split the document into chunks
            texts = self.text_splitter.split_text(document.content)
            metadatas = [
                {
                    "document_id": document.id,
                    "filename": document.filename,
                    "chunk_id": i,
                    **document.metadata
                }
                for i in range(len(texts))
            ]
        
        # Add the chunks to the vector database
        self.db.add_texts(texts=texts, metadatas=metadatas)
        
        # Persist the database
        self.db.persist()
        
        return document.id
    
    def _is_generic_query(self, query_text: str) -> bool:
        """
        Check if the query is a generic greeting or small talk.
        
        Args:
            query_text: The query text to check
            
        Returns:
            True if the query is generic, False otherwise
        """
        generic_queries = [
            # Basic greetings
            "hi", "hello", "hey", "hiya", "yo",
            "hi there", "hello there", "hey there",
            
            # Time-based greetings
            "good morning", "good afternoon", "good evening",
            "morning", "afternoon", "evening",
            
            # How are you variations
            "how are you", "how's it going", "how's everything",
            "how are you doing", "how have you been",
            "how's your day", "how's your day going",
            "how's life", "how's everything going",
            
            # What's up variations
            "what's up", "whats up", "what's new",
            "what's happening", "what's going on",
            "what's the latest", "what's new with you",
            
            # Formal greetings
            "greetings", "salutations", "how do you do",
            "pleased to meet you", "nice to meet you",
            
            # Small talk
            "how's the weather", "nice weather", "beautiful day",
            "how's your weekend", "how was your weekend",
            "how's your week", "how was your week",
            
            # Thank you variations
            "thanks", "thank you", "thanks a lot",
            "thank you so much", "appreciate it",
            
            # Goodbye variations
            "bye", "goodbye", "see you", "see you later",
            "take care", "have a good day", "have a nice day",
            
            # Polite responses
            "you're welcome", "no problem", "anytime",
            "my pleasure", "don't mention it",
            
            # Help requests
            "can you help me", "i need help", "help me",
            "i have a question", "i need assistance"
        ]
        
        query_lower = query_text.lower().strip()
        return any(greeting in query_lower for greeting in generic_queries)

    def _get_generic_response(self, query_text: str) -> Dict[str, Any]:
        """
        Generate a response for generic queries with context-aware responses.
        
        Args:
            query_text: The generic query text
            
        Returns:
            Dictionary containing the response
        """
        if not self.gemini_model:
            return {
                "query": query_text,
                "response": "Hello! I'm your HR assistant. How can I help you today?",
                "results": []
            }
        
        query_lower = query_text.lower().strip()
        
        # Determine the type of query
        if any(greeting in query_lower for greeting in ["hi", "hello", "hey", "hiya", "yo", "hi there", "hello there", "hey there"]):
            prompt = f"""You are a friendly HR assistant. The user has greeted you with: "{query_text}"
            Respond in a warm, conversational way. Introduce yourself as an HR assistant and ask how you can help them with HR-related questions.
            Keep it brief and natural."""
            
        elif any(time_greeting in query_lower for time_greeting in ["good morning", "good afternoon", "good evening", "morning", "afternoon", "evening"]):
            prompt = f"""You are a friendly HR assistant. The user has greeted you with: "{query_text}"
            Respond with an appropriate time-based greeting and introduce yourself as an HR assistant.
            Ask how you can help them with HR-related questions today."""
            
        elif any(how_are_you in query_lower for how_are_you in ["how are you", "how's it going", "how's everything", "how are you doing"]):
            prompt = f"""You are a friendly HR assistant. The user has asked: "{query_text}"
            Respond warmly, briefly mention you're doing well, and pivot to asking how you can help them with HR-related questions.
            Keep it professional but friendly."""
            
        elif any(thanks in query_lower for thanks in ["thanks", "thank you", "thanks a lot", "thank you so much", "appreciate it"]):
            prompt = f"""You are a friendly HR assistant. The user has said: "{query_text}"
            Respond with a warm acknowledgment and offer to help with anything else they need.
            Keep it brief and professional."""
            
        elif any(bye in query_lower for bye in ["bye", "goodbye", "see you", "see you later", "take care"]):
            prompt = f"""You are a friendly HR assistant. The user has said: "{query_text}"
            Respond with a warm goodbye and remind them they can return anytime for HR-related assistance.
            Keep it brief and professional."""
            
        elif any(help in query_lower for help in ["can you help me", "i need help", "help me", "i have a question", "i need assistance"]):
            prompt = f"""You are a friendly HR assistant. The user has asked for help with: "{query_text}"
            Respond warmly and ask what specific HR-related question or topic they need help with.
            Mention you can help with policies, benefits, or other HR matters."""
            
        else:
            # Default response for other generic queries
            prompt = f"""You are a friendly HR assistant. The user has said: "{query_text}"
            Respond in a warm, conversational way. Introduce yourself as an HR assistant and ask how you can help them with HR-related questions.
            Keep it brief and natural."""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return {
                "query": query_text,
                "response": response.text,
                "results": []
            }
        except Exception as e:
            logger.error(f"Error generating generic response: {str(e)}")
            return {
                "query": query_text,
                "response": "Hello! I'm your HR assistant. How can I help you today?",
                "results": []
            }

    def query(
        self, 
        query_text: str, 
        limit: int = 5, 
        filters: Optional[Dict[str, Any]] = None,
        use_gemini: bool = True
    ) -> Dict[str, Any]:
        """
        Query the vector database.
        
        Args:
            query_text: The query text
            limit: Maximum number of results to return
            filters: Optional filters to apply
            use_gemini: Whether to process results through Gemini
            
        Returns:
            Dictionary containing query results and structured response
        """
        # Check if it's a generic query first
        if self._is_generic_query(query_text):
            logger.info(f"Detected generic query: {query_text}")
            return self._get_generic_response(query_text)

        def is_valid_filter(filters: Optional[Dict[str, Any]]) -> bool:
            if not filters or not isinstance(filters, dict):
                return False
            # Check if values are dicts with exactly one operator key
            for value in filters.values():
                if not isinstance(value, dict) or len(value) != 1:
                    return False
            return True

        # Log database state before query
        self._log_database_state()

        # Query the vector database
        filter_arg = filters if is_valid_filter(filters) else None
        logger.info(f"Querying with text: {query_text}, limit: {limit}, filters: {filter_arg}")

        try:
            # Get raw results without score first to check if we have any matches
            raw_results = self.db.similarity_search(
                query=query_text,
                k=limit,
                filter=filter_arg
            )
            logger.info(f"Initial search results count: {len(raw_results)}")
            
            # Now get results with scores
            results = self.db.similarity_search_with_score(
                query=query_text,
                k=limit,
                filter=filter_arg
            )
            logger.info(f"Raw results from database: {len(results)}")
            
            # Convert to QueryResult objects
            query_results = []
            SIMILARITY_THRESHOLD = 0.2  # Set threshold to 20% similarity
            logger.info(f"Using similarity threshold: {SIMILARITY_THRESHOLD}")
            
            # Log all raw results first
            logger.info("=== Raw Results Analysis ===")
            for i, (doc, score) in enumerate(results):
                similarity = 1.0 - score
                logger.info(f"Result {i+1}:")
                logger.info(f"  Similarity: {similarity:.4f} (raw score: {score:.4f})")
                logger.info(f"  Content: {doc.page_content[:200]}...")
                logger.info(f"  Metadata: {doc.metadata}")
                logger.info("---")
                # Print full chunk content and similarity for debugging
                print(f"CHUNK DEBUG: Similarity={similarity:.4f} | Content={doc.page_content}")
            
            # Now process results with threshold
            for doc, score in results:
                # Convert score to similarity (higher is better)
                similarity = 1.0 - score  # Score is already normalized between 0 and 1
                logger.info(f"Processing document with similarity: {similarity:.4f} (raw score: {score:.4f})")
                logger.info(f"Document content preview: {doc.page_content[:200]}")
                logger.info(f"Document metadata: {doc.metadata}")
                
                # Only include results above the similarity threshold
                if similarity > SIMILARITY_THRESHOLD:
                    logger.info(f"Including document (similarity {similarity:.4f} > threshold {SIMILARITY_THRESHOLD})")
                    query_results.append(
                        QueryResult(
                            document_id=doc.metadata.get("document_id", ""),
                            content=doc.page_content,
                            metadata=doc.metadata,
                            score=similarity
                        )
                    )
                else:
                    logger.info(f"Document excluded due to low similarity: {similarity:.4f} <= {SIMILARITY_THRESHOLD}")
            
            logger.info(f"Final filtered results count: {len(query_results)}")
            logger.info("=== End Results Analysis ===")
            
            # Always process results through Gemini if requested
            if use_gemini:
                structured_response = self._process_with_gemini(query_text, query_results)
                # Return the complete response including both the Gemini response and results
                return structured_response
            
            return {
                "query": query_text,
                "response": "No response generated (Gemini disabled)",
                "results": [r.dict() for r in query_results]
            }
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            raise Exception(f"Error querying vector database: {str(e)}")

def init_vector_db() -> VectorDB:
    """
    Initialize the vector database.
    
    Returns:
        The vector database instance
    """
    global _vector_db_instance
    
    if _vector_db_instance is None:
        _vector_db_instance = VectorDB(
            db_path=settings.VECTOR_DB_PATH,
            embedding_model=settings.EMBEDDING_MODEL
        )
    
    return _vector_db_instance

def get_vector_db() -> VectorDB:
    """
    Get the vector database instance.
    
    Returns:
        The vector database instance
    """
    global _vector_db_instance
    
    if _vector_db_instance is None:
        _vector_db_instance = init_vector_db()
    
    return _vector_db_instance
