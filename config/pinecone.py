from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from config.logger import logger
# Configure logger

# Load .env
load_dotenv()

class PineconeDB: 
    client: Pinecone = None
    index = None

db = PineconeDB()
dimension = 384
def connect_to_pinecone(index_name: str = None):
    """
    Connects to Pinecone, creates the index if it doesn't exist, and sets the index.
    """
    try:
        logger.info("Attempting to connect to Pinecone...",{"index":index_name,"N":dimension})
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")  # e.g., 'us-east1-gcp'
        
        if not api_key or not environment:
            logger.error("PINECONE_API_KEY or PINECONE_ENVIRONMENT not set in .env")
            raise ValueError("PINECONE_API_KEY or PINECONE_ENVIRONMENT not set in .env")
        
        # Use index name from .env if not passed
        if index_name is None:
            index_name = os.getenv("PINECONE_INDEX_NAME", "quickstart")
        
        logger.debug(f"Initializing Pinecone client with index: {index_name}")
        db.client = Pinecone(api_key=api_key)

        # Check if index exists
        logger.debug("Checking for existing indexes...")
        existing_indexes = [i.name for i in db.client.list_indexes()]
        
        if index_name not in existing_indexes:
            logger.info(f"Index '{index_name}' not found. Creating...")
            db.client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=environment
                )
            )
            logger.info(f"Index '{index_name}' created successfully")
        else:
            logger.info(f"Index '{index_name}' already exists")

        # Connect to the index
        logger.debug(f"Connecting to index '{index_name}'...")
        db.index = db.client.Index(index_name)
        logger.info(f"Successfully connected to Pinecone index '{index_name}'")
        
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {e}")
        raise
    finally:
        logger.debug("Pinecone connection attempt completed")

def close_pinecone_connection():
    """
    Clears the Pinecone client/index.
    """
    try:
        logger.info("Closing Pinecone connection...")
        if db.index or db.client:
            db.index = None
            db.client = None
            logger.info("Pinecone connection closed successfully")
        else:
            logger.warning("No Pinecone connection to close")
    except Exception as e:
        logger.error(f"Error closing Pinecone connection: {e}")
        raise
    finally:
        logger.debug("Pinecone connection closure operation completed")

def get_pinecone_index():
    """
    Returns the active Pinecone index.
    """
    try:
        logger.debug("Getting Pinecone index...")
        if not db.index:
            logger.error("Pinecone index not initialized")
            raise ValueError("Pinecone index not initialized. Call connect_to_pinecone() first.")
        
        logger.debug("Pinecone index retrieved successfully")
        return db.index
        
    except Exception as e:
        logger.error(f"Error getting Pinecone index: {e}")
        raise
    finally:
        logger.debug("Get Pinecone index operation completed")
def insert_vector(id: str, vector: list, metadata: dict = None):
    """
    Insert a single vector with optional metadata into Pinecone.
    """
    try:
        logger.debug(f"Inserting vector with ID: {id}")
        
        if not db.index:
            logger.error("Pinecone index not initialized")
            raise ValueError("Pinecone index not initialized. Call connect_to_pinecone() first.")
        
        if not id or not vector:
            logger.error("Vector ID and values are required")
            raise ValueError("Vector ID and values are required")
        
        item = {"id": id, "values": vector}
        if metadata:
            item["metadata"] = metadata

        db.index.upsert(vectors=[item])
        logger.info(f"Successfully inserted vector ID '{id}' with metadata: {metadata}")
        
    except Exception as e:
        logger.error(f"Error inserting vector '{id}': {e}")
        raise
    finally:
        logger.debug(f"Insert vector '{id}' operation completed")

def query_vector(vector: list, top_k: int = 5, filter_metadata: dict = None):
    """
    Query Pinecone for similar vectors and return a list of matches.
    """
    response = db.index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_metadata or {}
    )

    # Safe access to matches
    return response.matches if hasattr(response, "matches") else response["matches"]