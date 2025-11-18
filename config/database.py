# app/db/database.py

from pymongo import MongoClient
from dotenv import load_dotenv
import os
from config.logger import logger
# Configure logger

# Load environment variables from .env file
load_dotenv()

class DataBase:
    client: MongoClient = None

db = DataBase()

def get_database():
    """
    Returns the database instance from the client.
    """
    try:
        logger.debug("Getting database instance...")
        mongo_db_name = os.getenv("DB_NAME", "HRM_AGENT")
        
        if not mongo_db_name:
            logger.error("DB_NAME not set in environment variables")
            raise ValueError("DB_NAME not set in environment variables")
        
        if not db.client:
            logger.error("Database client not initialized")
            raise ValueError("Database client not initialized. Call connect_to_mongo() first.")
        
        database = db.client[mongo_db_name]
        logger.debug(f"Database instance '{mongo_db_name}' retrieved successfully")
        return database
        
    except Exception as e:
        logger.error(f"Error getting database instance: {e}")
        raise
    finally:
        logger.debug("Get database operation completed")



def connect_to_mongo():
    """
    Connects to the MongoDB database.
    """
    try:
        logger.info("Attempting to connect to MongoDB...")
        mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")
        
        if not mongo_connection_string:
            logger.error("MONGO_CONNECTION_STRING not set in environment variables")
            raise ValueError("MONGO_CONNECTION_STRING not set in environment variables")
        
        logger.debug("Creating MongoDB client...")
        db.client = MongoClient(mongo_connection_string)
        
        # Test the connection
        db.client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise
    finally:
        logger.debug("MongoDB connection attempt completed")

def close_mongo_connection():
    """
    Closes the MongoDB connection.
    """
    try:
        logger.info("Closing MongoDB connection...")
        if db.client:
            db.client.close()
            db.client = None
            logger.info("MongoDB connection closed successfully")
        else:
            logger.warning("No MongoDB connection to close")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {e}")
        raise
    finally:
        logger.debug("MongoDB connection closure operation completed")
def get_collection(collection_name: str):
    """
    Returns a specific collection from the database.
    Example:
        users_collection = get_collection("users")
    """
    try:
        logger.debug(f"Getting collection '{collection_name}'...")
        
        if not collection_name:
            logger.error("Collection name cannot be empty")
            raise ValueError("Collection name cannot be empty")
        
        database = get_database()
        collection = database[collection_name]
        logger.debug(f"Collection '{collection_name}' retrieved successfully")
        return collection
        
    except Exception as e:
        logger.error(f"Error getting collection '{collection_name}': {e}")
        raise
    finally:
        logger.debug(f"Get collection '{collection_name}' operation completed")

def get_mongo_db():
    """
    Alias for get_database() to maintain compatibility.
    Returns the database instance for both sync and async operations.
    """
    return get_database()
