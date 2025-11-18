from config.pinecone import insert_vector, query_vector, connect_to_pinecone
from config.database import get_database, connect_to_mongo, get_collection
from core.graph_agent import embed_query
from config.logger import logger

def get_all_collection_names():
    """
    Fetch all collection names from MongoDB database.
    """
    try:
        logger.info("Fetching all collection names from MongoDB...")
        db = get_database()
        collection_names = db.list_collection_names()
        logger.info(f"Found {len(collection_names)} collections: {collection_names}")
        return collection_names
    except Exception as e:
        logger.error(f"Error fetching collection names: {e}")
        raise

def insert_collections_to_pinecone():
    """
    Main function to get all collection names and insert them into Pinecone.
    """
    try:
        # Connect to MongoDB
        logger.info("Connecting to MongoDB...")
        connect_to_mongo()
        
        # Connect to Pinecone
        logger.info("Connecting to Pinecone...")
        connect_to_pinecone()
        
        # Get all collection names
        collection_names = get_all_collection_names()
        
        if not collection_names:
            logger.warning("No collections found in database")
            return
        
        # Insert each collection name as a vector
        logger.info(f"Starting to insert {len(collection_names)} collections into Pinecone...")
        
        for idx, collection_name in enumerate(collection_names, 1):
            try:
                # Generate embedding for collection name
                vector = embed_query(collection_name)
                
                # Create metadata
                metadata = {
                    "collection_name": collection_name,
                    "type": "collection",
                    "inserted_at": str(idx)
                }
                
                # Insert into Pinecone with collection name as ID
                vector_id = f"collection_{collection_name}"
                insert_vector(id=vector_id, vector=vector, metadata=metadata)
                
                logger.info(f"[{idx}/{len(collection_names)}] Inserted '{collection_name}' successfully")
                
            except Exception as e:
                logger.error(f"Failed to insert collection '{collection_name}': {e}")
                continue
        
        logger.info("Successfully completed inserting all collections into Pinecone")
        
    except Exception as e:
        logger.error(f"Error in insert_collections_to_pinecone: {e}")
        raise

def search_similar_collections(query_text: str, top_k: int = 5):
    """
    Search for similar collection names in Pinecone.
    """
    try:
        logger.info(f"Searching for collections similar to: {query_text}")
        
        # Generate embedding for query
        query_vector = embed_query(query_text)
        
        # Query Pinecone
        results = query_vector(vector=query_vector, top_k=top_k)
        
        logger.info(f"Found {len(results)} similar collections")
        
        # Display results
        for match in results:
            collection_name = match.metadata.get('collection_name', 'Unknown')
            score = match.score
            logger.info(f"Collection: {collection_name}, Similarity: {score:.4f}")
    except Exception as e:
        print(e)