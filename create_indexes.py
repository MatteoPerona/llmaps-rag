from pymongo import MongoClient
from langchain_mongodb.index import create_fulltext_search_index
from config import MONGODB_URI, DB_NAME, COLLECTION_NAME, MONGODB_CLIENT_KWARGS

def create_indexes():
    # Connect to MongoDB
    client = MongoClient(MONGODB_URI, **MONGODB_CLIENT_KWARGS)
    collection = client[DB_NAME][COLLECTION_NAME]
    
    # Create vector search index
    vector_index = {
        "mappings": {
            "dynamic": True,
            "fields": {
                "embedding": {
                    "dimensions": 1536,
                    "similarity": "dotProduct",
                    "type": "knnVector"
                }
            }
        }
    }
    
    try:
        collection.create_search_index(
            "vector_index",
            vector_index
        )
        print("Vector search index created successfully")
    except Exception as e:
        print(f"Error creating vector index: {e}")
    
    # Create text search index
    try:
        create_fulltext_search_index(
            collection=collection,
            field="text",  # Adjust based on your field name
            index_name="search_index"
        )
        print("Full-text search index created successfully")
    except Exception as e:
        print(f"Error creating text index: {e}")

if __name__ == "__main__":
    create_indexes() 