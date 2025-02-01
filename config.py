import os
from dotenv import load_dotenv
from pymongo.server_api import ServerApi

# Load environment variables
load_dotenv()

# MongoDB configurations
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "stores_db")  # Default to stores_db
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "llmaps")  # Default to llmaps

# MongoDB client configuration with server API version 1
MONGODB_CLIENT_KWARGS = {
    "server_api": ServerApi('1'),
    "retryWrites": True,
    "w": "majority"
} 