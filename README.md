# LLMaps RAG

A Retrieval-Augmented Generation (RAG) system for LLMaps, designed to provide intelligent search and information retrieval for store and location data.

## Features
- Vector search capabilities using MongoDB Atlas
- Hybrid search combining semantic and keyword search
- API endpoints for querying store information
- Chat interface for natural language interactions

## Setup
1. Clone the repository
2. Create a `.env` file with your MongoDB and OpenAI credentials
3. Install dependencies: `pip install -r requirements.txt`
4. Create indexes: `python create_indexes.py`
5. Add your PDF documents to the `raw-documents` directory
6. Build the vector store: `python build_vectorstore.py`

Note: You may need to whitelist your IP address in MongoDB Atlas to allow connections. This can be done through the MongoDB Atlas dashboard under Network Access.

## Usage
- Start the API: `python api.py`
- Use the chat interface: `python chat.py`
