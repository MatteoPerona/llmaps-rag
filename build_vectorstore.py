import os
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from pypdf import PdfReader
from config import MONGODB_URI, DB_NAME, COLLECTION_NAME, MONGODB_CLIENT_KWARGS
import argparse
from langchain.schema import Document

def pdf_to_text(pdf_path):
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text_to_documents(text, source_name):
    """Splits store data content into smaller documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger chunks for store data
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    # Create proper Langchain Document objects with store-specific metadata
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": source_name,
                "type": "store_data"
            }
        ) 
        for chunk in chunks
    ]
    return documents

def clear_vectorstore():
    """Clears all documents from the MongoDB collection."""
    try:
        client = MongoClient(MONGODB_URI, **MONGODB_CLIENT_KWARGS)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        collection.delete_many({})
        print("Cleared all documents from the collection")
    except Exception as e:
        print(f"Error clearing vectorstore: {str(e)}")
        raise

def create_mongodb_vectorstore(documents, clear_existing=False):
    """Creates or updates a vector store in MongoDB using the provided documents."""
    try:
        print(f"Connecting to MongoDB...")
        client = MongoClient(MONGODB_URI, **MONGODB_CLIENT_KWARGS)
        
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        if clear_existing:
            collection.delete_many({})
            print("Cleared existing documents from collection")

        embeddings = OpenAIEmbeddings()

        # Create vector store with properly formatted documents
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents=documents,  # Documents are already properly formatted
            embedding=embeddings,
            collection=collection,
            index_name="default"
        )

        return vector_store
    except Exception as e:
        print(f"MongoDB connection error: {str(e)}")
        raise

def process_pdfs_for_rag(pdf_dir="raw-documents", clear_existing=False):
    """Process all PDFs in the specified directory for RAG."""
    all_documents = []
    
    # Ensure the directory exists
    if not os.path.exists(pdf_dir):
        raise ValueError(f"Directory {pdf_dir} does not exist")
    
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"Processing {pdf_file}...")
        
        # Convert PDF to text
        text = pdf_to_text(pdf_path)
        print(f"Extracted {len(text)} characters of text")
        
        # Split text into RAG documents
        documents = split_text_to_documents(text, source_name=pdf_file)
        print(f"Created {len(documents)} document chunks")
        all_documents.extend(documents)
    
    print(f"Total documents created: {len(all_documents)}")
    
    # Create or update vector store
    vector_store = create_mongodb_vectorstore(all_documents, clear_existing=clear_existing)
    
    return vector_store

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build or update vector store from PDF documents')
    parser.add_argument('--clear', action='store_true', help='Clear existing documents before adding new ones')
    parser.add_argument('--clear-only', action='store_true', help='Only clear the vector store without adding documents')
    parser.add_argument('--pdf-dir', type=str, default='raw-documents', help='Directory containing PDF files')
    
    args = parser.parse_args()
    
    try:
        if args.clear_only:
            clear_vectorstore()
            print("Vector store cleared successfully!")
        else:
            vector_store = process_pdfs_for_rag(
                pdf_dir=args.pdf_dir,
                clear_existing=args.clear
            )
            print("Vector store updated successfully!")
    except Exception as e:
        print(f"Error occurred: {str(e)}") 