from typing import List, TypedDict
from langchain_core.documents import Document
from langchain import hub
from langgraph.graph import StateGraph, START
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers.hybrid_search import MongoDBAtlasHybridSearchRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pymongo import MongoClient
from config import MONGODB_URI, DB_NAME, COLLECTION_NAME, MONGODB_CLIENT_KWARGS

# Define the state of our RAG application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def setup_vectorstore():
    """Initialize and return the MongoDB vector store."""
    # Create vector store instance
    vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=MONGODB_URI,
        embedding=OpenAIEmbeddings(disallowed_special=()),
        namespace=f"{DB_NAME}.{COLLECTION_NAME}",
        text_key="text",  # Adjust based on your field name
        embedding_key="embedding",  # Adjust based on your field name
        relevance_score_fn="dotProduct"
    )
    return vectorstore

# Initialize components
vectorstore = setup_vectorstore()
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)

# Get the RAG prompt from LangChain hub
prompt = hub.pull("rlm/rag-prompt")

# Define the retrieval step
def retrieve(state: State):
    """Retrieve relevant documents using hybrid search."""
    print(f"\nSearching for: {state['question']}")
    
    try:
        # Initialize the hybrid search retriever
        retriever = MongoDBAtlasHybridSearchRetriever(
            vectorstore=vectorstore,
            search_index_name="default",  # Update with your search index name
            top_k=5,
            fulltext_penalty=50,
            vector_penalty=50
        )
        
        # Get documents using hybrid search
        retrieved_docs = retriever.invoke(state["question"])
        
        print(f"Found {len(retrieved_docs)} documents")
        if not retrieved_docs:
            print("No documents found in vector store!")
            
        return {"context": retrieved_docs}
    except Exception as e:
        print(f"Error during retrieval: {str(e)}")
        return {"context": []}

# Define the generation step
def generate(state: State):
    """Generate an answer using the retrieved documents."""
    # Combine all retrieved documents
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Format the prompt with question and context
    messages = prompt.invoke({
        "context": docs_content,
        "question": state["question"]
    }).to_messages()
    
    # Generate response
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build the graph
def build_rag_graph():
    """Build the RAG graph with retrieval and generation steps."""
    workflow = StateGraph(State)
    
    # Add the sequential steps
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    
    # Add the edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    
    # Compile the graph
    return workflow.compile()

def main():
    print("Initializing the RAG chatbot...")
    
    # Test vector store
    try:
        test_docs = vectorstore.similarity_search(
            "test query",
            k=1
        )
        print(f"Vector store test: found {len(test_docs)} documents")
    except Exception as e:
        print(f"Vector store test failed: {str(e)}")
    
    graph = build_rag_graph()
    
    print("\nWelcome to LLMaps! I can help you find information about stores and locations.")
    print("Type 'quit' to exit.\n")
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
            
        if question:
            try:
                # Invoke the graph with the question
                result = graph.invoke({
                    "question": question
                })
                
                print("\nBot:", result["answer"])
                
                # Debug information
                print("\nSources used:")
                for doc in result["context"]:
                    print("---")
                    print(f"From: {doc.metadata.get('source', 'Unknown source')}")
                    print(doc.page_content[:200] + "...")
                print()
                
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again.\n")

if __name__ == "__main__":
    main() 