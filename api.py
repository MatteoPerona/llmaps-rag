from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from chat import build_rag_graph
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Initialize FastAPI app
app = FastAPI(
    title="LLMaps RAG API",
    description="API for intelligent store and location search using RAG",
    version="1.0.0"
)

# Define request/response models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

# Initialize the RAG graph
graph = build_rag_graph()

# Add this after creating the FastAPI app
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that processes questions and returns answers with sources.
    """
    try:
        # Invoke the graph with the question
        result = graph.invoke({
            "question": request.question
        })
        
        # Format sources
        sources = []
        for doc in result["context"]:
            sources.append({
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "Unknown source"),
                "score": doc.metadata.get("score", 0.0)
            })
        
        return ChatResponse(
            answer=result["answer"],
            sources=sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy"}

@app.get("/")
async def read_root():
    """
    Serve the chat interface
    """
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 