from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import aiofiles
from .models import SearchQuery, SearchResponse, SearchResult, UploadResponse, StatusResponse
from .rag_engine import RAGEngine

app = FastAPI(
    title="RAG Resume Microservice",
    description="Upload resumes and perform intelligent search using RAG + Ollama",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Engine with Ollama
rag_engine = RAGEngine(llm_model="llama3.2")
UPLOAD_DIR = "uploads"
STATE_FILE = "rag_state.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Load existing state on startup"""
    print("\n" + "="*50)
    print("🚀 Starting RAG Resume Microservice (Ollama)")
    print("="*50)
    
    # Check Ollama connection
    if rag_engine.check_ollama_connection():
        print("✅ Ollama is running")
        models = rag_engine.list_available_models()
        print(f"📦 Available models: {', '.join(models)}")
    else:
        print("⚠️  Ollama not running - AI answers will be disabled")
        print("   Start Ollama with: ollama serve")
    
    if os.path.exists(STATE_FILE):
        rag_engine.load_state(STATE_FILE)
    
    print(f"✅ Service ready with {len(rag_engine.resumes)} resumes")
    print("="*50 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Save state on shutdown"""
    rag_engine.save_state(STATE_FILE)
    print("👋 Service stopped")

@app.get("/", response_model=StatusResponse)
async def root():
    """Service status"""
    return StatusResponse(
        service="RAG Resume Microservice (Ollama)",
        status="running",
        total_resumes=len(rag_engine.resumes),
        embedding_model="all-MiniLM-L6-v2",
        llm_model=rag_engine.llm_model
    )

@app.post("/upload", response_model=UploadResponse)
async def upload_resume(file: UploadFile = File(...)):
    """Upload a resume file"""
    try:
        if not file.filename.endswith('.txt'):
            raise HTTPException(400, "Only .txt files allowed for now")
        
        resume_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{resume_id}_{file.filename}")
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        text_content = content.decode('utf-8')
        rag_engine.add_resume(resume_id, text_content, file.filename)
        rag_engine.save_state(STATE_FILE)
        
        return UploadResponse(
            id=resume_id,
            filename=file.filename,
            message="Resume uploaded and indexed successfully"
        )
    
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_resumes(query: SearchQuery):
    """Search resumes using RAG"""
    try:
        results = rag_engine.search(query.query, query.top_k)
        
        answer = None
        if query.generate_answer and results:
            answer = rag_engine.generate_answer(query.query, results)
        
        return SearchResponse(
            query=query.query,
            results=[SearchResult(**r) for r in results],
            answer=answer,
            total_resumes=len(rag_engine.resumes)
        )
    
    except Exception as e:
        raise HTTPException(500, f"Search failed: {str(e)}")

@app.get("/resumes")
async def list_resumes():
    """List all uploaded resumes"""
    return {
        "total": len(rag_engine.resumes),
        "resumes": rag_engine.metadata
    }

@app.delete("/resumes/{resume_id}")
async def delete_resume(resume_id: str):
    """Delete a resume"""
    try:
        for i, meta in enumerate(rag_engine.metadata):
            if meta['id'] == resume_id:
                deleted_filename = rag_engine.metadata[i]['filename']
                rag_engine.resumes.pop(i)
                rag_engine.metadata.pop(i)
                
                if rag_engine.resumes:
                    rag_engine.embeddings = rag_engine.embedding_model.encode(rag_engine.resumes)
                else:
                    rag_engine.embeddings = None
                
                rag_engine.save_state(STATE_FILE)
                return {
                    "message": f"Resume '{deleted_filename}' deleted successfully",
                    "remaining": len(rag_engine.resumes)
                }
        
        raise HTTPException(404, "Resume not found")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ollama_status = rag_engine.check_ollama_connection()
    return {
        "status": "healthy",
        "resumes_loaded": len(rag_engine.resumes),
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": rag_engine.llm_model,
        "ollama_running": ollama_status
    }

@app.get("/models")
async def list_models():
    """List available Ollama models"""
    models = rag_engine.list_available_models()
    return {
        "current_model": rag_engine.llm_model,
        "available_models": models
    }