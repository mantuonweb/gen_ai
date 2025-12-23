from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uuid
from app.rag_engine import RAGEngine
import os

app = FastAPI(title="Resume Management Microservice")

# Initialize RAG Engine
rag = RAGEngine()

# Load existing state if available
rag.load_state()

class SearchQuery(BaseModel):
    query: str
    top_k: int = 3

class ResumeResponse(BaseModel):
    id: str
    filename: str
    content: str

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    """Upload and store resume in FAISS vector database"""
    try:
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Generate unique ID
        resume_id = str(uuid.uuid4())
        
        # Add to RAG engine
        rag.add_resume(resume_id, text_content, file.filename)
        
        # Save state
        rag.save_state()
        
        return {
            "message": "Resume uploaded successfully",
            "id": resume_id,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading resume: {str(e)}")

@app.get("/resumes")
async def view_resumes():
    """View all stored resumes"""
    try:
        resumes = rag.get_all_resumes()
        return {
            "total": len(resumes),
            "resumes": resumes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching resumes: {str(e)}")

@app.post("/search")
async def search_resumes(query: SearchQuery):
    """Search resumes based on skills/query using similarity match"""
    try:
        results = rag.search(query.query, query.top_k)
        
        return {
            "query": query.query,
            "total_results": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching resumes: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Resume Management Microservice",
        "status": "running",
        "total_resumes": len(rag.resumes)
    }

@app.get("/stats")
async def get_stats():
    """Get statistics about the vector database"""
    return {
        "total_resumes": len(rag.resumes),
        "vector_dimension": rag.dimension,
        "index_size": rag.index.ntotal
    }