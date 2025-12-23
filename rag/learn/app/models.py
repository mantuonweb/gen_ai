from pydantic import BaseModel
from typing import List, Optional

class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 3
    generate_answer: Optional[bool] = True

class SearchResult(BaseModel):
    id: str
    filename: str
    score: float
    content: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    answer: Optional[str] = None
    total_resumes: int

class UploadResponse(BaseModel):
    id: str
    filename: str
    message: str

class StatusResponse(BaseModel):
    service: str
    status: str
    total_resumes: int
    embedding_model: str
    llm_model: str