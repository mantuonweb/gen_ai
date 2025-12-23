import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from typing import List, Dict, Optional
import json
import os
import faiss
import pickle

class RAGEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2", llm_model="llama3.2"):
        print(f"🔄 Initializing RAG Engine with FAISS...")
        self.embedding_model = SentenceTransformer(model_name)
        self.llm_model = llm_model
        self.dimension = 384  # all-MiniLM-L6-v2 embedding size
        self.index = faiss.IndexFlatL2(self.dimension)
        self.resumes = []
        self.metadata = []
        print(f"✅ RAG Engine initialized with FAISS")
        
    def add_resume(self, resume_id: str, content: str, filename: str):
        """Add a resume to FAISS vector store"""
        # Generate embedding
        embedding = self.embedding_model.encode([content])[0]
        
        # Add to FAISS index
        self.index.add(np.array([embedding], dtype=np.float32))
        
        # Store resume and metadata
        self.resumes.append(content)
        self.metadata.append({
            'id': resume_id,
            'filename': filename
        })
        
        print(f"✅ Added resume: {filename} (Total: {len(self.resumes)})")
        
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant resumes based on skills/query"""
        if not self.resumes:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        query_vector = np.array(query_embedding, dtype=np.float32)
        
        # Search in FAISS
        distances, indices = self.index.search(query_vector, min(top_k, len(self.resumes)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.resumes):
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 / (1 + distances[0][i])
                results.append({
                    'id': self.metadata[idx]['id'],
                    'filename': self.metadata[idx]['filename'],
                    'content': self.resumes[idx],
                    'score': float(similarity)
                })
        
        return results
    
    def get_all_resumes(self) -> List[Dict]:
        """Get all stored resumes"""
        return [
            {
                'id': meta['id'],
                'filename': meta['filename'],
                'content': content
            }
            for meta, content in zip(self.metadata, self.resumes)
        ]
    
    def save_state(self, filepath: str = "data/rag_state.pkl"):
        """Save FAISS index and data to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, filepath.replace('.pkl', '.faiss'))
        
        # Save metadata and resumes
        state = {
            'resumes': self.resumes,
            'metadata': self.metadata
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"💾 Saved state to {filepath}")
    
    def load_state(self, filepath: str = "data/rag_state.pkl"):
        """Load FAISS index and data from disk"""
        faiss_path = filepath.replace('.pkl', '.faiss')
        
        if os.path.exists(filepath) and os.path.exists(faiss_path):
            # Load FAISS index
            self.index = faiss.read_index(faiss_path)
            
            # Load metadata and resumes
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                self.resumes = state['resumes']
                self.metadata = state['metadata']
            
            print(f"📂 Loaded {len(self.resumes)} resumes from {filepath}")
            return True
        return False