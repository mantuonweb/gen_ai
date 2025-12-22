import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from typing import List, Dict
import json
import os

class RAGEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2", llm_model="llama3.2"):
        print(f"🔄 Initializing RAG Engine...")
        self.embedding_model = SentenceTransformer(model_name)
        self.llm_model = llm_model
        self.resumes = []
        self.embeddings = None
        self.metadata = []
        print(f"✅ RAG Engine initialized")
        
    def add_resume(self, resume_id: str, content: str, filename: str):
        """Add a resume to the vector store"""
        self.resumes.append(content)
        self.metadata.append({
            'id': resume_id,
            'filename': filename
        })
        # Recreate embeddings
        self.embeddings = self.embedding_model.encode(self.resumes)
        print(f"✅ Added resume: {filename} (Total: {len(self.resumes)})")
        
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant resumes"""
        if not self.resumes:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'id': self.metadata[idx]['id'],
                'filename': self.metadata[idx]['filename'],
                'content': self.resumes[idx],
                'score': float(similarities[idx])
            })
        
        return results
    
    def generate_answer(self, query: str, relevant_resumes: List[Dict]) -> str:
        """Generate AI answer based on retrieved resumes"""
        if not relevant_resumes:
            return "No relevant resumes found."
        
        # Prepare context
        context = "\n\n---\n\n".join([
            f"Resume: {r['filename']}\n{r['content'][:500]}" 
            for r in relevant_resumes
        ])
        
        prompt = f"""Based on the following resumes, answer this question: {query}

Resumes:
{context}

Provide a clear, concise answer based only on the information in these resumes."""
        
        try:
            # Call Ollama
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': 'You are an HR assistant analyzing resumes. Be concise and factual.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            ollama.list()
            return True
        except Exception as e:
            print(f"⚠️ Ollama connection failed: {str(e)}")
            return False
    
    def list_available_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            result = ollama.list()
            # Handle different response formats
            if isinstance(result, dict) and 'models' in result:
                return [model.get('model', model.get('name', 'unknown')) for model in result['models']]
            return []
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            return []
    
    def save_state(self, filepath: str):
        """Save vector store to disk"""
        state = {
            'resumes': self.resumes,
            'metadata': self.metadata,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else None
        }
        with open(filepath, 'w') as f:
            json.dump(state, f)
        print(f"💾 Saved state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load vector store from disk"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                state = json.load(f)
                self.resumes = state['resumes']
                self.metadata = state['metadata']
                if state['embeddings']:
                    self.embeddings = np.array(state['embeddings'])
            print(f"📂 Loaded {len(self.resumes)} resumes from {filepath}")