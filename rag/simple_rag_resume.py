import os
import numpy as np
from sentence_transformers import SentenceTransformer

def load_resumes(folder_path):
    """Load all resume text files"""
    resumes = []
    filenames = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                resumes.append(f.read())
                filenames.append(filename)
    
    return resumes, filenames

def search_resumes(query, resumes, filenames):
    """Search and display relevant resumes"""
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings
    resume_embeddings = model.encode(resumes)
    query_embedding = model.encode([query])[0]
    
    # Calculate similarities
    similarities = np.dot(resume_embeddings, query_embedding) / (
        np.linalg.norm(resume_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Sort by relevance
    ranked_indices = np.argsort(similarities)[::-1]
    
    print(f"\n🔍 Results for: '{query}'\n")
    for i, idx in enumerate(ranked_indices, 1):
        print(f"{i}. {filenames[idx]} (Relevance: {similarities[idx]:.2%})")
        print(f"{resumes[idx][:200]}...\n")

# Main
resumes, filenames = load_resumes("resumes")
print("Resume Search System Ready!\n")

while True:
    query = input("Search query (or 'quit'): ").strip()
    if query.lower() == 'quit':
        break
    search_resumes(query, resumes, filenames)