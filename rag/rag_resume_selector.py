import os
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Step 1: Load and Read Resumes
def load_resumes(folder_path):
    """Load all resume text files from a folder"""
    resumes = []
    filenames = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                resumes.append(content)
                filenames.append(filename)
    
    print(f"✓ Loaded {len(resumes)} resumes")
    return resumes, filenames


# Step 2: Convert Resumes to Embeddings (Vector Representations)
def create_embeddings(resumes):
    """Convert text to numerical vectors for similarity search"""
    print("Creating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model
    embeddings = model.encode(resumes)
    print(f"✓ Created embeddings with shape: {embeddings.shape}")
    return model, embeddings


# Step 3: Search for Relevant Resumes
def search_resumes(query, model, embeddings, resumes, filenames, top_k=2):
    """Find most relevant resumes based on query"""
    print(f"\nSearching for: '{query}'")
    
    # Convert query to embedding
    query_embedding = model.encode([query])[0]
    
    # Calculate similarity scores (cosine similarity)
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get top matches
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'filename': filenames[idx],
            'content': resumes[idx],
            'score': similarities[idx]
        })
        print(f"  - {filenames[idx]} (score: {similarities[idx]:.3f})")
    
    return results


# Step 4: Generate Answer using AI
def generate_answer(query, relevant_resumes, api_key):
    """Use OpenAI to generate answer based on retrieved resumes"""
    
    # Prepare context from retrieved resumes
    context = "\n\n---\n\n".join([
        f"Resume: {r['filename']}\n{r['content']}" 
        for r in relevant_resumes
    ])
    
    # Create prompt
    prompt = f"""Based on the following resumes, answer this question: {query}

Resumes:
{context}

Please provide a clear, concise answer based only on the information in these resumes."""

    # Call OpenAI API
    client = OpenAI(api_key=api_key)
    
    print("\nGenerating AI response...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful HR assistant analyzing resumes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    
    return response.choices[0].message.content


# Main RAG Pipeline
def main():
    print("=== Resume RAG System ===\n")
    
    # Configuration
    RESUME_FOLDER = "resumes"
    OPENAI_API_KEY = "your-api-key-here"  # Replace with your actual API key
    
    # Step 1: Load resumes
    resumes, filenames = load_resumes(RESUME_FOLDER)
    
    # Step 2: Create embeddings
    model, embeddings = create_embeddings(resumes)
    
    # Step 3 & 4: Query loop
    while True:
        print("\n" + "="*50)
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        
        if query.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        # Retrieve relevant resumes
        relevant_resumes = search_resumes(query, model, embeddings, resumes, filenames, top_k=2)
        
        # Generate answer
        answer = generate_answer(query, relevant_resumes, OPENAI_API_KEY)
        
        print("\n📝 Answer:")
        print(answer)


if __name__ == "__main__":
    main()