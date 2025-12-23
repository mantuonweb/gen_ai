#!/bin/bash

echo "🚀 Starting RAG Resume Microservice (Local)"
echo "=========================================="

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama not running. Starting Ollama..."
    ollama serve &
    sleep 3
fi

# Check if model is downloaded
if ! ollama list | grep -q "llama3.2"; then
    echo "📥 Downloading llama3.2 model..."
    ollama pull llama3.2
fi

echo "✅ Ollama ready"
echo "🌐 Starting FastAPI server..."
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000