uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pip install -r requirements.txt
ollama serve &
ollama pull llama3.2
uvicorn app.main:app --reload --port 8000



ollama serve
brew install ollama
ollama pull llama3.2

