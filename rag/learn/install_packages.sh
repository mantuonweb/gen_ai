#!/bin/bash

echo "=== Installing Python Packages for Resume RAG System ==="
echo ""

# Update pip
echo "📦 Updating pip..."
python3 -m pip install --upgrade pip

# Install required packages
echo ""
echo "📦 Installing sentence-transformers..."
python3 -m pip install sentence-transformers

echo ""
echo "📦 Installing numpy..."
python3 -m pip install numpy

echo ""
echo "📦 Installing openai..."
python3 -m pip install openai

echo ""
echo "📦 Installing PyPDF2 (for PDF support)..."
python3 -m pip install PyPDF2

echo ""
echo "✅ All packages installed successfully!"
echo ""
echo "Verifying installations..."
python3 -c "import sentence_transformers; print('✓ sentence-transformers:', sentence_transformers.__version__)"
python3 -c "import numpy; print('✓ numpy:', numpy.__version__)"
python3 -c "import openai; print('✓ openai:', openai.__version__)"
python3 -c "import PyPDF2; print('✓ PyPDF2:', PyPDF2.__version__)"

echo ""
echo "🎉 Setup complete! You can now run the RAG system."