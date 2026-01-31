#!/bin/bash
# Complete setup script for local RAG

echo "🚀 Setting up Local RAG System"
echo "================================"
echo ""

# Step 1: Check if Ollama is installed
echo "📦 Step 1: Checking Ollama..."
if ! command -v ollama &>/dev/null; then
  echo "   Installing Ollama..."
  brew install ollama
else
  echo "   ✅ Ollama already installed"
fi

# Step 2: Start Ollama and pull model
echo ""
echo "🤖 Step 2: Setting up LLM..."
echo "   Starting Ollama service..."
ollama serve &>/dev/null &
OLLAMA_PID=$!
sleep 3

echo "   Pulling gemma2:27b model (this may take a few minutes)..."
ollama pull gemma2:27b

# Step 3: Set up Python environment
echo ""
echo "🐍 Step 3: Setting up Python environment..."
if [ ! -d "rag_env" ]; then
  echo "   Creating virtual environment..."
  python3 -m venv rag_env
fi

echo "   Activating virtual environment..."
source rag_env/bin/activate

echo "   Installing Python packages..."
pip install -q sentence-transformers chromadb PyPDF2

# Step 4: Test setup
echo ""
echo "✅ Setup complete!"
echo ""
echo "================================"
echo "Next steps:"
echo "================================"
echo ""
echo "1. Activate virtual environment:"
echo "   source rag_env/bin/activate"
echo ""
echo "2. Index your PDFs:"
echo "   python local_rag.py index ~/Documents/Textbooks"
echo ""
echo "3. Ask questions:"
echo "   python local_rag.py interactive"
echo ""
echo "================================"
echo ""
echo "💡 Tips:"
echo "   - Indexing only needs to be done once"
echo "   - Database is saved in ./chroma_db/"
echo "   - Everything runs locally (no internet needed after setup)"
echo "   - Zero cost, completely private"
echo ""
