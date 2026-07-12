#!/usr/bin/env bash
# Setup script for the Hallucination Detection project

set -e

echo "=== Setting up Hallucination Detection Framework ==="

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed data/vector_store models

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from template"
fi

# Download NLTK data (for sentence tokenization)
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Initialize the database
python -c "
from core.database import DatabaseManager
db = DatabaseManager()
print('Database initialized successfully')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the system:"
echo "  1. Activate virtualenv: source venv/bin/activate"
echo "  2. Start backend: uvicorn api.main:app --reload --port 8000"
echo "  3. Start frontend: streamlit run frontend/app.py"
echo ""
echo "Optional: Install Ollama for local LLM support:"
echo "  curl -fsSL https://ollama.com/install.sh | sh"
echo "  ollama pull mistral"
