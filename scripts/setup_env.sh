#!/bin/bash

# Environment Setup Script
# Sets up the complete development environment

set -e

echo "=================================="
echo "OCR-Agentic-RAG Environment Setup"
echo "=================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Error: Python 3.10+ required"
    exit 1
fi

echo "✓ Python version OK"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✓ Dependencies installed"

# Setup directories
echo ""
echo "Creating directory structure..."
mkdir -p data/images
mkdir -p data/evaluation/omnidocbench_sample
mkdir -p data/evaluation/sroie_sample
mkdir -p data/faiss_index
mkdir -p output
mkdir -p logs
mkdir -p models

echo "✓ Directories created"

# Create .env file
echo ""
echo "Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ Created .env file"
    echo ""
    echo "⚠ IMPORTANT: Edit .env and add your ANTHROPIC_API_KEY"
else
    echo "✓ .env file already exists"
fi

# Download models
echo ""
echo "Downloading models..."
bash scripts/download_models.sh

# Install system dependencies
echo ""
echo "Checking system dependencies..."

# Check for Tesseract
if command -v tesseract &> /dev/null; then
    echo "✓ Tesseract OCR is installed"
else
    echo "⚠ Tesseract OCR not found"
    echo "  Install with:"
    echo "    Ubuntu/Debian: sudo apt-get install tesseract-ocr"
    echo "    macOS: brew install tesseract"
    echo "    Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
fi

# Check for poppler (for PDF handling)
if command -v pdftoppm &> /dev/null; then
    echo "✓ Poppler is installed"
else
    echo "⚠ Poppler not found (needed for PDF processing)"
    echo "  Install with:"
    echo "    Ubuntu/Debian: sudo apt-get install poppler-utils"
    echo "    macOS: brew install poppler"
fi

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your ANTHROPIC_API_KEY"
echo "2. Add sample images to data/images/"
echo "3. Run demo: python examples/03_e2e_demo.py"
echo ""
echo "To activate environment:"
echo "  source venv/bin/activate"