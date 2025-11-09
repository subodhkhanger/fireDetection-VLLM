#!/bin/bash

# Fire-ViT Quick Start Script
# This script sets up the environment and runs tests

echo "=================================================="
echo "Fire-ViT Quick Start"
echo "=================================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "fire_vit_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv fire_vit_env
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source fire_vit_env/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."
echo ""

# Install PyTorch (CPU version for testing, adjust for GPU)
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install --quiet -r requirements.txt

echo ""
echo "✓ All dependencies installed"
echo ""

# Run tests
echo "=================================================="
echo "Running Fire-ViT Test Suite"
echo "=================================================="
echo ""

python test_model.py

# Check test result
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Setup Complete!"
    echo "=================================================="
    echo ""
    echo "Next steps:"
    echo "  1. Activate the environment: source fire_vit_env/bin/activate"
    echo "  2. Review README.md for usage examples"
    echo "  3. Check SKILLS.md for implementation guide"
    echo "  4. Download D-Fire dataset (see SKILLS.md)"
    echo "  5. Implement training pipeline"
    echo ""
else
    echo ""
    echo "❌ Tests failed. Please check the errors above."
    exit 1
fi
