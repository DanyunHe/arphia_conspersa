#!/bin/bash

# Grasshopper Wing Analysis Pipeline - Setup Script
# This script automates the installation of all dependencies

set -e  # Exit on error

echo "========================================="
echo "Grasshopper Wing Analysis Pipeline Setup"
echo "========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"
if ! python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
    echo -e "${RED}Error: Python 3.8 or higher is required. Current version: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version $PYTHON_VERSION detected${NC}"
echo ""

# Detect if running in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]] && [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo -e "${YELLOW}Warning: No virtual environment detected!${NC}"
    echo "It is highly recommended to use a virtual environment."
    echo "Create one with:"
    echo "  conda create -n wing python=3.10"
    echo "  conda activate wing"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Virtual environment detected${NC}"
    echo ""
fi

# Step 1: Install PyTorch
echo "----------------------------------------"
echo "Step 1: Installing PyTorch"
echo "----------------------------------------"
echo ""

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing PyTorch with CUDA support..."
    echo "Note: Installing for CUDA 11.8. If you need a different version, visit pytorch.org"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected. Installing CPU-only PyTorch..."
    pip install torch torchvision
fi
echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Step 2: Install main dependencies
echo "----------------------------------------"
echo "Step 2: Installing dependencies from requirements.txt"
echo "----------------------------------------"
echo ""
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 3: Install Segment Anything Model
echo "----------------------------------------"
echo "Step 3: Installing Segment Anything Model (SAM)"
echo "----------------------------------------"
echo ""
pip install git+https://github.com/facebookresearch/segment-anything.git
echo -e "${GREEN}✓ SAM installed${NC}"
echo ""

# Step 4: Install local sknw package
echo "----------------------------------------"
echo "Step 4: Installing local sknw package"
echo "----------------------------------------"
echo ""
if [ -d "source/sknw-master" ]; then
    cd source/sknw-master
    pip install -e .
    cd ../..
    echo -e "${GREEN}✓ sknw package installed${NC}"
else
    echo -e "${YELLOW}Warning: sknw-master directory not found. Skipping...${NC}"
fi
echo ""

# Step 5: Download SAM model checkpoint
echo "----------------------------------------"
echo "Step 5: Downloading SAM model checkpoint"
echo "----------------------------------------"
echo ""
SAM_MODEL_PATH="source/02_extraction/sam_vit_h_4b8939.pth"
if [ -f "$SAM_MODEL_PATH" ]; then
    echo "SAM model checkpoint already exists. Skipping download."
else
    echo "Downloading SAM ViT-H model checkpoint (2.4 GB)..."
    mkdir -p source/02_extraction
    cd source/02_extraction
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    cd ../..
    echo -e "${GREEN}✓ SAM checkpoint downloaded${NC}"
fi
echo ""

# Step 6: Create result directories
echo "----------------------------------------"
echo "Step 6: Creating result directories"
echo "----------------------------------------"
echo ""
mkdir -p result/01_find_pt
mkdir -p result/02_extraction
mkdir -p result/03_svd
mkdir -p result/04_segmentation
mkdir -p result/05_venation_network
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Step 7: Verify installation
echo "----------------------------------------"
echo "Step 7: Verifying installation"
echo "----------------------------------------"
echo ""

# Test imports
python -c "
import sys
errors = []

try:
    import numpy
    print('✓ numpy')
except ImportError:
    errors.append('numpy')
    print('✗ numpy')

try:
    import pandas
    print('✓ pandas')
except ImportError:
    errors.append('pandas')
    print('✗ pandas')

try:
    import cv2
    print('✓ opencv-python')
except ImportError:
    errors.append('opencv-python')
    print('✗ opencv-python')

try:
    import torch
    print('✓ torch')
except ImportError:
    errors.append('torch')
    print('✗ torch')

try:
    import tensorflow
    print('✓ tensorflow')
except ImportError:
    errors.append('tensorflow')
    print('✗ tensorflow')

try:
    import deeplabcut
    print('✓ deeplabcut')
except ImportError:
    errors.append('deeplabcut')
    print('✗ deeplabcut')

try:
    import cellpose
    print('✓ cellpose')
except ImportError:
    errors.append('cellpose')
    print('✗ cellpose')

try:
    import segment_anything
    print('✓ segment_anything')
except ImportError:
    errors.append('segment_anything')
    print('✗ segment_anything')

try:
    import tifffile
    print('✓ tifffile')
except ImportError:
    errors.append('tifffile')
    print('✗ tifffile')

try:
    import imageio
    print('✓ imageio')
except ImportError:
    errors.append('imageio')
    print('✗ imageio')

try:
    import openpyxl
    print('✓ openpyxl')
except ImportError:
    errors.append('openpyxl')
    print('✗ openpyxl')

if errors:
    print(f'\\nWarning: {len(errors)} package(s) failed to import')
    sys.exit(1)
else:
    print('\\nAll packages imported successfully!')
    sys.exit(0)
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Place your images in the data/ directory"
    echo "2. Navigate to source/ and run the pipeline steps"
    echo "3. See README.md for detailed usage instructions"
    echo ""
else
    echo ""
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}Installation completed with warnings${NC}"
    echo -e "${RED}=========================================${NC}"
    echo ""
    echo "Some packages failed to import. Please check the errors above."
    echo "You may need to manually install missing packages."
fi
