#!/bin/bash
# Download script for wing analysis pipeline data
# This script helps download required models and test data

set -e

echo "========================================"
echo "Wing Analysis Pipeline - Data Download"
echo "========================================"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo -e "${YELLOW}gdown not found. Installing...${NC}"
    pip install gdown
fi

echo "This script will help download the required data files."
echo "Note: Some downloads may require Google Drive authentication."
echo ""

# Create directories
mkdir -p data/deeplabcut_whole
mkdir -p ~/.cellpose/models
mkdir -p data/test_images

echo "========================================"
echo "1. DeepLabCut Model"
echo "========================================"
echo "Download from: https://drive.google.com/drive/folders/1pfFGtV4hHQSBNwLjq10zhs3KKi13phm0"
echo "Target: data/deeplabcut_whole/"
echo ""
echo "Manual steps:"
echo "1. Open the link above in your browser"
echo "2. Download the folder contents"
echo "3. Extract to data/deeplabcut_whole/"
echo ""
read -p "Press Enter when download is complete..."

echo ""
echo "========================================"
echo "2. Wing Test Images"
echo "========================================"
echo "Download from: https://drive.google.com/drive/folders/1lRfwuUhhVadkfz2ixvTbiqTruDvx72Kq"
echo "Target: data/test_images/"
echo ""
echo "Manual steps:"
echo "1. Open the link above in your browser"
echo "2. Download a few test images (DNG format)"
echo "3. Save to data/test_images/"
echo ""
read -p "Press Enter when download is complete..."

echo ""
echo "========================================"
echo "3. Verification"
echo "========================================"
echo ""

# Check SAM model
if [ -f "source/02_extraction/fine_tuned_sam_im1b.pth" ]; then
    echo -e "${GREEN}✓ SAM model found${NC}"
else
    echo -e "${RED}✗ SAM model missing${NC}"
    echo "  Download from: https://ucla.box.com/s/pvdivl59ttxjg50f9s2pt8hu9z972jbn"
    echo "  Place in: source/02_extraction/fine_tuned_sam_im1b.pth"
fi

# Check Cellpose model
if [ -f "$HOME/.cellpose/models/CP_20230503_151910" ]; then
    echo -e "${GREEN}✓ Cellpose model found${NC}"
else
    echo -e "${YELLOW}⚠ Cellpose model not found${NC}"
    echo "  Download from: https://drive.google.com/drive/folders/1KuuNEO-jhqwLQLR2-17uaZi7A8OlvBex"
    echo "  Place in: ~/.cellpose/models/"
fi

# Check DeepLabCut
if [ -d "data/deeplabcut_whole/segment_whole-dy-2024-06-06" ]; then
    echo -e "${GREEN}✓ DeepLabCut model found${NC}"
else
    echo -e "${RED}✗ DeepLabCut model not found${NC}"
fi

# Check test images
TEST_IMAGES=$(ls data/test_images/*.dng 2>/dev/null | wc -l)
if [ "$TEST_IMAGES" -gt 0 ]; then
    echo -e "${GREEN}✓ Found $TEST_IMAGES test images${NC}"
else
    echo -e "${RED}✗ No test images found${NC}"
fi

echo ""
echo "========================================"
echo "Download process complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Convert DNG images to PNG: cd source/00_prepare_data && python convert_dng_to_png.py --input_dir ../../data/test_images --output_dir ../../data"
echo "2. Run the pipeline from source/ directory"
echo ""
