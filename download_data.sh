#!/bin/bash
# Automatic download script for wing analysis pipeline data
# This script automatically downloads required models and test data from web sources

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

echo "This script will automatically download the required data files."
echo "Note: Large files may take time. Some downloads may require authentication."
echo ""

# Create directories
mkdir -p data/deeplabcut_whole
mkdir -p ~/.cellpose/models
mkdir -p data/test_images

echo "========================================"
echo "1. DeepLabCut Model"
echo "========================================"
echo "Downloading from Google Drive..."
echo "Target: data/deeplabcut_whole/"
echo ""
if gdown --folder https://drive.google.com/drive/folders/1pfFGtV4hHQSBNwLjq10zhs3KKi13phm0 -O data/deeplabcut_whole/ --remaining-ok; then
    echo -e "${GREEN}✓ DeepLabCut model downloaded successfully${NC}"
else
    echo -e "${YELLOW}⚠ Automatic download failed. Manual download may be required.${NC}"
    echo "  URL: https://drive.google.com/drive/folders/1pfFGtV4hHQSBNwLjq10zhs3KKi13phm0"
fi

echo ""
echo "========================================"
echo "2. Wing Test Images"
echo "========================================"
echo "Downloading test images from Google Drive..."
echo "Target: data/test_images/"
echo ""
if gdown --folder https://drive.google.com/drive/folders/1lRfwuUhhVadkfz2ixvTbiqTruDvx72Kq -O data/test_images/ --remaining-ok; then
    echo -e "${GREEN}✓ Test images downloaded successfully${NC}"
else
    echo -e "${YELLOW}⚠ Automatic download failed. Manual download may be required.${NC}"
    echo "  URL: https://drive.google.com/drive/folders/1lRfwuUhhVadkfz2ixvTbiqTruDvx72Kq"
fi

echo ""
echo "========================================"
echo "3. SAM Model"
echo "========================================"
echo "Downloading SAM model from UCLA Box..."
echo "Target: source/02_extraction/fine_tuned_sam_im1b.pth"
echo ""
mkdir -p source/02_extraction
if wget -O source/02_extraction/fine_tuned_sam_im1b.pth "https://ucla.box.com/shared/static/pvdivl59ttxjg50f9s2pt8hu9z972jbn.pth" 2>/dev/null || \
   curl -L -o source/02_extraction/fine_tuned_sam_im1b.pth "https://ucla.box.com/shared/static/pvdivl59ttxjg50f9s2pt8hu9z972jbn.pth" 2>/dev/null; then
    echo -e "${GREEN}✓ SAM model downloaded successfully${NC}"
else
    echo -e "${YELLOW}⚠ Automatic download failed. Manual download may be required.${NC}"
    echo "  URL: https://ucla.box.com/s/pvdivl59ttxjg50f9s2pt8hu9z972jbn"
fi

echo ""
echo "========================================"
echo "4. Cellpose Model"
echo "========================================"
echo "Downloading Cellpose model from Google Drive..."
echo "Target: ~/.cellpose/models/"
echo ""
if gdown --folder https://drive.google.com/drive/folders/1KuuNEO-jhqwLQLR2-17uaZi7A8OlvBex -O ~/.cellpose/models/ --remaining-ok; then
    echo -e "${GREEN}✓ Cellpose model downloaded successfully${NC}"
else
    echo -e "${YELLOW}⚠ Automatic download failed. Manual download may be required.${NC}"
    echo "  URL: https://drive.google.com/drive/folders/1KuuNEO-jhqwLQLR2-17uaZi7A8OlvBex"
fi

echo ""
echo "========================================"
echo "5. Verification"
echo "========================================"
echo ""

# Check SAM model
if [ -f "source/02_extraction/fine_tuned_sam_im1b.pth" ]; then
    echo -e "${GREEN}✓ SAM model found${NC}"
else
    echo -e "${RED}✗ SAM model missing${NC}"
fi

# Check Cellpose model
if [ -f "$HOME/.cellpose/models/CP_20230503_151910" ]; then
    echo -e "${GREEN}✓ Cellpose model found${NC}"
else
    echo -e "${YELLOW}⚠ Cellpose model not found${NC}"
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
