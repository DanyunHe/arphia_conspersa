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

echo "This script will automatically download the required data files from UCLA Box."
echo "Note: Large files may take time to download."
echo ""

# Check if wget or curl is available
if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
    echo -e "${RED}Error: Neither wget nor curl is installed.${NC}"
    echo "Please install wget or curl to proceed."
    exit 1
fi

# Check if unzip is available
if ! command -v unzip &> /dev/null; then
    echo -e "${RED}Error: unzip is not installed.${NC}"
    echo "Please install unzip to proceed."
    exit 1
fi

# Create directories
mkdir -p data/deeplabcut_whole
mkdir -p ~/.cellpose/models
mkdir -p data/test_images

echo "========================================"
echo "1. DeepLabCut Model"
echo "========================================"
echo "Downloading from UCLA Box..."
echo "Target: data/deeplabcut_whole/"
echo ""
if wget --content-disposition -O data/deeplabcut_whole.zip "https://ucla.box.com/shared/static/k7ee8iyhqhzo3g30ak888sny9ssrsycc.zip" 2>&1 || \
   curl -L -o data/deeplabcut_whole.zip "https://ucla.box.com/shared/static/k7ee8iyhqhzo3g30ak888sny9ssrsycc.zip" 2>&1; then
    echo "Extracting files..."
    unzip -q -o data/deeplabcut_whole.zip -d data/
    rm data/deeplabcut_whole.zip
    echo -e "${GREEN}✓ DeepLabCut model downloaded successfully${NC}"
else
    echo -e "${YELLOW}⚠ Automatic download failed. Manual download may be required.${NC}"
    echo "  URL: https://ucla.box.com/s/k7ee8iyhqhzo3g30ak888sny9ssrsycc"
fi

echo ""
echo "========================================"
echo "2. Quick Start Example Images"
echo "========================================"
echo "Downloading example images from UCLA Box..."
echo "Target: data/test_images/"
echo ""
if wget --content-disposition -O data/test_images.zip "https://ucla.box.com/shared/static/d7jldigc4r6ej7xsdbq587mxdltfzc3j.zip" 2>&1 || \
   curl -L -o data/test_images.zip "https://ucla.box.com/shared/static/d7jldigc4r6ej7xsdbq587mxdltfzc3j.zip" 2>&1; then
    echo "Extracting files..."
    unzip -q -o data/test_images.zip -d data/
    rm data/test_images.zip
    echo -e "${GREEN}✓ Example images downloaded successfully${NC}"
else
    echo -e "${YELLOW}⚠ Automatic download failed. Manual download may be required.${NC}"
    echo "  URL: https://ucla.box.com/s/d7jldigc4r6ej7xsdbq587mxdltfzc3j"
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
echo "Downloading Cellpose model from UCLA Box..."
echo "Target: ~/.cellpose/models/"
echo ""
if wget --content-disposition -O ~/.cellpose/models/CP_20230503_151910 "https://ucla.box.com/shared/static/9orv40s9r4xzdlg1g93a1e7fzbob0j3k" 2>&1 || \
   curl -L -o ~/.cellpose/models/CP_20230503_151910 "https://ucla.box.com/shared/static/9orv40s9r4xzdlg1g93a1e7fzbob0j3k" 2>&1; then
    echo -e "${GREEN}✓ Cellpose model downloaded successfully${NC}"
else
    echo -e "${YELLOW}⚠ Automatic download failed. Manual download may be required.${NC}"
    echo "  URL: https://ucla.box.com/s/9orv40s9r4xzdlg1g93a1e7fzbob0j3k"
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
TEST_IMAGES=$(ls data/images-selected/*.dng 2>/dev/null | wc -l)
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
