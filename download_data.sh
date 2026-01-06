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
mkdir -p data/DNG

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
echo "Downloading population wing images from UCLA Box..."
echo "Target: data/DNG/"
echo ""

# Download first population dataset
echo "Downloading first population dataset..."
if wget --content-disposition -O data/DNG/temp_population_1.zip "https://ucla.box.com/shared/static/6ld5szlnfjkmyp3rap2duzc3z4fbjrhr.zip" 2>&1 || \
   curl -L -o data/DNG/temp_population_1.zip "https://ucla.box.com/shared/static/6ld5szlnfjkmyp3rap2duzc3z4fbjrhr.zip" 2>&1; then
    echo "Extracting first population dataset..."
    unzip -q -o data/DNG/temp_population_1.zip -d data/DNG/
    rm data/DNG/temp_population_1.zip
    echo -e "${GREEN}✓ First population dataset downloaded successfully${NC}"
else
    echo -e "${YELLOW}⚠ First population download failed. Manual download may be required.${NC}"
    echo "  URL: https://ucla.box.com/s/6ld5szlnfjkmyp3rap2duzc3z4fbjrhr"
fi

echo ""
# Download second population dataset
echo "Downloading second population dataset..."
if wget --content-disposition -O data/DNG/temp_population_2.zip "https://ucla.box.com/shared/static/86evgi78chetsga4e04ebgd6dwqknyko.zip" 2>&1 || \
   curl -L -o data/DNG/temp_population_2.zip "https://ucla.box.com/shared/static/86evgi78chetsga4e04ebgd6dwqknyko.zip" 2>&1; then
    echo "Extracting second population dataset..."
    unzip -q -o data/DNG/temp_population_2.zip -d data/DNG/
    rm data/DNG/temp_population_2.zip
    echo -e "${GREEN}✓ Second population dataset downloaded successfully${NC}"
else
    echo -e "${YELLOW}⚠ Second population download failed. Manual download may be required.${NC}"
    echo "  URL: https://ucla.box.com/s/86evgi78chetsga4e04ebgd6dwqknyko"
fi

echo ""
echo "========================================"
echo "3. SAM Model"
echo "========================================"
echo "Downloading SAM model from UCLA Box..."
echo "Target: data/SAM_model/"
echo ""
if wget --content-disposition -O data/sam_model.zip "https://ucla.box.com/shared/static/sx20teqlvuoqhchi810olma8rtwqo6gh.zip" 2>&1 || \
   curl -L -o data/sam_model.zip "https://ucla.box.com/shared/static/sx20teqlvuoqhchi810olma8rtwqo6gh.zip" 2>&1; then
    echo "Extracting files..."
    unzip -q -o data/sam_model.zip -d data/
    rm data/sam_model.zip
    echo -e "${GREEN}✓ SAM model downloaded successfully${NC}"
else
    echo -e "${YELLOW}⚠ Automatic download failed. Manual download may be required.${NC}"
    echo "  URL: https://ucla.box.com/s/sx20teqlvuoqhchi810olma8rtwqo6gh"
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
if [ -f "data/SAM_model/fine_tuned_sam_im1b.pth" ] || [ -f "data/SAM_model/sam_vit_h_4b8939.pth" ]; then
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

# Check DNG population images
DNG_DIRS=$(find data/DNG -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [ "$DNG_DIRS" -gt 0 ]; then
    DNG_COUNT=$(find data/DNG -name "*.dng" -o -name "*.DNG" 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Found $DNG_DIRS population folder(s) with $DNG_COUNT DNG images${NC}"
else
    echo -e "${RED}✗ No DNG population folders found${NC}"
fi

echo ""
echo "========================================"
echo "Download process complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Check available populations: ls data/DNG/"
echo "2. Convert DNG images to PNG: cd source/00_prepare_data && python convert_dng_to_png.py"
echo "3. Run the pipeline from source/ directory"
echo ""
