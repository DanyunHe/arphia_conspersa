# Grasshopper Wing Analysis Pipeline

Code for extracting and analyzing morphological information from grasshopper wing images (Arphia conspersa).

## Pipeline Overview

1. **Find Points** - Identify key regions (background, forewing, hindwing, body) using DeepLabCut
2. **Wing Extraction** - Separate forewing and hindwing using Segment Anything Model (SAM)
3. **Alignment & SVD** - Align images and apply SVD to emphasize wing skeleton
4. **Domain Segmentation** - Segment wing domains and veins using Cellpose
5. **Venation Network** - Convert masks into graph structure with vertices and edges

---

## Installation

### Requirements
- Python 3.10 (recommended) or 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM recommended

### Quick Setup (Recommended)

```bash
# 1. Create virtual environment
conda create -n wing python=3.10
conda activate wing

# 2. Clone repository
git clone https://github.com/DanyunHe/arphia_conspersa.git
cd arphia_conspersa

# 3. Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA
# OR for CPU-only: pip install torch torchvision

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Install C++ dependencies for alignment (Step 03)
conda install -c conda-forge gsl libtiff libpng

# 6. Install SAM and local packages
pip install git+https://github.com/facebookresearch/segment-anything.git
cd source/sknw-master && pip install -e . && cd ../..

# 7. Download pre-trained models
bash download_data.sh
```

**Note**: The download script will download all required pre-trained models automatically

### Manual Installation

If the automated setup fails:

```bash
# 1. Create environment
conda create -n wing python=3.10
conda activate wing

# 2. Install PyTorch (visit pytorch.org for your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA
# OR
pip install torch torchvision  # CPU only

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install C++ dependencies for alignment (Step 03)
conda install -c conda-forge gsl libtiff libpng

# 4a. Compile alignment code (Step 03)
cd source/03_svd/align_img && make clean && make && cd ../../..

# 5. Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# 6. Install local sknw package
cd source/sknw-master && pip install -e . && cd ../..

# 7. Download pre-trained models
# Download and place models in the appropriate directories:

# SAM model (automatically downloaded by download_data.sh)
# - Fine-tuned model (fine_tuned_sam_im1b.pth) is downloaded automatically
# - Download from: https://ucla.box.com/s/sx20teqlvuoqhchi810olma8rtwqo6gh
# - Extracts to: data/SAM_model/
# - Note: The code will also work with the default SAM model (sam_vit_h_4b8939.pth)
#   if you place it in data/SAM_model/

# Cellpose model (for domain segmentation)
# Download from: https://drive.google.com/drive/folders/1KuuNEO-jhqwLQLR2-17uaZi7A8OlvBex?usp=drive_link
# Place in: ~/.cellpose/models/

# DeepLabCut model (for keypoint detection)
# Download from: https://drive.google.com/drive/folders/1pfFGtV4hHQSBNwLjq10zhs3KKi13phm0?usp=drive_link
# Place in: data/deeplabcut_whole/

# 8. Verify installation
python -c "import torch, tensorflow, deeplabcut, cellpose, segment_anything; print('Success!')"
```

---

## Usage

See [`source/README.md`](source/README.md) for detailed command-line usage of each step.

### Quick Example

```bash
# Step 0: Convert DNG to PNG (if starting with raw .dng files)
cd source/00_prepare_data
python convert_dng_to_png.py
cd ../..

# Step 1: Find points (uses PNG from data/PNG/population_XX/, outputs to result/01_find_pt/)
# NOTE: The script automatically updates DeepLabCut config paths for your computer
cd source/01_find_pt
# Option A: Process all populations at once
python find_pt.py $HOME/arphia_conspersa

# Option B: Process specific population only
# python find_pt.py $HOME/arphia_conspersa population_34

cd ../..

# Step 2: Extract wings (reads DNG from data/DNG/population_XX/, outputs to result/02_extraction/)
cd source/02_extraction
# Option A: Process all populations at once
python extract_wings.py

# Option B: Process specific population only
# python extract_wings.py --population_id 34

# Option C: Process specific individual within a population
# python extract_wings.py --population_id 34 --individual_index 0

cd ../..

# Step 3: Alignment & SVD (reads from result/02_extraction/, outputs to result/03_svd/population_XX/)
cd source/03_svd
# Option A: Process all populations at once
python process_all_svd.py

# Option B: Process specific population only
# python process_all_svd.py --population_id 34

# Option C: Process specific wing
# python process_all_svd.py --wing_name population_34+FMNH_4669630

cd ../..

# Step 4: Segment domains (reads from result/03_svd/, outputs to result/04_segmentation/)
cd source/04_segmentation
# Option A: Process all populations
python segmentation.py

# Option B: Process specific population
# python segmentation.py --population_id 34

# Option C: Process specific wing
# python segmentation.py --wing_name population_34+FMNH_4669630
cd ../..

# Step 5: Venation network (reads from result/04_segmentation/, outputs to result/05_venation_network/)
cd source/05_venation_network
# Option A: Process all populations
python venation_network.py

# Option B: Process specific population
# python venation_network.py --population_id 34

# Option C: Process specific wing
# python venation_network.py --wing_name population_34+FMNH_4669630
cd ../..
```




### Data Preparation

**1. Create Data Directory**

```bash
mkdir -p data
```

**2. Download Required Data**

Use the provided download script (recommended):
```bash
bash download_data.sh
```

Or download manually:
- Wing images: https://drive.google.com/drive/folders/1lRfwuUhhVadkfz2ixvTbiqTruDvx72Kq?usp=drive_link
- DeepLabCut model: https://drive.google.com/drive/folders/1pfFGtV4hHQSBNwLjq10zhs3KKi13phm0?usp=drive_link
- Place DeepLabCut model in `data/deeplabcut_whole/`

**Note on DeepLabCut Config**: The `find_pt.py` script (Step 1) automatically updates the DeepLabCut `config.yaml` file with the correct paths for your computer. No manual editing required!

**3. Convert DNG to PNG**

The downloaded images are in `.dng` (raw) format. Convert them to PNG:

```bash
cd source/00_prepare_data
python convert_dng_to_png.py
```

This will process all population folders in `data/DNG/` and create corresponding folders in `data/PNG/`:
- Input:  `data/DNG/population_XX/image.dng`
- Output: `data/PNG/population_XX/image.png`

You can also specify custom input/output directories:
```bash
python convert_dng_to_png.py --input_dir /path/to/dng --output_dir /path/to/png
```

**Note**: The script processes all population folders automatically, maintaining the folder structure. PNG naming convention:
- Reflected light: `population_XX+FMNH_XXXXXX+stack_0.png`
- Transmitted light: `population_XX+FMNH_XXXXXX+stack_1.png`

---

## Project Structure

```
arphia_conspersa/
├── README.md
├── requirements.txt
├── data/                    # Input images (converted PNG)
├── result/                  # Pipeline outputs
└── source/                  # Source code
    ├── 00_prepare_data/     # DNG to PNG conversion
    ├── 01_find_pt/          # DeepLabCut keypoint detection
    ├── 02_extraction/       # SAM wing extraction
    ├── 03_svd/              # Image alignment & SVD
    ├── 04_segmentation/     # Cellpose domain segmentation
    ├── 05_venation_network/ # Graph analysis
    └── sknw-master/         # Graph skeletonization library
```

---

## Troubleshooting

### GPU Out of Memory
```bash
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python script.py
```

### TensorFlow Version
Must use TensorFlow 2.10.0 for DeepLabCut 2.2.3 compatibility. Do not upgrade.

### Python Version
Use Python 3.10. Some packages may fail with 3.11+.

### CUDA Library Warnings
Warnings like `libnvinfer.so.7: cannot open shared object` are typically non-critical.

### Directory Structure and Data Flow
All pipeline steps now read from and write to the `result/` directory:
```
arphia_conspersa/
├── data/                           # Input images (created by setup.sh)
├── result/                         # All pipeline outputs
│   ├── 01_find_pt/                # DeepLabCut keypoint detection outputs
│   │   ├── resize_img/            # Resized images
│   │   └── 01_output.csv          # Detected keypoints
│   ├── 02_extraction/             # SAM wing extraction outputs
│   │   └── population_XX/
│   │       └── perfect_cropped/   # Cropped wing images
│   ├── 03_svd/                    # SVD alignment outputs
│   │   └── population_XX/
│   │       └── *_hw_1.png         # SVD processed images
│   ├── 04_segmentation/           # Cellpose segmentation outputs
│   │   └── population_XX/
│   │       ├── *_hw_seg.npy       # Segmentation masks
│   │       ├── *_hw_outline.png   # Outline images
│   │       └── *_hw_outline.npy   # Outline arrays
│   └── 05_venation_network/       # Network analysis outputs
│       └── population_XX/
│           ├── cell/              # Cell statistics and plots
│           └── vein/              # Vein network graphs and plots
└── source/                         # Source code
```

**Data Flow**:
- Step 01 reads from `data/`, writes to `result/01_find_pt/`
- Step 02 reads from `result/01_find_pt/` and `data/`, writes to `result/02_extraction/`
- Step 03 reads from `result/02_extraction/`, writes to `result/03_svd/`
- Step 04 reads from `result/03_svd/`, writes to `result/04_segmentation/`
- Step 05 reads from `result/03_svd/` and `result/04_segmentation/`, writes to `result/05_venation_network/`

### Missing Output Files
If you see "file not found" errors:
- Ensure all prerequisite steps completed successfully
- Check that output directories exist
- Verify file naming matches the expected pattern (e.g., `population_XX+FMNH_XXXXXX`)

### Environment Variables
- `CUDA_VISIBLE_DEVICES=0` - Select GPU
- `TF_FORCE_GPU_ALLOW_GROWTH=true` - Dynamic GPU memory allocation
- `CELLPOSE_LOCAL_MODELS_PATH=~/.cellpose/models` - Custom Cellpose models

---

## License

MIT License

---

## Contributing

Pull requests welcome. Please open an issue first to discuss proposed changes.
