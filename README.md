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

# 3. Run automated setup
bash setup.sh
```

The `setup.sh` script will:
- Check Python version
- Install PyTorch (with CUDA support if available)
- Install all dependencies
- Install SAM and local packages
- Create output directories
- Verify installation

**Note**: Pre-trained models must be downloaded manually (see Manual Installation step 6)

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

# 4. Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# 5. Install local sknw package
cd source/sknw-master && pip install -e . && cd ../..

# 6. Download pre-trained models
# Download and place models in the appropriate directories:

# SAM model (fine-tuned for wings)
# Download from: https://ucla.box.com/s/pvdivl59ttxjg50f9s2pt8hu9z972jbn
# Place in: source/02_extraction/fine_tuned_sam_im1b.pth

# Cellpose model (for domain segmentation)
# Download from: https://drive.google.com/drive/folders/1KuuNEO-jhqwLQLR2-17uaZi7A8OlvBex?usp=drive_link
# Place in: ~/.cellpose/models/

# DeepLabCut model (for keypoint detection)
# Download from: https://drive.google.com/drive/folders/1pfFGtV4hHQSBNwLjq10zhs3KKi13phm0?usp=drive_link
# Place in: data/deeplabcut_whole/

# 7. Verify installation
python -c "import torch, tensorflow, deeplabcut, cellpose, segment_anything; print('Success!')"
```

---

## Usage

See [`source/README.md`](source/README.md) for detailed command-line usage of each step.

### Quick Example

```bash
cd source

# Step 1: Find points
python 01_find_pt/find_pt.py /path/to/project population_34+FMNH_4669630+stack_0.png

# Step 2: Extract wings
python 02_extraction/extract_wings.py --population_id 34 --individual_index 0

# Step 3: Alignment & SVD
bash 03_svd/03_svd.sh population_34+FMNH_4669526

# Step 4: Segment domains
python 04_segmentation/segmentation.py --folder_name ../result/03_svd/ --file_name population_34+FMNH_4669526

# Step 5: Venation network
python 05_venation_network/venation_network.py \
  --input_dir ../result/04_segmentation \
  --output_dir ../result/05_venation_network \
  --population 34 --species 4601939
```

### Data Preparation

**1. Create Data Directory**

```bash
mkdir -p data
```

**2. Download Images**

Download wing images from: https://drive.google.com/drive/folders/1lRfwuUhhVadkfz2ixvTbiqTruDvx72Kq?usp=drive_link

**3. Convert DNG to PNG**

The downloaded images are in `.dng` (raw) format. Convert them to PNG:

```bash
cd source/00_prepare_data
python convert_dng_to_png.py --input_dir ~/Downloads/wing_images --output_dir ../../data
```

This creates PNG files in `data/` with the naming convention:
- Reflected light: `population_XX+FMNH_XXXXXX+stack_0.png`
- Transmitted light: `population_XX+FMNH_XXXXXX+stack_1.png`

**Note**: Some pipeline steps (e.g., Step 02 extraction) can work directly with `.dng` files if you place them in the `data/` directory. However, Step 01 (find_pt.py) requires PNG format.

---

## Project Structure

```
arphia_conspersa/
├── README.md
├── requirements.txt
├── setup.sh                 # Automated installation
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
│   │   └── *_hw_1.png             # SVD processed images
│   ├── 04_segmentation/           # Cellpose segmentation outputs
│   │   ├── *_seg.npy              # Segmentation masks
│   │   └── *_hw_outline.png       # Outline images
│   └── 05_venation_network/       # Network analysis outputs
└── source/                         # Source code
```

**Data Flow**:
- Step 01 reads from `data/`, writes to `result/01_find_pt/`
- Step 02 reads from `result/01_find_pt/` and `data/`, writes to `result/02_extraction/`
- Step 03 reads from `result/02_extraction/`, writes to `result/03_svd/`
- Step 04 reads from `result/03_svd/`, writes to `result/04_segmentation/`
- Step 05 reads from `result/04_segmentation/`, writes to specified output directory

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
