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

# Step 4: Segment domains (edit script first to set paths)
python 04_segmentation/segmentation.py

# Step 5: Venation network
python 05_venation_network/venation_network.py \
  --input_dir ../result/04_segmentation \
  --output_dir ../result/05_venation_network \
  --population 34 --species 4601939
```

### Data Preparation

Place image pairs in `data/`:
- `<name>_0.dng` or `<name>_0.png` (reflected light)
- `<name>_1.dng` or `<name>_1.png` (transmitted light)

---

## Project Structure

```
arphia_conspersa/
├── README.md
├── requirements.txt
├── setup.sh                 # Automated installation
├── data/                    # Input images
├── result/                  # Pipeline outputs
└── source/                  # Source code
    ├── 01_find_pt/
    ├── 02_extraction/
    ├── 03_svd/
    ├── 04_segmentation/
    ├── 05_venation_network/
    └── sknw-master/
```

---

## Troubleshooting

### GPU Out of Memory
```bash
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python script.py
```

### TensorFlow Version
Must use TensorFlow 2.10.0 for DeepLabCut 2.2.10 compatibility. Do not upgrade.

### Python Version
Use Python 3.10. Some packages may fail with 3.11+.

### CUDA Library Warnings
Warnings like `libnvinfer.so.7: cannot open shared object` are typically non-critical.

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
