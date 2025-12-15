# Grasshopper Wing Analysis Pipeline

This repository contains code for extracting and analyzing morphological information from grasshopper wing images (Arphia conspersa).

## Overview

The pipeline consists of 5 main steps:
1. **Find Points**: Identify key regions (background, forewing, hindwing, body) using DeepLabCut
2. **Wing Extraction**: Separate forewing and hindwing using Segment Anything Model (SAM)
3. **Alignment & SVD**: Align images and apply SVD to emphasize wing skeleton
4. **Domain Segmentation**: Segment wing domains and veins using Cellpose
5. **Venation Network**: Convert masks into graph structure with vertices and edges

---

## Requirements

- Python 3.10 (recommended) or 3.8+
- CUDA-capable GPU (recommended for deep learning models)
- 16GB+ RAM recommended

---

## Installation

### 1. Create a Virtual Environment

We strongly recommend using a virtual environment (conda or venv):

#### Using Conda (Recommended)
```bash
conda create -n wing python=3.10
conda activate wing
```

#### Using venv
```bash
python3 -m venv wing_env
source wing_env/bin/activate  # On Windows: wing_env\Scripts\activate
```

### 2. Install PyTorch

Install PyTorch first (required for SAM and Cellpose). Visit [pytorch.org](https://pytorch.org) for the appropriate command for your system.

#### For CUDA 11.8 (GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### For CPU only:
```bash
pip install torch torchvision
```

### 3. Clone the Repository

```bash
git clone <repository-url>
cd arphia_conspersa
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Segment Anything Model (SAM)

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 6. Install Local Package (sknw)

```bash
cd source/sknw-master
pip install -e .
cd ../..
```

### 7. Download Pre-trained Models

#### SAM Model Checkpoint
Download the SAM model checkpoint (required for step 02):
```bash
cd source/02_extraction
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../..
```

#### Cellpose Model (Optional)
If using a custom Cellpose model, place it in:
```bash
mkdir -p ~/.cellpose/models/
# Copy your custom model to ~/.cellpose/models/
```

---

## Quick Start

### Prepare Your Data

Place your raw images in the `data/` directory. The pipeline expects pairs of images:
- `<name>_0.dng` or `<name>_0.png` (reflected light)
- `<name>_1.dng` or `<name>_1.png` (transmitted light)

### Running the Pipeline

Navigate to the source directory:
```bash
cd source
```

#### Step 1: Find Key Points on Wings
```bash
python 01_find_pt/find_pt.py /path/to/arphia_conspersa population_XX+INDIVIDUAL_ID+stack_0.png
```

Example:
```bash
python 01_find_pt/find_pt.py /data/jiayin/arphia_conspersa population_34+FMNH_4669630+stack_0.png
```

**Output**: CSV file with point coordinates in `../result/01_find_pt/resize_img/`

#### Step 2: Extract Wings
```bash
python 02_extraction/extract_wings.py --population_id <population_id> --individual_index <index>
```

**Output**: Separated forewing and hindwing images

#### Step 3: Alignment and SVD
```bash
bash 03_svd/03_svd.sh <image_name>
```

Example:
```bash
bash 03_svd/03_svd.sh population_34+FMNH_4669526
```

**Output**: SVD-processed image in `../result/03_svd/<image_name>_hw_1.png`

#### Step 4: Domain Segmentation
```bash
python 04_segmentation/segmentation.py
```

**Output**: Domain masks and outline images

#### Step 5: Venation Network Analysis
```bash
python 05_venation_network/venation_network.py
```

**Output**: Graph structure with vertices and edges (thickness information included)

---

## Project Structure

```
arphia_conspersa/
├── README.md
├── requirements.txt
├── setup.sh                      # Installation script
├── data/                         # Input images
├── result/                       # Output results
│   ├── 01_find_pt/
│   ├── 02_extraction/
│   ├── 03_svd/
│   ├── 04_segmentation/
│   └── 05_venation_network/
└── source/                       # Source code
    ├── 01_find_pt/
    │   └── find_pt.py
    ├── 02_extraction/
    │   ├── extract_wings.py
    │   ├── _extract.py
    │   └── _crop.py
    ├── 03_svd/
    │   ├── 03_svd.sh
    │   ├── svd_img.py
    │   └── align_img/
    ├── 04_segmentation/
    │   └── segmentation.py
    ├── 05_venation_network/
    │   └── venation_network.py
    └── sknw-master/              # Graph skeletonization
```

---

## Troubleshooting

### GPU Out of Memory

If you encounter GPU memory errors, try:
```bash
# Limit to single GPU with memory growth
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python <script.py>
```

### TensorFlow/DeepLabCut Compatibility

This project requires TensorFlow 2.10.0 for compatibility with DeepLabCut 2.2.10. Do not upgrade TensorFlow beyond this version.

### Missing CUDA Libraries

If you see warnings about missing CUDA libraries (e.g., `libnvinfer.so.7`), these are typically non-critical for basic functionality.

### Python Version Issues

If you encounter compatibility issues, ensure you're using Python 3.10. Some packages may have issues with Python 3.11+.

---

## Environment Variables

Useful environment variables for running the pipeline:

- `CUDA_VISIBLE_DEVICES`: Select which GPU(s) to use (e.g., `CUDA_VISIBLE_DEVICES=0`)
- `TF_FORCE_GPU_ALLOW_GROWTH`: Allow TensorFlow to allocate GPU memory as needed
- `CELLPOSE_LOCAL_MODELS_PATH`: Path to custom Cellpose models

---

## Citation

If you use this code, please cite [appropriate paper/reference].

---

## License

This project is licensed under the MIT License.

---

## Contributing

Pull requests are welcome! Please open an issue first to discuss what you'd like to change.

---

## Contact

For questions or issues, please open a GitHub issue or contact [contact information].
