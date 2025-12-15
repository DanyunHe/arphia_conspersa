# Quick Installation Guide

## Automated Installation (Recommended)

The easiest way to set up the environment is to use the provided setup script:

```bash
# 1. Create and activate a virtual environment
conda create -n wing python=3.10
conda activate wing

# 2. Run the setup script
bash setup.sh
```

The script will:
- Check your Python version
- Install PyTorch (with CUDA support if available)
- Install all dependencies from requirements.txt
- Install Segment Anything Model (SAM)
- Install the local sknw package
- Download the SAM model checkpoint
- Create necessary directories
- Verify the installation

---

## Manual Installation

If you prefer to install manually or the script fails:

### 1. Create Virtual Environment
```bash
conda create -n wing python=3.10
conda activate wing
```

### 2. Install PyTorch
Visit [pytorch.org](https://pytorch.org) and install the appropriate version for your system.

For CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CPU only:
```bash
pip install torch torchvision
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install SAM
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 5. Install Local Package
```bash
cd source/sknw-master
pip install -e .
cd ../..
```

### 6. Download SAM Checkpoint
```bash
cd source/02_extraction
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../..
```

### 7. Create Directories
```bash
mkdir -p result/{01_find_pt,02_extraction,03_svd,04_segmentation,05_venation_network}
```

---

## Verification

Test your installation:

```bash
python -c "import torch, tensorflow, deeplabcut, cellpose, segment_anything; print('All packages imported successfully!')"
```

---

## Common Issues

### GPU Memory Errors
Set environment variables when running scripts:
```bash
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python script.py
```

### TensorFlow Version Conflict
Ensure TensorFlow 2.10.0 is installed (required for DeepLabCut 2.2.10):
```bash
pip install tensorflow==2.10.0
```

### Missing CUDA Libraries
These warnings are usually non-critical:
```
Could not load dynamic library 'libnvinfer.so.7'
```

---

## Next Steps

After installation, see [README.md](README.md) for detailed usage instructions.
