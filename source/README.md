# Source Code - Pipeline Steps

Run each step from the `source/` directory.

---

## Step 00: Prepare Data

**Goal**: Convert raw .dng images to PNG format for the pipeline

**Download Images**: https://drive.google.com/drive/folders/1lRfwuUhhVadkfz2ixvTbiqTruDvx72Kq?usp=drive_link

**Command**:
```bash
cd 00_prepare_data
python convert_dng_to_png.py --input_dir <dng_directory> --output_dir ../../data
```

**Example**:
```bash
python convert_dng_to_png.py --input_dir ~/Downloads/wing_images --output_dir ../../data
```

**Output**: PNG images in `../../data/` with naming `population_XX+FMNH_XXXXXX+stack_[0,1].png`

---

## Step 01: Find Points

**Goal**: Identify key regions (background, forewing, hindwing, body) using DeepLabCut

**Command**:
```bash
cd 01_find_pt
python find_pt.py <working_dir> <image_filename>
```

**Example**:
```bash
python find_pt.py /data/jiayin/arphia_conspersa population_34+FMNH_4669630+stack_0.png
```

**Output**: CSV with point coordinates in `../../result/01_find_pt/resize_img/`

---

## Step 02: Wing Extraction

**Goal**: Separate forewing and hindwing using SAM

**Command**:
```bash
cd 02_extraction
python extract_wings.py --population_id <id> --individual_index <idx>
```

**Options**:
- `--population_id`: Population ID (required)
- `--individual_index`: Row index to process (0-based, optional)
- `--individual_name`: Process by name prefix (optional)
- `--start_index`: Start from this index (default: 0)
- `--input_dir_csv`: CSV directory (default: `../../result/01_find_pt/`)
- `--input_dir_img`: Image directory (default: `../../data/`)
- `--save_dir`: Output directory (default: `../../result/02_extraction/`)

**Example**:
```bash
python extract_wings.py --population_id 34 --individual_index 0
```

**Output**: Separated wings in `../../result/02_extraction/population_<id>/perfect_cropped/`

---

## Step 03: Alignment & SVD

**Goal**: Align transmitted/reflected images and apply SVD to enhance wing skeleton

**Command**:
```bash
cd 03_svd
bash 03_svd.sh <image_name>
```

**Example**:
```bash
bash 03_svd.sh population_34+FMNH_4669526
```

**Input**: Cropped wings from `../../result/02_extraction/`
**Output**: SVD images in `../../result/03_svd/<image_name>_hw_1.png`

---

## Step 04: Domain Segmentation

**Goal**: Segment wing domains and veins using Cellpose

**Command**:
```bash
cd 04_segmentation
python segmentation.py --folder_name <input_folder> --file_name <basename> --model_type <model>
```

**Options**:
- `--folder_name`: Input folder with SVD images (default: `../../result/03_svd/`)
- `--file_name`: Base filename without extension (e.g., `population_34+FMNH_4669526`)
- `--model_type`: Cellpose model name (default: `CP_20230503_151910`)
- `--diameter`: Cell diameter for segmentation (default: 100)
- `--cellpose_models_path`: Custom path to Cellpose models (default: `~/.cellpose/models`)

**Example**:
```bash
python segmentation.py --folder_name ../../result/03_svd/ --file_name population_34+FMNH_4669526
```

**Input**: SVD images from `../../result/03_svd/`
**Output**: Segmentation masks and outline images in `../../result/04_segmentation/`

---

## Step 05: Venation Network

**Goal**: Convert masks to graph structure with vertices and edges

**Command**:
```bash
cd 05_venation_network
python venation_network.py \
  --input_dir <input_dir> \
  --output_dir <output_dir> \
  --population <pop_id> \
  --species <species_id>
```

**Options**:
- `--save_venation_network`: Save network data (default: True)
- `--save_plot`: Save visualization plots (default: True)
- `--save_data`: Save processed data (default: True)

**Example**:
```bash
python venation_network.py \
  --input_dir 04_segmentation_output \
  --output_dir 05_venation_network_output \
  --population 151 \
  --species 4601939
```

**Input**: Segmentation results from `../../result/04_segmentation/`
**Output**: Graph structure, thickness data, and visualizations in specified output directory

---

## Environment Variables

For GPU memory management:
```bash
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python <script.py>
```

For custom Cellpose models:
```bash
export CELLPOSE_LOCAL_MODELS_PATH=~/.cellpose/models
```
