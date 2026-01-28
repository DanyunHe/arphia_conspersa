#!/usr/bin/env python3
"""
Segment wing domains and veins using Cellpose.

Reads SVD images from Step 3 output (result/03_svd/) and produces segmentation
masks and outline images in result/04_segmentation/, organized by population.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import glob
import cv2
import argparse

from cellpose import models
from cellpose.io import imread
from cellpose import io
from cellpose import utils


def find_all_wings(input_dir):
    """
    Find all wing base names from Step 3 SVD output.

    Args:
        input_dir: Directory containing SVD output (e.g., ../../result/03_svd/)

    Returns:
        List of base names (e.g., ["population_60+FMNH_4602398", ...])
    """
    pattern = os.path.join(input_dir, "population_*", "*_hw_1.png")
    hw_files = glob.glob(pattern)

    if not hw_files:
        print(f"Warning: No SVD images found matching pattern: {pattern}")
        return []

    base_names = []
    for hw_file in hw_files:
        basename = os.path.basename(hw_file)
        # Remove "_hw_1.png" to get base name like "population_60+FMNH_4602398"
        base_name = basename.replace("_hw_1.png", "")
        base_names.append(base_name)

    return sorted(base_names)


def process_single_wing(base_name, input_dir, output_dir, model, diameter):
    """
    Run Cellpose segmentation on a single wing.

    Args:
        base_name: Wing identifier (e.g., "population_60+FMNH_4602398")
        input_dir: SVD output directory (e.g., ../../result/03_svd/)
        output_dir: Segmentation output directory (e.g., ../../result/04_segmentation/)
        model: Loaded CellposeModel instance
        diameter: Cell diameter for Cellpose

    Returns:
        True if successful, False otherwise
    """
    # Extract population from base name (e.g., "population_60" from "population_60+FMNH_4602398")
    population = base_name.split("+")[0]

    # Input paths
    svd_img_path = os.path.join(input_dir, population, f"{base_name}_hw_1.png")
    orig_img_path = os.path.join(
        os.path.dirname(os.path.abspath(input_dir)),
        "02_extraction", population, "perfect_cropped",
        f"{base_name}+stack_1_hw_crop.tif"
    )

    # Check inputs exist
    if not os.path.exists(svd_img_path):
        print(f"  Error: SVD image not found: {svd_img_path}")
        return False
    if not os.path.exists(orig_img_path):
        print(f"  Error: Original TIF not found: {orig_img_path}")
        return False

    # Output directory (organized by population)
    output_pop_dir = os.path.join(output_dir, population)
    os.makedirs(output_pop_dir, exist_ok=True)

    # Run Cellpose segmentation
    img = imread(svd_img_path)
    channels = [[0, 0]]
    masks, flows, styles = model.eval(img, diameter=diameter, channels=channels)

    # Save cellpose outputs (_seg.npy)
    seg_prefix = os.path.join(output_pop_dir, f"{base_name}_hw")
    io.masks_flows_to_seg(img, masks, flows, seg_prefix, channels)

    # Generate outline image using original TIF
    orig_img = cv2.imread(orig_img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)  # 255 is mask

    edge = utils.masks_to_edges(masks)
    result = orig_img / 255.0

    ny = len(result[:, 0])
    nx = len(result[0, :])

    for jj in range(0, ny):
        for ii in range(0, nx):
            if orig_img[jj, ii] == 255:
                result[jj, ii] = 0.5  # background
            elif masks[jj, ii] != 0:
                result[jj, ii] = 0  # cells
            else:
                result[jj, ii] = 1

            if edge[jj, ii] != 0:
                result[jj, ii] = 1  # edges

    # Save outline outputs
    outline_png_path = os.path.join(output_pop_dir, f"{base_name}_hw_outline.png")
    outline_npy_path = os.path.join(output_pop_dir, f"{base_name}_hw_outline")
    plt.imsave(outline_png_path, result)
    np.save(outline_npy_path, result)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Segment wing domains and veins using Cellpose"
    )
    parser.add_argument("--input_dir", default="../../result/03_svd/",
                        help="SVD output directory (default: ../../result/03_svd/)")
    parser.add_argument("--output_dir", default="../../result/04_segmentation/",
                        help="Segmentation output directory (default: ../../result/04_segmentation/)")
    parser.add_argument("--wing_name", type=str,
                        help="Process only this wing (e.g., population_34+FMNH_4669630)")
    parser.add_argument("--population_id", type=int,
                        help="Process only this population (e.g., 34)")
    parser.add_argument("--model_type", type=str, default="CP_20230503_151910",
                        help="Cellpose model type (default: CP_20230503_151910)")
    parser.add_argument("--diameter", type=int, default=100,
                        help="Cell diameter for Cellpose (default: 100)")
    parser.add_argument("--cellpose_models_path", type=str, default=None,
                        help="Path to Cellpose models directory (default: ~/.cellpose/models)")
    args = parser.parse_args()

    # Set cellpose model path
    if args.cellpose_models_path:
        os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = args.cellpose_models_path
    elif "CELLPOSE_LOCAL_MODELS_PATH" not in os.environ:
        os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.path.expanduser("~/.cellpose/models")

    print(f"Input directory: {os.path.abspath(args.input_dir)}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")

    # Discover wings from Step 3 output
    all_wings = find_all_wings(args.input_dir)

    if not all_wings:
        print(f"Error: No wings found in {args.input_dir}")
        print("Please run Step 3 (process_all_svd.py) first.")
        sys.exit(1)

    # Filter by specific wing or population if requested
    if args.wing_name:
        if args.wing_name in all_wings:
            wings_to_process = [args.wing_name]
            print(f"\nProcessing specific wing: {args.wing_name}")
        else:
            print(f"Error: Wing '{args.wing_name}' not found in SVD output")
            print(f"Available wings: {all_wings[:5]}...")
            sys.exit(1)
    elif args.population_id is not None:
        pop_prefix = f"population_{args.population_id}+"
        wings_to_process = [w for w in all_wings if w.startswith(pop_prefix)]
        if not wings_to_process:
            print(f"Error: No wings found for population {args.population_id}")
            sys.exit(1)
        print(f"\nProcessing population {args.population_id}: {len(wings_to_process)} wing(s)")
    else:
        wings_to_process = all_wings
        print(f"\nFound {len(wings_to_process)} wing(s) to process")

    # Load Cellpose model once for all wings
    print(f"Loading Cellpose model: {args.model_type}")
    model = models.CellposeModel(model_type=args.model_type)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each wing
    successful = 0
    failed = 0
    failed_wings = []

    for i, wing_name in enumerate(wings_to_process, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(wings_to_process)}] Processing: {wing_name}")
        print(f"{'='*60}")

        try:
            if process_single_wing(wing_name, args.input_dir, args.output_dir,
                                   model, args.diameter):
                print(f"  Success: {wing_name}")
                successful += 1
            else:
                print(f"  Failed: {wing_name}")
                failed += 1
                failed_wings.append(wing_name)
        except Exception as e:
            print(f"  Failed: {wing_name} - {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            failed_wings.append(wing_name)

    # Summary
    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if failed_wings:
        print(f"Failed wings: {', '.join(failed_wings)}")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
