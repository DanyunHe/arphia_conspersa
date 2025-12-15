"""
Convert .dng images to .png format for the wing analysis pipeline.

This script:
1. Reads .dng files from a source directory
2. Converts them to PNG using rawpy
3. Saves them in the data/ directory with consistent naming

Usage:
    python convert_dng_to_png.py --input_dir <path_to_dng_files> --output_dir <output_path>

Example:
    python convert_dng_to_png.py --input_dir ~/Downloads/wing_images --output_dir ../../data
"""

import os
import sys
import argparse
import glob
import rawpy
from PIL import Image
import numpy as np

def convert_dng_to_png(dng_path, output_path):
    """Convert a single .dng file to .png"""
    try:
        with rawpy.imread(dng_path) as raw:
            # Post-process the raw image
            rgb = raw.postprocess()

        # Convert to PIL Image
        img = Image.fromarray(rgb)

        # Save as PNG
        img.save(output_path)
        print(f"✓ Converted: {os.path.basename(dng_path)} -> {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"✗ Failed to convert {os.path.basename(dng_path)}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert .dng images to .png for wing analysis')
    parser.add_argument('--input_dir', required=True, help='Directory containing .dng files')
    parser.add_argument('--output_dir', default='../../data', help='Output directory for .png files (default: ../../data)')
    args = parser.parse_args()

    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all .dng files
    dng_files = glob.glob(os.path.join(args.input_dir, '*.dng'))

    if not dng_files:
        print(f"No .dng files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(dng_files)} .dng files")
    print(f"Converting to: {os.path.abspath(args.output_dir)}")
    print("-" * 60)

    # Convert each file
    success_count = 0
    fail_count = 0

    for dng_file in sorted(dng_files):
        # Get base name and replace extension
        base_name = os.path.basename(dng_file)
        png_name = base_name.replace('.dng', '.png').replace('.DNG', '.png')
        output_path = os.path.join(args.output_dir, png_name)

        # Convert
        if convert_dng_to_png(dng_file, output_path):
            success_count += 1
        else:
            fail_count += 1

    print("-" * 60)
    print(f"Conversion complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"\nPNG files saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
