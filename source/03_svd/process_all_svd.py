#!/usr/bin/env python3
"""
Process all wings from Step 2 through the SVD alignment pipeline.

This script automatically discovers all wings extracted in Step 2 and processes
them through the alignment and SVD steps, organizing outputs by population.
"""

import os
import sys
import glob
import subprocess
import argparse


def find_all_wings(input_dir):
    """
    Find all unique wing base names from Step 2 output.

    Args:
        input_dir: Base directory containing Step 2 output (e.g., ../../result/02_extraction/)

    Returns:
        List of base names (e.g., ["population_34+FMNH_4669630", ...])
    """
    # Find all hindwing cropped files (stack_1_hw_crop.tif)
    # We use hindwing files to identify unique individuals
    pattern = os.path.join(input_dir, "population_*/perfect_cropped/*+stack_1_hw_crop.tif")
    hw_files = glob.glob(pattern)

    if not hw_files:
        print(f"Warning: No hindwing files found matching pattern: {pattern}")
        return []

    # Extract base names (remove +stack_1_hw_crop.tif)
    base_names = []
    for hw_file in hw_files:
        basename = os.path.basename(hw_file)
        # Remove "+stack_1_hw_crop.tif" to get base name like "population_34+FMNH_4669630"
        base_name = basename.replace("+stack_1_hw_crop.tif", "")
        base_names.append(base_name)

    return sorted(base_names)


def process_single_wing(base_name, script_dir):
    """
    Process a single wing through the SVD pipeline.

    Args:
        base_name: Base name like "population_34+FMNH_4669630"
        script_dir: Directory containing 03_svd.sh (source/03_svd/)

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Processing: {base_name}")
    print(f"{'='*60}")

    try:
        # The script expects to be run from source/ directory
        script_path = os.path.join(script_dir, "03_svd.sh")
        working_dir = os.path.dirname(script_dir)  # source/

        # Run the bash script
        result = subprocess.run(
            ["bash", "03_svd/03_svd.sh", base_name],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per wing
        )

        if result.returncode == 0:
            print(f"✓ {base_name} completed successfully")
            return True
        else:
            print(f"✗ {base_name} failed with return code {result.returncode}")
            if result.stdout:
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            if result.stderr:
                print("STDERR:", result.stderr[-500:])
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ {base_name} timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"✗ {base_name} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Process all wings from Step 2 through SVD alignment pipeline"
    )
    parser.add_argument("--input_dir", default="../../result/02_extraction/",
                       help="Input directory from Step 2 (default: ../../result/02_extraction/)")
    parser.add_argument("--wing_name", type=str,
                       help="Process only this specific wing (e.g., population_34+FMNH_4669630)")
    parser.add_argument("--population_id", type=int,
                       help="Process only this population (e.g., 34)")
    args = parser.parse_args()

    # Get the directory where this script is located (source/03_svd/)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Input directory: {os.path.abspath(args.input_dir)}")
    print(f"Script directory: {script_dir}")

    # Check if script exists
    script_path = os.path.join(script_dir, "03_svd.sh")
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)

    # Find all wings
    all_wings = find_all_wings(args.input_dir)

    if not all_wings:
        print(f"Error: No wings found in {args.input_dir}")
        print("Please run Step 2 (extract_wings.py) first.")
        sys.exit(1)

    # Filter by specific wing or population if requested
    if args.wing_name:
        if args.wing_name in all_wings:
            wings_to_process = [args.wing_name]
            print(f"\nProcessing specific wing: {args.wing_name}")
        else:
            print(f"Error: Wing '{args.wing_name}' not found in output")
            print(f"Available wings: {all_wings[:5]}...")
            sys.exit(1)
    elif args.population_id:
        pop_prefix = f"population_{args.population_id}+"
        wings_to_process = [w for w in all_wings if w.startswith(pop_prefix)]
        if not wings_to_process:
            print(f"Error: No wings found for population {args.population_id}")
            sys.exit(1)
        print(f"\nProcessing population {args.population_id}: {len(wings_to_process)} wing(s)")
    else:
        wings_to_process = all_wings
        print(f"\nFound {len(wings_to_process)} wing(s) to process")

    # Process each wing
    successful = 0
    failed = 0
    failed_wings = []

    for i, wing_name in enumerate(wings_to_process, 1):
        print(f"\n[{i}/{len(wings_to_process)}]")
        if process_single_wing(wing_name, script_dir):
            successful += 1
        else:
            failed += 1
            failed_wings.append(wing_name)

    # Summary
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if failed_wings:
        print(f"Failed wings: {', '.join(failed_wings)}")
    print("="*60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
