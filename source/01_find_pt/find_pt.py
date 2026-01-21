'''
Goal: Find the points for wing extraction
Input: All wing images in data/PNG/ organized by population folders
Output: Points for wing extraction, organized by population
'''

import deeplabcut as dlc
import pandas as pd
from PIL import Image
import sys
import os
import glob
import matplotlib.pyplot as plt


def process_single_image(working_dir, population_id, image_filename, config_path, population_output_dir, temp_dir):
    """
    Process a single wing image to find keypoints.

    Args:
        working_dir: Root directory of the project
        population_id: Population ID (e.g., "population_34")
        image_filename: Full filename (e.g., "population_34+FMNH_4669630+stack_1.png")
        config_path: Path to DeepLabCut config file
        population_output_dir: Output directory for this population (result/01_find_pt/population_XX/)
        temp_dir: Temporary directory for resized images
    """
    # Construct paths
    input_path = os.path.join(working_dir, 'data', 'PNG', population_id, image_filename)

    # Load and resize image
    image = Image.open(input_path)
    y1, x1 = image.size  # Original size
    y2, x2 = 1024, 682   # Target size
    resized_image = image.resize((y2, x2))

    # Clean temp directory to avoid DeepLabCut caching issues
    import glob as glob_module
    for old_file in glob_module.glob(os.path.join(temp_dir, '*')):
        os.remove(old_file)

    # Save resized image to temporary directory
    resized_path = os.path.join(temp_dir, image_filename)
    resized_image.save(resized_path)

    # Run DeepLabCut analysis on the resized image
    dlc.analyze_time_lapse_frames(config_path, temp_dir, save_as_csv=True)

    # Read DeepLabCut output (find the generated CSV file)
    csv_files = glob.glob(os.path.join(temp_dir, '*DLC_resnet50_segment_wholeJun6shuffle1_2000.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No DeepLabCut output CSV found in {temp_dir}")
    dlc_output_csv = csv_files[0]
    df = pd.read_csv(dlc_output_csv)

    # Scale coordinates back to original image size
    xfac = x1 / x2
    yfac = y1 / y2
    df2 = df.copy(deep=True)
    for i in range(8):
        df2.iloc[2, i*3+1] = int(float(df.iloc[2, i*3+1]) * xfac)
        df2.iloc[2, i*3+2] = int(float(df.iloc[2, i*3+2]) * yfac)

    # Create full identifier (e.g., "population_34+FMNH_4669630+stack_1")
    full_id = image_filename.replace('.png', '')

    # Save individual result CSV in population subdirectory
    individual_csv = os.path.join(population_output_dir, f'{full_id}_keypoints.csv')
    df2.to_csv(individual_csv, index=False)

    # Save validation plot in population subdirectory
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.scatter(df2.iloc[2, 1::3], df2.iloc[2, 2::3], s=50, c='red', marker='x')
    plt.title(f'{full_id}')
    plt.axis('off')
    validation_plot = os.path.join(population_output_dir, f'{full_id}_keypoints.png')
    plt.savefig(validation_plot, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"  ✓ Processed: {image_filename}")

    return df2


def process_population(working_dir, population_id, config_path, temp_dir):
    """
    Process all wing images for a single population.

    Args:
        working_dir: Root directory of the project
        population_id: Population ID (e.g., "population_34")
        config_path: Path to DeepLabCut config file
        temp_dir: Temporary directory for resized images
    """
    print(f"\n{'='*60}")
    print(f"Processing {population_id}")
    print(f"{'='*60}")

    # Set up population output directory
    population_output_dir = os.path.join(working_dir, 'result', '01_find_pt', population_id)
    os.makedirs(population_output_dir, exist_ok=True)

    # Find all transmitted light images (stack_1.png) in this population folder
    png_dir = os.path.join(working_dir, 'data', 'PNG', population_id)
    image_files = sorted(glob.glob(os.path.join(png_dir, '*stack_1.png')))

    if not image_files:
        print(f"  No transmitted light images (*stack_1.png) found in {png_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Process each image
    all_results = []
    for image_path in image_files:
        image_filename = os.path.basename(image_path)
        try:
            df_result = process_single_image(working_dir, population_id, image_filename,
                                            config_path, population_output_dir, temp_dir)
            all_results.append(df_result)
        except Exception as e:
            print(f"  ✗ Failed to process {image_filename}: {e}")
            continue

    # Combine all results into a single CSV for the population
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Extract population number (e.g., "34" from "population_34")
        pop_num = population_id.replace('population_', '')
        combined_csv = os.path.join(working_dir, 'result', '01_find_pt', f'whole_label_{pop_num}.csv')
        combined_df.to_csv(combined_csv, index=False)
        print(f"\n✓ Combined results saved: whole_label_{pop_num}.csv")
        print(f"✓ Individual results saved in: result/01_find_pt/{population_id}/")


def update_config_paths(config_path, working_dir):
    """
    Update paths in DeepLabCut config.yaml to match current working directory.

    This ensures the config works on different computers without manual editing.
    """
    import yaml

    # Read current config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update project_path
    old_project_path = config.get('project_path', '')
    new_project_path = os.path.join(working_dir, 'data', 'deeplabcut_whole',
                                     'segment_whole-dy-2024-06-06')

    if old_project_path != new_project_path:
        print(f"Updating DeepLabCut config for this computer...")
        config['project_path'] = new_project_path

        # Update video_sets paths
        if 'video_sets' in config and config['video_sets']:
            old_video_sets = config['video_sets'].copy()
            config['video_sets'] = {}
            for old_path, settings in old_video_sets.items():
                new_path = os.path.join(new_project_path, 'videos', 'whole.mov')
                config['video_sets'][new_path] = settings

        # Write updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Config updated: {new_project_path}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python find_pt.py <working_directory> [population_id]")
        print("\nExamples:")
        print("  # Process all populations:")
        print("  python find_pt.py /home/user/arphia_conspersa")
        print("\n  # Process specific population:")
        print("  python find_pt.py /home/user/arphia_conspersa population_34")
        sys.exit(1)

    working_dir = sys.argv[1]

    # DeepLabCut config path
    config_path = os.path.join(working_dir, 'data', 'deeplabcut_whole',
                               'segment_whole-dy-2024-06-06', 'config.yaml')

    if not os.path.exists(config_path):
        print(f"Error: DeepLabCut config not found at {config_path}")
        sys.exit(1)

    # Automatically update config paths for this computer
    update_config_paths(config_path, working_dir)

    # Create main output directory
    os.makedirs(os.path.join(working_dir, 'result', '01_find_pt'), exist_ok=True)

    # Create temporary directory for resized images (will be deleted after processing)
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp(prefix='find_pt_temp_')

    try:
        # Check if specific population was provided
        if len(sys.argv) >= 3:
            # Process single population
            population_id = sys.argv[2]
            if not population_id.startswith('population_'):
                population_id = f'population_{population_id}'

            population_dir = os.path.join(working_dir, 'data', 'PNG', population_id)
            if not os.path.exists(population_dir):
                print(f"Error: Population directory not found: {population_dir}")
                sys.exit(1)

            process_population(working_dir, population_id, config_path, temp_dir)
        else:
            # Process all populations
            png_base_dir = os.path.join(working_dir, 'data', 'PNG')
            population_dirs = sorted(glob.glob(os.path.join(png_base_dir, 'population_*')))

            if not population_dirs:
                print(f"Error: No population folders found in {png_base_dir}")
                sys.exit(1)

            print(f"\nFound {len(population_dirs)} population(s) to process")

            for pop_dir in population_dirs:
                population_id = os.path.basename(pop_dir)
                process_population(working_dir, population_id, config_path, temp_dir)

        print("\n" + "="*60)
        print("Processing complete!")
        print("="*60)
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary files") 