import os, sys, argparse, glob
import rawpy, pandas as pd, cv2, torch
from segment_anything import sam_model_registry, SamPredictor
from _extract import extract_wing_im
from _crop import crop_wing_func


class WingExtractionPopulation:
    def __init__(self, population_id, input_dir_csv, input_dir_img, save_dir):
        self.population_id = population_id
        self.input_dir_csv = input_dir_csv
        self.input_dir_img = input_dir_img
        self.save_dir = os.path.join(save_dir, f"population_{population_id}")

        # Load metadata
        self.df = pd.read_csv(
            os.path.join(input_dir_csv, f"whole_label_{population_id}.csv"),
            header=1
        ).iloc[1:].reset_index(drop=True)
        self.n_wing = len(self.df)

        # Prepare output dirs
        for d in ["perfect_full","missing_full","perfect_cropped","missing_cropped"]:
            os.makedirs(os.path.join(self.save_dir, d), exist_ok=True)

    # ---------- helpers ----------
    def _load_image_pair(self, w_name_base):
        w_name_0 = f"{w_name_base}_0"
        w_name_1 = f"{w_name_base}_1"
        with rawpy.imread(os.path.join(self.input_dir_img, f"{w_name_0}.dng")) as raw:
            im0 = raw.postprocess()
        with rawpy.imread(os.path.join(self.input_dir_img, f"{w_name_1}.dng")) as raw:
            im1 = raw.postprocess()
        return w_name_0, im0, w_name_1, im1

    def _coords_for_row(self, row):
        return dict(
            bg=(float(row["background"]), float(row["background.1"])),
            hw=[(float(row["hind1"]), float(row["hind1.1"])),
                (float(row["hind2"]), float(row["hind2.1"]))],
            fw=[(float(row["fore1"]), float(row["fore1.1"])),
                (float(row["fore2"]), float(row["fore2.1"]))],
            body=[(float(row["bodypart1"]), float(row["bodypart1.1"])),
                  (float(row["bodypart2"]), float(row["bodypart2.1"])),
                  (float(row["bodypart3"]), float(row["bodypart3.1"]))],
        )

    # ---------- core logic ----------
    def process_individual(self, predictor, row_idx):
        """Process exactly one individual (row index)."""
        row = self.df.iloc[row_idx]
        w_name_base = row["bodyparts"][:-6]
        w_name_0 = f"{w_name_base}_0.dng"
        w_name_1 = f"{w_name_base}_1.dng"

        # ----- Skip if either image file is missing -----
        f0 = os.path.join(self.input_dir_img, w_name_0)
        f1 = os.path.join(self.input_dir_img, w_name_1)
        if not (os.path.exists(f0) and os.path.exists(f1)):
            print(f"[Skip] {row_idx}: missing image(s) {w_name_0} or {w_name_1}")
            return
        # -----------------------------------------------

        print(f"Individual {row_idx}: {w_name_0}")
        w_name_0, im0, w_name_1, im1 = self._load_image_pair(w_name_base)
        coords = self._coords_for_row(row)

        for idx, (nm, im) in enumerate([(w_name_0, im0), (w_name_1, im1)]):
            extract_wing_im(
                predictor, im, idx, nm,
                coords["bg"][0], coords["bg"][1],
                *coords["hw"][0], *coords["fw"][0],
                *coords["hw"][1], *coords["fw"][1],
                *coords["body"][0], *coords["body"][1], *coords["body"][2],
                out_dir=self.save_dir
            )


    def process_all(self, predictor, start_index=0):
        for i in range(start_index, self.n_wing):
            self.process_individual(predictor, i)

    def crop_all(self, category="perfect"):
        full_dir = os.path.join(self.save_dir, f"{category}_full")
        crop_dir = os.path.join(self.save_dir, f"{category}_cropped")
        os.makedirs(crop_dir, exist_ok=True)

        files = sorted(os.listdir(full_dir))
        num_wings = len(files)//4 if category!="missing" else len(files)//2
        for i in range(num_wings):
            print(f"{category}: {i+1}/{num_wings}")
            if category == "missing":
                w0 = cv2.imread(os.path.join(full_dir, files[2*i]), -1)
                w1 = cv2.imread(os.path.join(full_dir, files[2*i+1]), -1)
                crop_wing_func(w0, w1, crop_dir, files[2*i], files[2*i+1],
                                self.save_dir, self.population_id, f"{category}_cropped/")
            else:
                fw0 = cv2.imread(os.path.join(full_dir, files[4*i]), -1)
                hw0 = cv2.imread(os.path.join(full_dir, files[4*i+1]), -1)
                fw1 = cv2.imread(os.path.join(full_dir, files[4*i+2]), -1)
                hw1 = cv2.imread(os.path.join(full_dir, files[4*i+3]), -1)
                crop_wing_func(fw0, fw1, crop_dir, files[4*i], files[4*i+2],
                                self.save_dir, self.population_id, f"{category}_cropped/")
                crop_wing_func(hw0, hw1, crop_dir, files[4*i+1], files[4*i+3],
                                self.save_dir, self.population_id, f"{category}_cropped/")


def process_single_population(population_id, input_dir_csv, input_dir_img_base, save_dir, predictor):
    """
    Process a single population.

    Args:
        population_id: Population ID (integer, e.g., 34)
        input_dir_csv: Directory containing whole_label CSV files
        input_dir_img_base: Base directory for DNG images (data/DNG/)
        save_dir: Output directory
        predictor: SAM predictor instance
    """
    print(f"\n{'='*60}")
    print(f"Processing population {population_id}")
    print(f"{'='*60}")

    # Construct path to population DNG images
    input_dir_img = os.path.join(input_dir_img_base, f"population_{population_id}")

    if not os.path.exists(input_dir_img):
        print(f"  Warning: Image directory not found: {input_dir_img}")
        print(f"  Skipping population {population_id}")
        return

    # Check if CSV exists
    csv_path = os.path.join(input_dir_csv, f"whole_label_{population_id}.csv")
    if not os.path.exists(csv_path):
        print(f"  Warning: CSV not found: {csv_path}")
        print(f"  Skipping population {population_id}")
        return

    try:
        pop = WingExtractionPopulation(
            population_id, input_dir_csv, input_dir_img, save_dir
        )

        print(f"Found {pop.n_wing} individuals to process")
        pop.process_all(predictor, start_index=0)
        pop.crop_all(category="perfect")

        print(f"✓ Population {population_id} complete")
    except Exception as e:
        print(f"✗ Error processing population {population_id}: {e}")
        import traceback
        traceback.print_exc()


# -------------------- main -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract wings from grasshopper images using SAM model"
    )
    p.add_argument("--population_id", type=int,
                   help="Process specific population ID (e.g., 34). If not provided, processes all populations.")
    p.add_argument("--individual_index", type=int,
                   help="Process only this row index (0-based). Only works with --population_id.")
    p.add_argument("--individual_name",
                   help="Process only the row whose bodyparts starts with this name. Only works with --population_id.")
    p.add_argument("--start_index", type=int, default=0,
                   help="Start processing from this index (0-based). Only works with --population_id.")
    p.add_argument("--input_dir_csv", default="../../result/01_find_pt/",
                   help="Directory containing CSV files from Step 01 (default: ../../result/01_find_pt/)")
    p.add_argument("--input_dir_img", default="../../data/DNG/",
                   help="Base directory containing DNG image folders (default: ../../data/DNG/)")
    p.add_argument("--save_dir", default="../../result/02_extraction/",
                   help="Output directory for extracted wings (default: ../../result/02_extraction/)")
    args = p.parse_args()

    # Load SAM model once for all populations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check for fine-tuned model first, fall back to default model
    fine_tuned_path = "../../data/SAM_model/fine_tuned_sam_im1b.pth"
    default_path = "../../data/SAM_model/sam_vit_h_4b8939.pth"

    if os.path.exists(fine_tuned_path):
        checkpoint_path = fine_tuned_path
        print(f"Using fine-tuned SAM model: {checkpoint_path}")
    elif os.path.exists(default_path):
        checkpoint_path = default_path
        print(f"Using default SAM model: {checkpoint_path}")
    else:
        raise FileNotFoundError(
            f"No SAM model found. Please download either:\n"
            f"  - Fine-tuned model: {fine_tuned_path}\n"
            f"  - Default model: {default_path}\n"
            f"Run download_data.sh to download the default model."
        )

    # Load checkpoint with proper device mapping for CPU compatibility
    print("Loading SAM model...")
    if torch.cuda.is_available():
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    else:
        # For CPU-only: manually load state dict with map_location
        sam = sam_model_registry["vit_h"](checkpoint=None)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        sam.load_state_dict(state_dict)
    sam.to(device)
    predictor = SamPredictor(sam)
    print("SAM model loaded successfully\n")

    if args.population_id is not None:
        # Process single population
        if args.individual_index is not None or args.individual_name:
            # Process specific individual(s) - use original logic
            input_dir_img = os.path.join(args.input_dir_img, f"population_{args.population_id}")
            pop = WingExtractionPopulation(
                args.population_id, args.input_dir_csv, input_dir_img, args.save_dir
            )

            if args.individual_index is not None:
                pop.process_individual(predictor, args.individual_index)
            elif args.individual_name:
                idx = pop.df.index[pop.df["bodyparts"].str.startswith(args.individual_name)][0]
                pop.process_individual(predictor, idx)

            pop.crop_all(category="perfect")
        else:
            # Process all individuals in this population
            process_single_population(
                args.population_id, args.input_dir_csv, args.input_dir_img,
                args.save_dir, predictor
            )
    else:
        # Process all populations
        # Find all whole_label_*.csv files
        csv_files = glob.glob(os.path.join(args.input_dir_csv, "whole_label_*.csv"))

        if not csv_files:
            print(f"Error: No whole_label_*.csv files found in {args.input_dir_csv}")
            print("Please run Step 1 (find_pt.py) first.")
            sys.exit(1)

        # Extract population IDs from CSV filenames
        population_ids = []
        for csv_file in csv_files:
            basename = os.path.basename(csv_file)
            # Extract number from "whole_label_34.csv" -> 34
            pop_id = int(basename.replace("whole_label_", "").replace(".csv", ""))
            population_ids.append(pop_id)

        population_ids = sorted(population_ids)
        print(f"Found {len(population_ids)} population(s) to process: {population_ids}")

        for pop_id in population_ids:
            process_single_population(
                pop_id, args.input_dir_csv, args.input_dir_img,
                args.save_dir, predictor
            )

        print("\n" + "="*60)
        print("All populations processed!")
        print("="*60)
