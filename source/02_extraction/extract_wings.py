import os
import sys
import argparse
import glob
import rawpy
import pandas as pd
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
from _extract import extract_wing_im
from _crop import crop_wing_func


class WingExtractionPopulation:
    """
    Handles segmentation (SAM + flood fill) and cropping
    of insect wings for one or multiple species within a population.
    """

    def __init__(self, population_id, input_dir_csv, input_dir_img, save_dir):
        self.population_id = population_id
        self.input_dir_csv = input_dir_csv
        self.input_dir_img = input_dir_img
        self.save_dir = os.path.join(save_dir, f"population_{population_id}")

    # ---------- helpers -------------------------------------------------------
    def _load_table(self, pop_id):
        df = pd.read_csv(
            os.path.join(self.input_dir_csv, f"whole_label_{pop_id}.csv"),
            header=1
        ).iloc[1:].reset_index(drop=True)
        return df

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

    # ---------- main pipeline -------------------------------------------------
    def process_species(self, predictor, species_id, start_index=0):
        """Segment and crop one species (single CSV)."""
        print(f"\n=== Processing species {species_id} ===")
        df = self._load_table(species_id)
        n_wing = len(df)

        # Make output dirs for this species
        save_dir_sp = os.path.join(self.save_dir, f"species_{species_id}")
        dirs = [
            "perfect_full", "missing_full",
            "perfect_cropped", "missing_cropped"
        ]
        for d in dirs:
            os.makedirs(os.path.join(save_dir_sp, d), exist_ok=True)

        for wi in range(start_index, n_wing):
            row = df.iloc[wi]
            w_name_base = row["bodyparts"][:-6]
            print(f"{wi+1}/{n_wing}: {w_name_base}_0.dng")

            w_name_0, im0, w_name_1, im1 = self._load_image_pair(w_name_base)
            coords = self._coords_for_row(row)

            for idx, (nm, im) in enumerate([(w_name_0, im0), (w_name_1, im1)]):
                extract_wing_im(
                    predictor, im, idx, nm,
                    coords["bg"][0], coords["bg"][1],
                    *coords["hw"][0], *coords["fw"][0],
                    *coords["hw"][1], *coords["fw"][1],
                    *coords["body"][0], *coords["body"][1], *coords["body"][2],
                    out_dir=save_dir_sp
                )

        # Crop after segmentation
        self.crop_species(save_dir_sp, category="perfect")

    def crop_species(self, save_dir_sp, category="perfect"):
        assert category in ["perfect", "missing", "sticky"], "Invalid category"
        full_dir = os.path.join(save_dir_sp, f"{category}_full")
        crop_dir = os.path.join(save_dir_sp, f"{category}_cropped")
        os.makedirs(crop_dir, exist_ok=True)

        files = sorted(os.listdir(full_dir))
        num_wings = len(files) // 4 if category != "missing" else len(files) // 2

        for i in range(num_wings):
            print(f"{category}: {i+1}/{num_wings}")
            if category == "missing":
                w0 = cv2.imread(os.path.join(full_dir, files[2*i]), -1)
                w1 = cv2.imread(os.path.join(full_dir, files[2*i+1]), -1)
                crop_wing_func(w0, w1, crop_dir, files[2*i], files[2*i+1])
            else:
                fw0 = cv2.imread(os.path.join(full_dir, files[4*i]), -1)
                hw0 = cv2.imread(os.path.join(full_dir, files[4*i+1]), -1)
                fw1 = cv2.imread(os.path.join(full_dir, files[4*i+2]), -1)
                hw1 = cv2.imread(os.path.join(full_dir, files[4*i+3]), -1)
                crop_wing_func(fw0, fw1, crop_dir, files[4*i], files[4*i+2])
                crop_wing_func(hw0, hw1, crop_dir, files[4*i+1], files[4*i+3])


# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--population_id", type=int, required=True)
    parser.add_argument("--species_id", type=int,
                        help="Run only this species ID")
    parser.add_argument("--all_species", action="store_true",
                        help="Run all species CSVs in input_dir_csv")
    parser.add_argument("--input_dir_csv", default="01_find_pt_output/")
    parser.add_argument("--input_dir_img", default="../../data/")
    parser.add_argument("--save_dir", default="02_extraction_output/")
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()

    # Load SAM
    sam = sam_model_registry["vit_h"](checkpoint="fine_tuned_sam_im1b.pth")
    sam.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    predictor = SamPredictor(sam)

    pop = WingExtractionPopulation(
        population_id=args.population_id,
        input_dir_csv=args.input_dir_csv,
        input_dir_img=args.input_dir_img,
        save_dir=args.save_dir
    )

    if args.all_species:
        # find all CSVs of the pattern whole_label_*.csv
        pattern = os.path.join(args.input_dir_csv, "whole_label_*.csv")
        for csvfile in sorted(glob.glob(pattern)):
            sp_id = int(os.path.basename(csvfile).split("_")[-1].split(".")[0])
            pop.process_species(predictor, sp_id, start_index=args.start_index)
    elif args.species_id is not None:
        pop.process_species(predictor, args.species_id, start_index=args.start_index)
    else:
        parser.error("Specify either --species_id or --all_species")
