import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(dirname)
import argparse

import rawpy
import pandas as pd
import cv2
import torch
from ._extract import extract_wing_im
from ._crop import crop_wing_func




class WingExtractionPopulation:
    """
    Handles segmentation (SAM + flood fill) and cropping
    of insect wings for a single population.
    """

    def __init__(self, population_id, input_dir_csv, input_dir_img, save_dir):
        self.population_id = population_id
        self.input_dir_csv = input_dir_csv
        self.input_dir_img = input_dir_img
        self.save_dir = os.path.join(save_dir, f"population_{population_id}")

        # Load metadata table (drop dummy first row)
        self.population_excel = pd.read_csv(
            os.path.join(input_dir_csv, f"whole_label_{population_id}.csv"),
            header=1
        ).iloc[1:].reset_index(drop=True)

        self.n_wing = len(self.population_excel)

        # Make output dirs
        self.save_dir_perfect_full = os.path.join(self.save_dir, "perfect_full")
        self.save_dir_missing_full = os.path.join(self.save_dir, "missing_full")
        self.save_dir_perfect_cropped = os.path.join(self.save_dir, "perfect_cropped")
        self.save_dir_missing_cropped = os.path.join(self.save_dir, "missing_cropped")
        for d in [
            self.save_dir_perfect_full,
            self.save_dir_missing_full,
            self.save_dir_perfect_cropped,
            self.save_dir_missing_cropped,
        ]:
            os.makedirs(d, exist_ok=True)

    def _load_image_pair(self, w_name_base):
        """Read the two stack images for a given wing base name."""
        w_name_0 = f"{w_name_base}_0"
        w_name_1 = f"{w_name_base}_1"
        with rawpy.imread(os.path.join(self.input_dir_img, f"{w_name_0}.dng")) as raw:
            im0 = raw.postprocess()
        with rawpy.imread(os.path.join(self.input_dir_img, f"{w_name_1}.dng")) as raw:
            im1 = raw.postprocess()
        return w_name_0, im0, w_name_1, im1

    def _coords_for_row(self, row):
        """Extract coordinate tuples from a metadata row."""
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

    def process_population(self, predictor, start_index=0):
        """
        Run SAM + flood fill segmentation for all wings in the population.

        Args:
            predictor: A SamPredictor object.
            start_index: Index to start processing from (default 0).
        """
        for wi in range(start_index, self.n_wing):
            row = self.population_excel.iloc[wi]
            w_name_base = row["bodyparts"][:-6]  # strip suffix
            print(f"{wi+1}/{self.n_wing}: {w_name_base}_0.dng")

            w_name_0, im0, w_name_1, im1 = self._load_image_pair(w_name_base)
            coords = self._coords_for_row(row)

            # Stack 0
            extract_wing_im(
                predictor, im0, 0, w_name_0,
                coords["bg"][0], coords["bg"][1],
                *coords["hw"][0], *coords["fw"][0],
                *coords["hw"][1], *coords["fw"][1],
                *coords["body"][0], *coords["body"][1], *coords["body"][2],
                out_dir=self.save_dir
            )
            # Stack 1
            extract_wing_im(
                predictor, im1, 1, w_name_1,
                coords["bg"][0], coords["bg"][1],
                *coords["hw"][0], *coords["fw"][0],
                *coords["hw"][1], *coords["fw"][1],
                *coords["body"][0], *coords["body"][1], *coords["body"][2],
                out_dir=self.save_dir
            )

    def crop_population(self, category="perfect"):
        """
        Batch crop masks and save cropped versions.

        Args:
            category: "perfect", "missing", or "sticky".
        """
        assert category in ["perfect", "missing", "sticky"], "Invalid category"

        full_dir = os.path.join(self.save_dir, f"{category}_full")
        crop_dir = os.path.join(self.save_dir, f"{category}_cropped")
        os.makedirs(crop_dir, exist_ok=True)

        files = sorted(os.listdir(full_dir))
        num_wings = len(files) // 4 if category != "missing" else len(files) // 2

        for i in range(num_wings):
            print(f"{category}: {i+1}/{num_wings}")

            if category == "missing":
                w0 = cv2.imread(os.path.join(full_dir, files[2*i]), -1)
                w1 = cv2.imread(os.path.join(full_dir, files[2*i+1]), -1)
                crop_wing_func(w0, w1, crop_dir, files[2*i], files[2*i+1])
            else:  # perfect or sticky
                fw0 = cv2.imread(os.path.join(full_dir, files[4*i]), -1)
                hw0 = cv2.imread(os.path.join(full_dir, files[4*i+1]), -1)
                fw1 = cv2.imread(os.path.join(full_dir, files[4*i+2]), -1)
                hw1 = cv2.imread(os.path.join(full_dir, files[4*i+3]), -1)
                crop_wing_func(fw0, fw1, crop_dir, files[4*i], files[4*i+2])
                crop_wing_func(hw0, hw1, crop_dir, files[4*i+1], files[4*i+3])


if __name__ == '__main__':
    from segment_anything import sam_model_registry, SamPredictor

    # Load SAM
    sam = sam_model_registry["vit_h"](checkpoint="fine_tuned_sam_im1b.pth")
    sam.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    predictor = SamPredictor(sam)

    # Run pipeline
    pop = WingExtractionPopulation(
        population_id=44,
        input_dir_csv="01_find_pt_output/", #excel location
        input_dir_img="../../data/",
        save_dir="/content/drive/My Drive/wing/wing_new_populations/images_masks_population_44"
    )
    pop.process_population(predictor, start_index=32)
    pop.crop_population(category="perfect")
