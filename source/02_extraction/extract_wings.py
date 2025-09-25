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


# -------------------- main -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--population_id", type=int, required=True)
    p.add_argument("--individual_index", type=int,
                   help="Process only this row index (0-based).")
    p.add_argument("--individual_name",
                   help="Process only the row whose bodyparts starts with this name.")
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--input_dir_csv", default="01_find_pt_output/")
    p.add_argument("--input_dir_img", default="../../data/")
    p.add_argument("--save_dir", default="02_extraction_output/")
    args = p.parse_args()

    # # Load SAM
    # sam = sam_model_registry["vit_h"](checkpoint="fine_tuned_sam_im1b.pth")
    # sam.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # predictor = SamPredictor(sam)

    pop = WingExtractionPopulation(
        args.population_id, args.input_dir_csv, args.input_dir_img, args.save_dir
    )

    # if args.individual_index is not None:
    #     pop.process_individual(predictor, args.individual_index)
    # elif args.individual_name:
    #     idx = pop.df.index[pop.df["bodyparts"].str.startswith(args.individual_name)][0]
    #     pop.process_individual(predictor, idx)
    # else:
    #     pop.process_all(predictor, start_index=args.start_index)

    pop.crop_all(category="perfect")
