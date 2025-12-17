import numpy as np
from PIL import Image
from skimage import io
import cv2
import tifffile as tif 
import matplotlib.pyplot as plt
import multi_svd
import sys

if __name__=="__main__":

    # Read aligned images from result/03_svd/ (output from alignment step)
    # Input: Aligned images from result/03_svd/
    # Output: SVD processed images to result/03_svd/
    fn="../../result/03_svd/"
    file_name = sys.argv[1] # population_34+FMNH_4669526

    # Reflected image (after alignment/mapping)
    im_rl = cv2.imread(fn+file_name+'+stack_0_hw_crop_mapped.png')
    im_rl=im_rl.astype(np.float32);im_rl/=255.0

    # Transmitted image (original cropped from Step 02)
    # Try result/03_svd first, then fall back to Step 02 output
    im_tl_path = fn+file_name+'+stack_1_hw_crop.tif'
    if not os.path.exists(im_tl_path):
        # Fall back to Step 02 output location
        import glob
        possible_paths = glob.glob(f"../../result/02_extraction/population_*/perfect_cropped/{file_name}+stack_1_hw_crop.tif")
        if possible_paths:
            im_tl_path = possible_paths[0]

    im_tl = cv2.imread(im_tl_path)
    im_tl=im_tl.astype(np.float64);im_tl/=255.0

    # Save result to result/03_svd/
    save_fn = "../../result/03_svd/"+file_name
    multi_svd.save_comp(save_fn,im_rl,im_tl)

    # Make sure the background is black
    im_target=cv2.imread(save_fn+"_hw_1.png")
    multi_svd.dark_bkg(fn,im_rl,im_target)
