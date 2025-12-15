import numpy as np
from PIL import Image
from skimage import io
import cv2
import tifffile as tif 
import matplotlib.pyplot as plt
import multi_svd
import sys

if __name__=="__main__":

    # Read images
    fn="../../data/images/"
    file_name = sys.argv[1] # population_34+FMNH_4669526

    # Reflected image
    im_rl = cv2.imread(fn+file_name+'+stack_0_hw_crop_mapped.png')
    im_rl=im_rl.astype(np.float32);im_rl/=255.0

    # Transmitted image 
    im_tl = cv2.imread(fn+file_name+'+stack_1_hw_crop.tif')
    im_tl=im_tl.astype(np.float64);im_tl/=255.0

    # Save result 
    save_fn = "../../result/03_svd/"+file_name
    multi_svd.save_comp(save_fn,im_rl,im_tl)

     # Make sure the background is black
    im_target=cv2.imread(save_fn+"_hw_1.png")
    multi_svd.dark_bkg(fn,im_rl,im_target)
