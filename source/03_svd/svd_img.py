import numpy as np
from PIL import Image
from skimage import io
import cv2
import tifffile as tif 
import matplotlib.pyplot as plt
from multi_svd import save_comp
import sys

if __name__=="__main__":

    # Read images
    fn="../../data/images/"
    file_name = sys.argv[1] # population_34+FMNH_4669526

    # Reflected image
    im_rl = cv2.imread(fn+file_name+'+stack_0_hw_crop_mapped.png')
    im_rl=im_rl.astype(np.float64);
    im_rl/=255.0
    im_rl=np.transpose(im_rl, (1, 0, 2))

    # Transmitted image 
    im_tl = cv2.imread(fn+file_name+'+stack_1_hw_crop.tif')
    im_tl=im_tl.astype(np.float64);
    im_tl/=255.0
    im_tl=np.transpose(im_tl, (1, 0, 2))
    print(im_tl.shape)

    # Save result 
    save_fn = "../../result/03_svd/"+file_name
    save_comp(save_fn,im_rl,im_tl)
