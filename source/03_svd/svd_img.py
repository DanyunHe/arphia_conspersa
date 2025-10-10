import numpy as np
from PIL import Image
from skimage import io
import cv2
import tifffile as tif 
import matplotlib.pyplot as plt
from multi_svd import save_comp

if __name__=="__main__":
    
    folder_name="./imgs/population_34/"
    
    file_name="population_34+FMNH_4669526"
    
     # Read images
    # Reflected image
    # im_rl = cv2.imread('./imgs/population_34/mapped.png')
    im_rl = cv2.imread(folder_name+file_name+'+stack_0_hw_crop_mapped.png')
    print(type(im_rl))
    #remove background
    # im_rl = remove(im_rl)
    # convert four channels (RGBA) to three channels (RGB)
    # im_rl=cv2.cvtColor(im_rl, cv2.COLOR_RGBA2RGB) 
    im_rl=im_rl.astype(np.float64);
    im_rl/=255.0
    im_rl=np.transpose(im_rl, (1, 0, 2))

    # Transmitted image 
    im_tl = cv2.imread(folder_name+file_name+'+stack_1_hw_crop.tif')
    # im_tl = cv2.imread('./align_img/target.png')

    #remove background
    # im_tl = remove(im_tl)
    # convert four channels (RGBA) to three channels (RGB)
    # im_tl=cv2.cvtColor(im_tl, cv2.COLOR_RGBA2RGB) 
    im_tl=im_tl.astype(np.float64);
    im_tl/=255.0
    im_tl=np.transpose(im_tl, (1, 0, 2))
    print(im_tl.shape)

    fn=folder_name+"svd_result/"+file_name
    save_comp(fn,im_rl,im_tl)
