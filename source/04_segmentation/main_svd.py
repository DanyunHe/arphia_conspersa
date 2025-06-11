import multi_svd 
import cv2
import numpy as np
import time

t0=time.time()
# read all cropped images in a folder and do svd 
folder_name="./images_align/select_perfect_cropped/"

with open(folder_name+"name_list.txt", 'r') as f:
    text = f.read()
    names = text.split()

for i in range(len(names)):
    file_name=names[i]
    print("processing: ", i, file_name)

    # Read images
    # Reflected image
    im_rl = cv2.imread(folder_name+file_name+'+stack_0_hw_crop_mapped.png')
    #remove background
    # im_rl = remove(im_rl)
    # convert four channels (RGBA) to three channels (RGB)
    # im_rl=cv2.cvtColor(im_rl, cv2.COLOR_RGBA2RGB) 
    im_rl=im_rl.astype(np.float64);im_rl/=255.0
    im_rl=np.transpose(im_rl, (1, 0, 2))

    # Transmitted image 
    im_tl = cv2.imread(folder_name+file_name+'+stack_1_hw_crop.tif')
    im_tl=im_tl.astype(np.float64);im_tl/=255.0
    im_tl=np.transpose(im_tl, (1, 0, 2))
    
    # Calculate the first component and save
    fn=folder_name+"svd_result/"+file_name
    multi_svd.save_comp(fn,im_rl,im_tl)

t1=time.time()   
print("total time: %f minuetes"%(t1-t0)/60.)